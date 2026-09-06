"""Backfill 'to-analyze' playlist automation (#127).

Pure-logic units (no HTTP) first: exclude-already-played (AC4), remove-on-
analyzed candidate filtering (AC3's selection half), the 10k cap (AC2), and
the opt-in gate (AC1's gate half). HTTP-mocked playlist sync (AC1/AC2/AC3's
Spotify-API half) follows in ``test_backfill_playlist_sync.py``.
"""

from __future__ import annotations

import threading

from music_intel_mcp.backfill_playlist import (
    DEFAULT_REFRESH_INTERVAL_S,
    MAX_BACKFILL_TRACKS,
    PlaylistDiff,
    diff_playlist_membership,
    is_backfill_enabled,
    played_track_ids,
    run_continuous_backfill,
    select_backfill_tracks,
)
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef


def _track(name: str, artist: str = "Artist", **kwargs) -> TrackRef:
    return TrackRef(name=name, artist=artist, **kwargs)


def _event(track: TrackRef) -> ListenEvent:
    return ListenEvent(
        track=track,
        played_at="2026-01-01T00:00:00Z",
        source="test",
        context=PlayContext(),
    )


# --- AC4: already-played tracks never enter the backfill queue ------------ #


def test_select_backfill_tracks_excludes_already_played():
    played = _track("Played Song", spotify_id="p1")
    unplayed = _track("New Song", spotify_id="u1")
    played_ids = played_track_ids([_event(played)])

    selected = select_backfill_tracks(
        [played, unplayed],
        played_ids=played_ids,
        has_audio_analysis=lambda _cid: False,
    )

    assert selected == [unplayed]


def test_select_backfill_tracks_excludes_already_played_even_when_unanalyzed():
    # AC4's exact wording: played + unanalyzed must still be excluded.
    track = _track("Both", spotify_id="x1")
    played_ids = played_track_ids([_event(track)])

    selected = select_backfill_tracks(
        [track],
        played_ids=played_ids,
        has_audio_analysis=lambda _cid: False,  # unanalyzed
    )

    assert selected == []


# --- AC3 (selection half): analyzed tracks are filtered from the queue ---- #


def test_select_backfill_tracks_excludes_already_analyzed():
    analyzed = _track("Analyzed", spotify_id="a1")
    unanalyzed = _track("Unanalyzed", spotify_id="a2")

    def has_analysis(cid: str) -> bool:
        return cid == "spotify:a1"

    selected = select_backfill_tracks(
        [analyzed, unanalyzed],
        played_ids=set(),
        has_audio_analysis=has_analysis,
    )

    assert selected == [unanalyzed]


# --- AC2: capped at 10,000 tracks ------------------------------------------ #


def test_select_backfill_tracks_caps_at_limit():
    candidates = [_track(f"T{i}", spotify_id=f"s{i}") for i in range(5)]

    selected = select_backfill_tracks(
        candidates,
        played_ids=set(),
        has_audio_analysis=lambda _cid: False,
        limit=3,
    )

    assert len(selected) == 3
    assert selected == candidates[:3]


def test_default_backfill_limit_is_10000():
    assert MAX_BACKFILL_TRACKS == 10_000


def test_select_backfill_tracks_excludes_tracks_with_no_spotify_id():
    # Spotify returns `id: null` for locally-added/unavailable saved tracks
    # (fetch_saved_track_refs propagates this as spotify_id=None). Such a
    # track can never be added to a playlist by uri, so it must never enter
    # the desired set -- otherwise spotify_track_uri() raises on its
    # name-keyed canonical id when the playlist sync tries to add it.
    unavailable = _track("Local Only", spotify_id=None)
    available = _track("Real Track", spotify_id="r1")

    selected = select_backfill_tracks(
        [unavailable, available],
        played_ids=set(),
        has_audio_analysis=lambda _cid: False,
    )

    assert selected == [available]


def test_select_backfill_tracks_resolve_mbid_bridges_isrc_to_analyzed_mbid():
    # fetch_saved_track_refs only ever knows spotify_id (+ now isrc, from
    # external_ids) for a saved-library candidate -- it never has the mbid
    # the live AcoustID pipeline writes its audio-analysis files under. If
    # nothing bridges isrc -> mbid, canonical_track_id(candidate) always
    # bottoms out at spotify:<id>, which never matches an mbid:-keyed
    # analysis file for the *same* recording (the bug the reviewer flagged).
    # resolve_mbid is that bridge, used only to compute the membership key
    # -- checked here via has_audio_analysis.
    candidate = _track("Around the World", spotify_id="s1", isrc="FR-Z03-97-00212")

    def has_analysis(cid: str) -> bool:
        return cid == "mbid:M-1"

    selected = select_backfill_tracks(
        [candidate],
        played_ids=set(),
        has_audio_analysis=has_analysis,
        resolve_mbid=lambda t: "M-1" if t.isrc == "FR-Z03-97-00212" else None,
    )

    assert selected == []


def test_select_backfill_tracks_resolve_mbid_does_not_mutate_returned_track():
    # The bridged mbid must only affect the membership check, never the
    # returned TrackRef -- resolve_mbid is consulted purely to compute the
    # membership key and must not be written back onto the track. Note that
    # canonical_track_id(selected_track) is NOT a safe way to build a Spotify
    # playlist uri here: the candidate's own isrc (set independently by
    # fetch_saved_track_refs) already makes canonical_track_id prefer
    # isrc:... over spotify:... per the identity waterfall (#158). Callers
    # that need a playlist uri must build it from track.spotify_id directly
    # (cli.py does this), never from canonical_track_id(track).
    candidate = _track("Around the World", spotify_id="s1", isrc="FR-Z03-97-00212")

    selected = select_backfill_tracks(
        [candidate],
        played_ids=set(),
        has_audio_analysis=lambda _cid: False,
        resolve_mbid=lambda t: "M-1" if t.isrc == "FR-Z03-97-00212" else None,
    )

    assert selected == [candidate]
    assert selected[0].mbid is None
    assert selected[0].spotify_id == "s1"


def test_select_backfill_tracks_resolve_mbid_excludes_already_played():
    # The same bridge must feed the AC4 played-ids check too: a track played
    # live (history event carries mbid) must exclude the matching backfill
    # candidate even though the candidate itself only has isrc/spotify_id.
    candidate = _track("Around the World", spotify_id="s1", isrc="FR-Z03-97-00212")

    selected = select_backfill_tracks(
        [candidate],
        played_ids={"mbid:M-1"},
        has_audio_analysis=lambda _cid: False,
        resolve_mbid=lambda t: "M-1" if t.isrc == "FR-Z03-97-00212" else None,
    )

    assert selected == []


def test_select_backfill_tracks_dedupes_repeated_candidates():
    track = _track("Dup", spotify_id="d1")

    selected = select_backfill_tracks(
        [track, track],
        played_ids=set(),
        has_audio_analysis=lambda _cid: False,
    )

    assert selected == [track]


# --- AC3 (diff half): playlist membership diff for daily refresh --------- #


def test_diff_playlist_membership_adds_new_and_removes_stale():
    diff = diff_playlist_membership(
        current_ids=["a", "b", "c"],
        desired_ids=["b", "c", "d"],
    )

    assert isinstance(diff, PlaylistDiff)
    assert diff.to_add == ["d"]
    assert diff.to_remove == ["a"]


def test_diff_playlist_membership_removes_analyzed_track_on_refresh():
    # #126's dedup check gates the desired set upstream; the diff itself must
    # surface "no longer desired" -> to_remove, which is what makes a
    # newly-analyzed track disappear on the next daily sync.
    diff = diff_playlist_membership(current_ids=["analyzed_now"], desired_ids=[])

    assert diff.to_remove == ["analyzed_now"]
    assert diff.to_add == []


def test_diff_playlist_membership_no_changes_when_membership_matches():
    diff = diff_playlist_membership(current_ids=["a", "b"], desired_ids=["b", "a"])

    assert diff.to_add == []
    assert diff.to_remove == []


# --- AC1 (gate half): opt-in env flag -------------------------------------- #


def test_backfill_disabled_by_default():
    assert is_backfill_enabled({}) is False


def test_backfill_enabled_via_explicit_opt_in():
    assert is_backfill_enabled({"MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED": "true"}) is True


def test_backfill_disabled_for_falsy_values():
    for value in ("0", "false", "no", ""):
        assert is_backfill_enabled({"MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED": value}) is False


# --- AC2 (cadence half): daily-refresh loop --------------------------------- #


def test_default_refresh_interval_is_daily():
    assert DEFAULT_REFRESH_INTERVAL_S == 24 * 60 * 60


def test_run_continuous_backfill_syncs_then_sleeps_then_stops():
    # sync_once runs once per cycle; the injected stop_event/sleep let the
    # loop be driven deterministically instead of actually sleeping a day.
    calls: list[str] = []
    stop_event = threading.Event()

    def sync_once() -> PlaylistDiff:
        calls.append("sync")
        return PlaylistDiff(to_add=[], to_remove=[])

    def fake_sleep(seconds: float) -> None:
        calls.append(f"sleep:{seconds}")
        stop_event.set()

    run_continuous_backfill(
        sync_once=sync_once,
        interval_s=99.0,
        stop_event=stop_event,
        sleep=fake_sleep,
    )

    assert calls == ["sync", "sleep:99.0"]


def test_run_continuous_backfill_reports_each_cycle_result():
    stop_event = threading.Event()
    results: list[PlaylistDiff] = []
    diff = PlaylistDiff(to_add=["a"], to_remove=[])

    def sync_once() -> PlaylistDiff:
        return diff

    def fake_sleep(_seconds: float) -> None:
        stop_event.set()

    run_continuous_backfill(
        sync_once=sync_once,
        stop_event=stop_event,
        sleep=fake_sleep,
        on_result=results.append,
    )

    assert results == [diff]


def test_run_continuous_backfill_reports_sync_errors_without_stopping():
    stop_event = threading.Event()
    errors: list[Exception] = []
    attempts = {"n": 0}

    def sync_once() -> PlaylistDiff:
        attempts["n"] += 1
        raise RuntimeError("boom")

    def fake_sleep(_seconds: float) -> None:
        stop_event.set()

    run_continuous_backfill(
        sync_once=sync_once,
        stop_event=stop_event,
        sleep=fake_sleep,
        on_error=errors.append,
    )

    assert attempts["n"] == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
