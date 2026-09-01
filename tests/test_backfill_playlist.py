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
