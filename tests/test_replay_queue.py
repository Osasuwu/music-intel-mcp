"""Replay queue selector (#163) — pilot slice 1.

Pure-logic units (no HTTP): valid-play counting against the ≥30 s rule
(AC1), pool-or-root analysis exclusion (AC2), artist/year stratification
with a cap (AC3). The CLI subcommand's coverage-stat printing (AC4) is
covered in ``test_cli_replay_queue.py``. AC5 (``select_backfill_tracks``
untouched) is the existing ``test_backfill_playlist.py`` suite staying
green — no new test needed here.
"""

from __future__ import annotations

import pytest

from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.replay_queue import (
    measure_coverage_ceiling,
    replay_queue_coverage,
    select_replay_queue,
    youtube_crosswalk_resolve_mbid,
)
from music_intel_mcp.shared_store import canonical_track_id


def _track(name: str, artist: str = "Artist", **kwargs) -> TrackRef:
    return TrackRef(name=name, artist=artist, **kwargs)


def _event(track: TrackRef, *, played_at: str, ms_played: int | None = None) -> ListenEvent:
    return ListenEvent(
        track=track,
        played_at=played_at,
        source="test",
        context=PlayContext(ms_played=ms_played),
    )


# --- AC1: play counting uses the >=30s validity rule --------------------- #


def test_three_short_plays_are_not_valid_and_track_is_not_queued():
    track = _track("Skip Bait", spotify_id="s1")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=10_000) for i in range(1, 4)
    ]

    queue = select_replay_queue(events, has_audio_analysis=lambda _cid: False)

    assert queue == []


def test_three_valid_plays_queue_the_track():
    track = _track("Real Listen", spotify_id="s2")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]

    queue = select_replay_queue(events, has_audio_analysis=lambda _cid: False)

    assert queue == [track]


def test_two_valid_plays_below_threshold_are_not_queued():
    track = _track("Almost There", spotify_id="s3")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 3)
    ]

    queue = select_replay_queue(events, has_audio_analysis=lambda _cid: False)

    assert queue == []


# --- AC2: pool/root analysis exclusion, keyed on the #158 slice-0 --------- #
# canonical prefixed key (mbid: > isrc: > spotify: > name:), not a bare id.


def test_track_already_analyzed_on_the_slice0_key_is_excluded():
    track = _track("Already Analyzed", spotify_id="s4", mbid="mb-123")
    slice0_key = canonical_track_id(track)
    assert slice0_key == "mbid:mb-123"
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]

    queue = select_replay_queue(events, has_audio_analysis=lambda cid: cid == slice0_key)

    assert queue == []


def test_history_import_track_resolved_to_an_analyzed_mbid_is_excluded():
    # History-import TrackRefs (spotify_extended/ingest/youtube_music) never
    # carry mbid/isrc -- only spotify_id/youtube_id/name+artist -- so their
    # canonical id bottoms out at "spotify:<id>", while has_audio_analysis is
    # keyed on the mbid-prefixed id the live/AcoustID capture pipeline wrote
    # the analysis file under. Without the resolve_mbid bridge (mirroring
    # backfill_playlist.select_backfill_tracks, #177) this track would never
    # be recognized as already analyzed.
    track = _track("History Import Track", spotify_id="s5")
    assert canonical_track_id(track) == "spotify:s5"
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]

    queue = select_replay_queue(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:mb-999",
        resolve_mbid=lambda t: "mb-999" if t.spotify_id == "s5" else None,
    )

    assert queue == []


def test_history_import_track_resolved_to_an_unanalyzed_mbid_is_still_queued():
    track = _track("Not Yet Analyzed", spotify_id="s6")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]

    queue = select_replay_queue(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:mb-999",
        resolve_mbid=lambda t: "mb-other",
    )

    assert queue == [track]


# --- AC3: artist/year stratification with a cap --------------------------- #


def _eligible_events(track: TrackRef, *, year: int) -> list[ListenEvent]:
    return [
        _event(track, played_at=f"{year}-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]


def test_cap_is_respected():
    events = []
    for i in range(10):
        track = _track(f"Track {i}", artist=f"Artist {i}", spotify_id=f"s{i}")
        events += _eligible_events(track, year=2020)

    queue = select_replay_queue(events, has_audio_analysis=lambda _cid: False, cap=4)

    assert len(queue) == 4


def test_dominant_artist_does_not_crowd_out_other_strata_under_cap():
    # Artist A has 5 eligible tracks (2020); Artist B and Artist C have 1
    # each (2021, 2022). With cap=3 a naive "sort by play count, take top N"
    # selection would return 3 Artist-A tracks and starve B/C entirely --
    # exactly the "single artist exceeds its proportional share" failure
    # AC3 rules out. Round-robin across (artist, year) strata instead gives
    # each of the three strata one slot before any stratum gets a second.
    events = []
    for i in range(5):
        track = _track(f"A Track {i}", artist="Artist A", spotify_id=f"a{i}")
        events += _eligible_events(track, year=2020)
    track_b = _track("B Track", artist="Artist B", spotify_id="b0")
    events += _eligible_events(track_b, year=2021)
    track_c = _track("C Track", artist="Artist C", spotify_id="c0")
    events += _eligible_events(track_c, year=2022)

    queue = select_replay_queue(events, has_audio_analysis=lambda _cid: False, cap=3)

    assert len(queue) == 3
    artists = sorted(t.artist for t in queue)
    assert artists == ["Artist A", "Artist B", "Artist C"]


# --- AC4 (logic half): AC5.1 coverage stat --------------------------------- #
# share of *valid plays* covered by already-analysed + newly-queued tracks,
# over the full history -- not share of unique/eligible tracks (decision
# 87277764). The CLI's print-the-stat half is test_cli_replay_queue.py.


def test_coverage_counts_valid_play_share_not_track_share():
    queued_track = _track("Queue Me", artist="Artist Q", spotify_id="q1")
    analyzed_track = _track("Already Done", artist="Artist D", spotify_id="d1")
    below_threshold_track = _track("Too Few", artist="Artist F", spotify_id="f1")

    events = (
        _eligible_events(queued_track, year=2020)
        + _eligible_events(analyzed_track, year=2020)
        + [
            _event(below_threshold_track, played_at="2020-01-01T00:00:00Z", ms_played=180_000),
            _event(below_threshold_track, played_at="2020-01-02T00:00:00Z", ms_played=180_000),
        ]
    )

    analyzed_cid = canonical_track_id(analyzed_track)
    stats = replay_queue_coverage(events, has_audio_analysis=lambda cid: cid == analyzed_cid)

    assert stats.eligible_track_count == 2  # queued_track + analyzed_track (>=3 valid plays)
    assert stats.already_analyzed_count == 1
    assert stats.queued_count == 1
    # 6 valid plays covered (queued + analyzed, 3 each) out of 8 total valid plays
    assert stats.valid_play_coverage == 6 / 8


def test_coverage_counts_a_history_import_track_via_resolve_mbid_bridge():
    # Same identity-mismatch gap as the queue-selection bridge tests above,
    # but for the AC5.1 coverage stat's already_analyzed_count/coverage share.
    analyzed_track = _track("History Analyzed", artist="Artist D", spotify_id="d2")
    events = _eligible_events(analyzed_track, year=2020)

    stats = replay_queue_coverage(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:mb-888",
        resolve_mbid=lambda t: "mb-888" if t.spotify_id == "d2" else None,
    )

    assert stats.already_analyzed_count == 1
    assert stats.queued_count == 0


# --- #170 AC3: youtube-key crosswalk skips decode when the pool already --- #
# covers the track under an aliased isrc:/mbid: key. Reuses the existing
# resolve_mbid seam (composed via youtube_crosswalk_resolve_mbid) rather than
# a new mechanism -- a youtube-origin TrackRef never carries its own
# isrc/mbid to hand to a plain resolve_mbid callable, so the bridge instead
# walks the alias chain #170 AC6 populates (``youtube:<id>`` -> the
# score-gated rung's winner key).


def test_youtube_track_aliased_to_an_analyzed_mbid_is_not_queued():
    track = _track("Song", youtube_id="yt-1")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]
    resolve_mbid = youtube_crosswalk_resolve_mbid(
        resolve_track_key=lambda key: {"youtube:yt-1": "mbid:M-1"}.get(key, key)
    )

    queue = select_replay_queue(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:M-1",
        resolve_mbid=resolve_mbid,
    )

    assert queue == []


def test_youtube_track_with_no_alias_is_still_queued():
    track = _track("Unaliased Song", youtube_id="yt-2")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]
    # resolve_track_key is a no-op bridge here -- no alias was ever written
    # for yt-2, so it echoes the key back unchanged (mirrors store.py's
    # resolve_key/resolve_track_key contract: no alias hop found -> the
    # input key itself).
    resolve_mbid = youtube_crosswalk_resolve_mbid(resolve_track_key=lambda key: key)

    queue = select_replay_queue(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:M-1",
        resolve_mbid=resolve_mbid,
    )

    assert queue == [track]


def test_youtube_crosswalk_falls_back_to_a_wrapped_resolve_mbid_for_non_youtube_tracks():
    # youtube_crosswalk_resolve_mbid must compose with (not replace) an
    # existing resolve_mbid bridge -- e.g. _cmd_replay_queue's
    # resolver.resolve(t).mbid lambda -- so a spotify-origin history track
    # still gets its ordinary mbid bridge, not just youtube-origin ones.
    track = _track("Spotify History Track", spotify_id="s7")
    events = [
        _event(track, played_at=f"2026-01-0{i}T00:00:00Z", ms_played=180_000) for i in range(1, 4)
    ]
    resolve_mbid = youtube_crosswalk_resolve_mbid(
        resolve_track_key=lambda key: key,
        resolve_mbid=lambda t: "mb-777" if t.spotify_id == "s7" else None,
    )

    queue = select_replay_queue(
        events,
        has_audio_analysis=lambda cid: cid == "mbid:mb-777",
        resolve_mbid=resolve_mbid,
    )

    assert queue == []


# --- #170 AC8: achievable coverage ceiling, measured before AC5.1's >=80% -- #
# target (decision 87277764) is applied. The ceiling is the coverage achieved
# at the current min_valid_plays threshold with the queue cap removed -- it
# isolates whether DEFAULT_REPLAY_QUEUE_CAP is the binding constraint from
# whether MIN_VALID_PLAYS is.


def _eligible_events_many(n: int, *, year: int = 2020) -> list[ListenEvent]:
    events = []
    for i in range(n):
        track = _track(f"Ceiling Track {i}", artist=f"Artist {i}", spotify_id=f"c{i}")
        events += _eligible_events(track, year=year)
    return events


def test_ceiling_ignores_a_binding_cap_that_configured_coverage_is_stuck_under():
    events = _eligible_events_many(3)

    report = measure_coverage_ceiling(events, has_audio_analysis=lambda _cid: False, cap=1)

    assert report.configured_coverage == pytest.approx(1 / 3)
    assert report.ceiling_coverage == pytest.approx(1.0)
    assert report.ceiling_meets_target is True
    assert report.cap_should_be_lifted is True
    assert report.min_valid_plays_should_be_lifted is False


def test_ceiling_lifting_the_cap_to_exactly_target_still_recommends_lifting_it():
    # Boundary case: ceiling_coverage == target exactly. The cap is still the
    # binding constraint here (configured_coverage is below target only
    # because of the cap), so lifting it should still be recommended -- an
    # off-by-one on this boundary (< instead of <=) would silently drop the
    # recommendation exactly when the ceiling just barely clears the target.
    events = _eligible_events_many(3)

    report = measure_coverage_ceiling(
        events, has_audio_analysis=lambda _cid: False, cap=1, target=1.0
    )

    assert report.configured_coverage == pytest.approx(1 / 3)
    assert report.ceiling_coverage == pytest.approx(1.0)
    assert report.cap_should_be_lifted is True


def test_ceiling_below_target_recommends_lowering_min_valid_plays_not_the_cap():
    eligible_track = _track("Eligible", artist="Artist E", spotify_id="e1")
    ineligible_track = _track("Too Few Plays", artist="Artist F", spotify_id="f1")
    events = _eligible_events(eligible_track, year=2020) + [
        _event(ineligible_track, played_at="2020-02-01T00:00:00Z", ms_played=180_000),
        _event(ineligible_track, played_at="2020-02-02T00:00:00Z", ms_played=180_000),
    ]

    report = measure_coverage_ceiling(events, has_audio_analysis=lambda _cid: False)

    # 3 valid plays covered (the eligible track) out of 5 total valid plays = 0.6
    assert report.ceiling_coverage == pytest.approx(0.6)
    assert report.configured_coverage == pytest.approx(0.6)
    assert report.ceiling_meets_target is False
    assert report.cap_should_be_lifted is False
    assert report.min_valid_plays_should_be_lifted is True


def test_ceiling_already_meets_target_recommends_nothing():
    track = _track("Covered", artist="Artist G", spotify_id="g1")
    events = _eligible_events(track, year=2020)

    report = measure_coverage_ceiling(
        events, has_audio_analysis=lambda cid: cid == canonical_track_id(track)
    )

    assert report.ceiling_coverage == pytest.approx(1.0)
    assert report.configured_coverage == pytest.approx(1.0)
    assert report.ceiling_meets_target is True
    assert report.cap_should_be_lifted is False
    assert report.min_valid_plays_should_be_lifted is False
