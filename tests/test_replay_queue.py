"""Replay queue selector (#163) — pilot slice 1.

Pure-logic units (no HTTP): valid-play counting against the ≥30 s rule
(AC1), pool-or-root analysis exclusion (AC2), artist/year stratification
with a cap (AC3). The CLI subcommand's coverage-stat printing (AC4) is
covered in ``test_cli_replay_queue.py``. AC5 (``select_backfill_tracks``
untouched) is the existing ``test_backfill_playlist.py`` suite staying
green — no new test needed here.
"""

from __future__ import annotations

from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.replay_queue import replay_queue_coverage, select_replay_queue
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
    assert stats.valid_play_coverage == 1.0
