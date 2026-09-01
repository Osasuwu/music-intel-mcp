"""Automated-playback pacing/revocation loop (#128 AC2/AC3), HTTP-free.

Mirrors ``test_backfill_playlist.py``'s pure-function convention: no network,
consent/sleep/play are injected callables so the loop is exercised without a
real Spotify session or a real wall-clock wait.
"""

from __future__ import annotations

from datetime import UTC, datetime

from music_intel_mcp.automated_playback import (
    AUTOMATED_PLAYBACK_SOURCE,
    build_automated_play_event,
    run_automated_playback,
)
from music_intel_mcp.models import TrackRef


def _track(name: str, artist: str = "Artist") -> TrackRef:
    return TrackRef(spotify_id=f"spotify:track:{name}", name=name, artist=artist)


# #128 AC2: pacing is human-like (real-time listening duration), not rapid
# skip-through -- proven by asserting the loop sleeps out each track's full
# duration via multiple poll-sized waits, not one big sleep or zero sleep.
def test_run_automated_playback_paces_each_track_by_its_duration():
    tracks = [_track("a"), _track("b")]
    durations = {"a": 12.0, "b": 7.0}
    slept: list[float] = []
    played: list[TrackRef] = []

    result = run_automated_playback(
        queue=tracks,
        play_track=played.append,
        track_duration_s=lambda t: durations[t.name],
        has_consent=lambda: True,
        sleep=slept.append,
        poll_interval_s=5.0,
    )

    assert [t.name for t in played] == ["a", "b"]
    assert result.played == played
    assert result.stopped_early is False
    # 12s at 5s polls -> [5, 5, 2]; 7s at 5s polls -> [5, 2]
    assert slept == [5.0, 5.0, 2.0, 5.0, 2.0]
    assert sum(slept) == sum(durations.values())


def test_run_automated_playback_does_not_skip_through_in_one_big_sleep():
    tracks = [_track("a")]
    slept: list[float] = []

    run_automated_playback(
        queue=tracks,
        play_track=lambda t: None,
        track_duration_s=lambda t: 20.0,
        has_consent=lambda: True,
        sleep=slept.append,
        poll_interval_s=5.0,
    )

    # multiple poll-sized sleeps, not a single 20s sleep or zero sleeps
    assert len(slept) > 1
    assert all(s <= 5.0 for s in slept)


# #128 AC3: consent is revocable at any time; revocation stops the automated
# session immediately, mid-track if needed.
def test_run_automated_playback_stops_mid_track_when_consent_revoked():
    tracks = [_track("a"), _track("b")]
    played: list[TrackRef] = []
    consent_calls = {"n": 0}

    def has_consent() -> bool:
        consent_calls["n"] += 1
        # consent holds for the first check (before playing "a") and the first
        # poll tick during "a", then is revoked mid-track.
        return consent_calls["n"] <= 2

    result = run_automated_playback(
        queue=tracks,
        play_track=played.append,
        track_duration_s=lambda t: 20.0,
        has_consent=has_consent,
        sleep=lambda s: None,
        poll_interval_s=5.0,
    )

    assert [t.name for t in played] == ["a"]  # "b" never started
    assert result.played == played
    assert result.stopped_early is True


def test_run_automated_playback_stops_before_first_track_if_no_consent():
    result = run_automated_playback(
        queue=[_track("a")],
        play_track=lambda t: (_ for _ in ()).throw(AssertionError("must not play")),
        track_duration_s=lambda t: 1.0,
        has_consent=lambda: False,
        sleep=lambda s: None,
    )

    assert result.played == []
    assert result.stopped_early is True


# #128 AC3 (real-device gap caught by review, PR #151): local revocation must
# also pause the actual Spotify device -- otherwise the already-playing track
# keeps making sound after the local loop has "stopped".
def test_run_automated_playback_pauses_device_when_revoked_mid_track():
    tracks = [_track("a"), _track("b")]
    played: list[TrackRef] = []
    pause_calls = {"n": 0}
    consent_calls = {"n": 0}

    def has_consent() -> bool:
        consent_calls["n"] += 1
        return consent_calls["n"] <= 2

    result = run_automated_playback(
        queue=tracks,
        play_track=played.append,
        track_duration_s=lambda t: 20.0,
        has_consent=has_consent,
        sleep=lambda s: None,
        poll_interval_s=5.0,
        pause=lambda: pause_calls.__setitem__("n", pause_calls["n"] + 1),
    )

    assert result.stopped_early is True
    assert pause_calls["n"] == 1


def test_run_automated_playback_does_not_pause_before_first_track_starts():
    # nothing has been played yet, so there is nothing on the device to pause
    pause_calls = {"n": 0}

    result = run_automated_playback(
        queue=[_track("a")],
        play_track=lambda t: (_ for _ in ()).throw(AssertionError("must not play")),
        track_duration_s=lambda t: 1.0,
        has_consent=lambda: False,
        sleep=lambda s: None,
        pause=lambda: pause_calls.__setitem__("n", pause_calls["n"] + 1),
    )

    assert result.stopped_early is True
    assert pause_calls["n"] == 0


def test_run_automated_playback_does_not_pause_when_queue_completes_normally():
    # the last track already finished its full duration -- nothing to pause
    pause_calls = {"n": 0}

    result = run_automated_playback(
        queue=[_track("a")],
        play_track=lambda t: None,
        track_duration_s=lambda t: 1.0,
        has_consent=lambda: True,
        sleep=lambda s: None,
        pause=lambda: pause_calls.__setitem__("n", pause_calls["n"] + 1),
    )

    assert result.stopped_early is False
    assert pause_calls["n"] == 0


def test_run_automated_playback_completes_full_queue_without_revocation():
    tracks = [_track("a"), _track("b"), _track("c")]

    result = run_automated_playback(
        queue=tracks,
        play_track=lambda t: None,
        track_duration_s=lambda t: 1.0,
        has_consent=lambda: True,
        sleep=lambda s: None,
        poll_interval_s=5.0,
    )

    assert [t.name for t in result.played] == ["a", "b", "c"]
    assert result.stopped_early is False


# #128 AC4: automated plays must be traceable in history as agent-originated
# (distinct from a genuine user-initiated play), so downstream consumers
# (e.g. #109 S8) can identify them -- via ListenEvent.source alone, no new field.
def test_build_automated_play_event_uses_distinct_source():
    track = _track("a")
    played_at = datetime(2026, 1, 1, tzinfo=UTC)

    event = build_automated_play_event(track, played_at=played_at)

    assert event.source == AUTOMATED_PLAYBACK_SOURCE
    assert event.track == track
    assert event.played_at == played_at


def test_automated_playback_source_is_distinct_from_other_known_sources():
    # "ifttt" / "lastfm" style sources are genuine user-initiated plays;
    # the automated-playback source must never collide with them.
    assert AUTOMATED_PLAYBACK_SOURCE not in {"ifttt", "lastfm", "spotify_extended_history"}
