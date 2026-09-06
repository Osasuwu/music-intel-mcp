"""HTTP-mocked half of automated playback (#128) — :class:`SpotifyPlaybackClient`
against a mocked Spotify Web API Player Playback Control surface. Pacing/
revocation/traceability logic is covered separately in
``test_automated_playback.py`` (HTTP-free); this file exercises the
network-touching play/pause/duration calls, mirroring
``test_backfill_playlist_sync.py``'s conventions.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from music_intel_mcp.automated_playback import (
    SPOTIFY_PLAYER_DEVICES_URL,
    SPOTIFY_PLAYER_PAUSE_URL,
    SPOTIFY_PLAYER_PLAY_URL,
    SPOTIFY_PLAYER_URL,
    SPOTIFY_TRACKS_URL,
    DeviceNotFoundError,
    DeviceNotResolvedError,
    PlayAttempt,
    SpotifyPlaybackClient,
    SpotifyPlayRejected,
    attempt_play,
)
from music_intel_mcp.models import TrackRef

_BEARER = "opaque-bearer-fixture"


def _client(device_id: str | None = "dev1") -> SpotifyPlaybackClient:
    return SpotifyPlaybackClient(access_token=lambda: _BEARER, device_id=device_id)


def test_play_calls_player_play_endpoint_with_track_uri():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        route = router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(204))
        client.play("AAA")

    assert route.calls[0].request.headers["Authorization"] == f"Bearer {_BEARER}"
    body = route.calls[0].request.content.decode()
    assert '"spotify:track:AAA"' in body


def test_play_normalizes_canonical_track_id_prefix():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        route = router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(204))
        client.play("spotify:AAA")

    body = route.calls[0].request.content.decode()
    assert '"spotify:track:AAA"' in body


# --- #159 AC1: device_id is mandatory, resolved by device name ------------ #


def test_play_raises_without_resolved_device_id():
    client = _client(device_id=None)

    with pytest.raises(DeviceNotResolvedError):
        client.play("AAA")


def test_play_includes_device_id_query_param():
    client = _client(device_id="dev1")
    with respx.mock(assert_all_called=True) as router:
        route = router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(204))
        client.play("AAA")

    assert route.calls[0].request.url.params["device_id"] == "dev1"


def test_resolve_device_id_finds_device_by_name():
    client = _client(device_id=None)
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_DEVICES_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "devices": [
                        {"id": "other-device", "name": "Phone"},
                        {"id": "dev-abc", "name": "replay-browser"},
                    ]
                },
            )
        )
        device_id = client.resolve_device_id("replay-browser")

    assert device_id == "dev-abc"

    with respx.mock(assert_all_called=True) as router:
        route = router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(204))
        client.play("AAA")

    assert route.calls[0].request.url.params["device_id"] == "dev-abc"


def test_resolve_device_id_raises_when_name_not_found():
    client = _client(device_id=None)
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_DEVICES_URL).mock(
            return_value=httpx.Response(200, json={"devices": [{"id": "x", "name": "Phone"}]})
        )
        with pytest.raises(DeviceNotFoundError):
            client.resolve_device_id("replay-browser")


# --- #159 AC3: 404/403 raise a typed rejection for retry orchestration ----- #


def test_play_raises_rejected_on_404():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(SpotifyPlayRejected) as excinfo:
            client.play("AAA")

    assert excinfo.value.status_code == 404


def test_play_raises_rejected_on_403():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(403))
        with pytest.raises(SpotifyPlayRejected) as excinfo:
            client.play("AAA")

    assert excinfo.value.status_code == 403


# --- #159 AC3: attempt_play orchestrates gate + bounded retry/re-queue ----- #


def _track(spotify_id: str) -> TrackRef:
    return TrackRef(name="Song", artist="Artist", spotify_id=spotify_id)


def test_attempt_play_returns_played_on_success():
    client = _client()
    retry_counts: dict[str, int] = {}
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(204))

        result = attempt_play(client, _track("AAA"), retry_counts=retry_counts)

    assert result == PlayAttempt(status="played", reason=None)


def test_attempt_play_defers_and_journals_when_account_busy():
    client = _client()
    retry_counts: dict[str, int] = {}
    journaled: list[tuple[TrackRef, str]] = []
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": True})
        )
        # No play route registered -- assert_all_called=True fails the test
        # if attempt_play calls play() while the account is busy.
        track = _track("AAA")
        result = attempt_play(
            client,
            track,
            retry_counts=retry_counts,
            journal=lambda t, r: journaled.append((t, r)),
        )

    assert result == PlayAttempt(status="deferred", reason="account_busy")
    assert journaled == [(track, "account_busy")]
    assert retry_counts == {}


def test_attempt_play_requeues_on_404_and_journals_reason_with_bounded_count():
    client = _client()
    retry_counts: dict[str, int] = {}
    journaled: list[tuple[TrackRef, str]] = []
    track = _track("AAA")
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(404))

        result = attempt_play(
            client,
            track,
            retry_counts=retry_counts,
            journal=lambda t, r: journaled.append((t, r)),
        )

    assert result == PlayAttempt(status="requeued", reason="http_404")
    assert journaled == [(track, "http_404")]
    assert retry_counts == {"AAA": 1}


def test_attempt_play_requeues_at_exactly_max_retries_boundary():
    client = _client()
    track = _track("AAA")
    retry_counts = {"AAA": 2}
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(404))

        result = attempt_play(client, track, retry_counts=retry_counts, max_retries=3)

    assert result == PlayAttempt(status="requeued", reason="http_404")
    assert retry_counts == {"AAA": 3}


def test_attempt_play_abandons_track_once_max_retries_exceeded():
    client = _client()
    track = _track("AAA")
    retry_counts = {"AAA": 3}
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        router.put(SPOTIFY_PLAYER_PLAY_URL).mock(return_value=httpx.Response(403))

        result = attempt_play(client, track, retry_counts=retry_counts, max_retries=3)

    assert result == PlayAttempt(status="abandoned", reason="http_403")
    assert retry_counts == {"AAA": 4}


def test_attempt_play_skips_track_with_no_spotify_id_without_crashing():
    """A saved track with no usable Spotify id (e.g. Spotify returned
    ``id: null`` for a locally-added/unavailable saved track, per
    ``fetch_saved_track_refs``) must be skipped, not crash the run -- there
    is no id to retry with, so this is never retriable."""
    client = _client()
    retry_counts: dict[str, int] = {}
    journaled: list[tuple[TrackRef, str]] = []
    track = TrackRef(name="Song", artist="Artist", spotify_id=None)
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        # No play route registered -- assert_all_called=True fails the test
        # if attempt_play calls play() with no resolvable id.

        result = attempt_play(
            client,
            track,
            retry_counts=retry_counts,
            journal=lambda t, r: journaled.append((t, r)),
        )

    assert result == PlayAttempt(status="skipped", reason="missing_spotify_id")
    assert journaled == [(track, "missing_spotify_id")]
    assert retry_counts == {}


# --- #159 AC2: account-state gate ------------------------------------------ #


def test_account_is_busy_true_when_a_device_is_playing():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": True, "device": {"id": "other"}})
        )
        assert client.account_is_busy() is True


def test_account_is_busy_false_when_quiet_account():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(return_value=httpx.Response(204))
        assert client.account_is_busy() is False


def test_account_is_busy_false_when_nothing_playing():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(SPOTIFY_PLAYER_URL).mock(
            return_value=httpx.Response(200, json={"is_playing": False})
        )
        assert client.account_is_busy() is False


def test_pause_calls_player_pause_endpoint():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        route = router.put(SPOTIFY_PLAYER_PAUSE_URL).mock(return_value=httpx.Response(204))
        client.pause()

    assert route.calls[0].request.headers["Authorization"] == f"Bearer {_BEARER}"


def test_track_duration_s_reads_duration_ms_from_tracks_endpoint():
    client = _client()
    with respx.mock(assert_all_called=True) as router:
        router.get(f"{SPOTIFY_TRACKS_URL}/AAA").mock(
            return_value=httpx.Response(200, json={"duration_ms": 210_000})
        )
        duration = client.track_duration_s("spotify:track:AAA")

    assert duration == 210.0
