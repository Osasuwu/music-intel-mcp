"""HTTP-mocked half of automated playback (#128) — :class:`SpotifyPlaybackClient`
against a mocked Spotify Web API Player Playback Control surface. Pacing/
revocation/traceability logic is covered separately in
``test_automated_playback.py`` (HTTP-free); this file exercises the
network-touching play/pause/duration calls, mirroring
``test_backfill_playlist_sync.py``'s conventions.
"""

from __future__ import annotations

import httpx
import respx

from music_intel_mcp.automated_playback import (
    SPOTIFY_PLAYER_PAUSE_URL,
    SPOTIFY_PLAYER_PLAY_URL,
    SPOTIFY_TRACKS_URL,
    SpotifyPlaybackClient,
)

_BEARER = "opaque-bearer-fixture"


def _client() -> SpotifyPlaybackClient:
    return SpotifyPlaybackClient(access_token=lambda: _BEARER)


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
