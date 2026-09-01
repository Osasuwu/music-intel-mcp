"""Automated playback mode (#128) -- the agent itself plays through the
backfill queue (#127) via the real Spotify client, human-paced.

**Off by default.** Enabling it requires a separately-recorded consent action
distinct from #127's ``MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED`` env-var opt-in
(:meth:`~music_intel_mcp.store.UserStore.grant_automated_playback_consent`) --
this drives a real playback session, not just a queue.

Pure pacing/revocation logic lives here, HTTP-free and independently testable
(mirrors :mod:`music_intel_mcp.backfill_playlist`'s split); the network-
touching half is :class:`SpotifyPlaybackClient`.

- **AC1** (opt-in, separate consent): enforced by the caller checking
  :meth:`UserStore.has_automated_playback_consent` before starting a session
  -- this module's ``has_consent`` callable is typically that method, called
  repeatedly so AC3 (mid-session revocation) falls out of the same check.
- **AC2** (human-like pacing): :func:`run_automated_playback` sleeps out each
  track's real duration in ``poll_interval_s``-sized increments rather than
  skip-through, so a run against N tracks takes ~sum(durations) wall-clock.
- **AC3** (revocable at any time, mid-track): ``has_consent`` is polled both
  before starting a new track and during every sleep increment within a
  track, so revocation takes effect within one poll interval; a mid-track
  revocation also calls the optional ``pause`` callable so the actual
  Spotify device stops, not just the local loop.
- **AC4** (traceable as agent-originated): :data:`AUTOMATED_PLAYBACK_SOURCE`
  is a distinct :attr:`~music_intel_mcp.models.ListenEvent.source` value,
  built by :func:`build_automated_play_event`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime

from .models import ListenEvent, TrackRef
from .shared_store import spotify_track_uri

AUTOMATED_PLAYBACK_SOURCE = "agent_automated_playback"
DEFAULT_CONSENT_POLL_INTERVAL_S = 5.0

SPOTIFY_PLAYER_PLAY_URL = "https://api.spotify.com/v1/me/player/play"
SPOTIFY_PLAYER_PAUSE_URL = "https://api.spotify.com/v1/me/player/pause"
SPOTIFY_TRACKS_URL = "https://api.spotify.com/v1/tracks"


@dataclass(frozen=True)
class AutomatedPlaybackResult:
    """What one automated-playback session did. ``stopped_early`` is True iff
    consent was revoked before the whole queue finished (AC3)."""

    played: list[TrackRef] = field(default_factory=list)
    stopped_early: bool = False


def run_automated_playback(
    *,
    queue: Iterable[TrackRef],
    play_track: Callable[[TrackRef], None],
    track_duration_s: Callable[[TrackRef], float],
    has_consent: Callable[[], bool],
    on_play: Callable[[TrackRef], None] | None = None,
    pause: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval_s: float = DEFAULT_CONSENT_POLL_INTERVAL_S,
) -> AutomatedPlaybackResult:
    """Play ``queue`` in order, human-paced (AC2), stopping immediately (even
    mid-track) the moment ``has_consent`` returns False (AC3).

    ``pause`` (typically :meth:`SpotifyPlaybackClient.pause`) is called only
    when revocation happens *mid-track* -- the one point where the Spotify
    device is actually still making sound. Revocation checked before a track
    starts never needs it: either nothing has played yet, or the previous
    track already ran out its full duration on the device.
    """
    played: list[TrackRef] = []
    for track in queue:
        if not has_consent():
            return AutomatedPlaybackResult(played=played, stopped_early=True)
        play_track(track)
        played.append(track)
        if on_play is not None:
            on_play(track)
        remaining = track_duration_s(track)
        while remaining > 0:
            if not has_consent():
                if pause is not None:
                    pause()
                return AutomatedPlaybackResult(played=played, stopped_early=True)
            wait = min(poll_interval_s, remaining)
            sleep(wait)
            remaining -= wait
    return AutomatedPlaybackResult(played=played, stopped_early=False)


def build_automated_play_event(track: TrackRef, *, played_at: datetime) -> ListenEvent:
    """A history entry for an agent-originated play (AC4) -- distinguishable
    from a genuine user-initiated play by ``source == AUTOMATED_PLAYBACK_SOURCE``
    alone, so downstream consumers (e.g. #109 S8) can filter on it without a
    schema change."""
    return ListenEvent(track=track, played_at=played_at, source=AUTOMATED_PLAYBACK_SOURCE)


class SpotifyPlaybackClient:
    """Thin wrapper over the Player Playback Control half of the Spotify Web
    API (``user-modify-playback-state`` / ``user-read-playback-state`` -- see
    :data:`~music_intel_mcp.spotify_user_auth.PLAYBACK_SCOPES`). ``access_token``
    is a callable (typically :meth:`SpotifyUserAuth.access_token`) so a
    refreshed bearer is fetched fresh on every call rather than captured once
    at construction (mirrors :class:`~music_intel_mcp.backfill_playlist.
    SpotifyPlaylistClient`)."""

    def __init__(self, *, access_token: Callable[[], str], timeout: float = 15.0) -> None:
        self._access_token = access_token
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    def play(self, track_id: str) -> None:
        import httpx

        resp = httpx.put(
            SPOTIFY_PLAYER_PLAY_URL,
            headers=self._headers(),
            json={"uris": [spotify_track_uri(track_id)]},
            timeout=self._timeout,
        )
        resp.raise_for_status()

    def pause(self) -> None:
        import httpx

        resp = httpx.put(SPOTIFY_PLAYER_PAUSE_URL, headers=self._headers(), timeout=self._timeout)
        resp.raise_for_status()

    def track_duration_s(self, track_id: str) -> float:
        """The track's runtime in seconds, from Spotify's ``duration_ms`` --
        what :func:`run_automated_playback` paces AC2's human-like listen
        duration against."""
        import httpx

        bare_id = track_id.rsplit(":", 1)[-1]
        resp = httpx.get(
            f"{SPOTIFY_TRACKS_URL}/{bare_id}",
            headers=self._headers(),
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["duration_ms"] / 1000.0
