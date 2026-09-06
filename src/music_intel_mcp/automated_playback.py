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

Pilot slice 1 (#159) hardens the real-device play path on top of the above:
mandatory ``device_id`` resolution, an account-busy gate, and bounded
404/403 retry -- see :meth:`SpotifyPlaybackClient.resolve_device_id`,
:meth:`SpotifyPlaybackClient.account_is_busy`, and :func:`attempt_play`.
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
SPOTIFY_PLAYER_URL = "https://api.spotify.com/v1/me/player"
SPOTIFY_PLAYER_DEVICES_URL = "https://api.spotify.com/v1/me/player/devices"


class TrackSkipped(Exception):
    """Raised by a ``play_track`` callable to signal that a track was not
    actually played -- e.g. deferred by :func:`attempt_play`'s account-busy
    gate, or abandoned after exhausting retries (#159 AC2/AC3). Distinct from
    consent revocation: the run keeps going with the next track, the skipped
    one is neither counted as played nor paced by its duration."""


class DeviceNotResolvedError(RuntimeError):
    """Raised by :meth:`SpotifyPlaybackClient.play` when no ``device_id`` has
    been resolved yet (#159 AC1) -- a play must never land on whatever device
    Spotify considers active by default."""


class DeviceNotFoundError(RuntimeError):
    """Raised by :meth:`SpotifyPlaybackClient.resolve_device_id` when no
    device on the account matches the requested name (#159 AC1)."""


class SpotifyPlayRejected(RuntimeError):
    """Raised by :meth:`SpotifyPlaybackClient.play` when Spotify rejects the
    play with 404 (device gone) or 403 (premium-required / restricted) (#159
    AC3) -- distinct from other HTTP errors so :func:`attempt_play` can
    re-queue specifically on these two, rather than treating every failure
    as retryable."""

    def __init__(self, status_code: int, message: str | None = None) -> None:
        super().__init__(message or f"play rejected with status {status_code}")
        self.status_code = status_code


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
        try:
            play_track(track)
        except TrackSkipped:
            continue
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

    def __init__(
        self,
        *,
        access_token: Callable[[], str],
        timeout: float = 15.0,
        device_id: str | None = None,
    ) -> None:
        self._access_token = access_token
        self._timeout = timeout
        self._device_id = device_id

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    def resolve_device_id(self, device_name: str) -> str:
        """Look up the device named ``device_name`` via ``GET /me/player/devices``
        and remember its id for subsequent :meth:`play` calls (#159 AC1) --
        replay must target the browser tab the driver launched, never
        whatever device Spotify considers active."""
        import httpx

        resp = httpx.get(SPOTIFY_PLAYER_DEVICES_URL, headers=self._headers(), timeout=self._timeout)
        resp.raise_for_status()
        for device in resp.json().get("devices", []):
            if device.get("name") == device_name:
                self._device_id = device["id"]
                return self._device_id
        raise DeviceNotFoundError(device_name)

    def account_is_busy(self) -> bool:
        """True iff ``GET /me/player`` reports playback already in progress on
        *any* device (#159 AC2) -- a play must defer rather than interrupt
        whatever the account is already doing. Spotify returns 204 with an
        empty body for a quiet account, so that case (and ``is_playing``
        missing/false) reads as not-busy."""
        import httpx

        resp = httpx.get(SPOTIFY_PLAYER_URL, headers=self._headers(), timeout=self._timeout)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return False
        return bool(resp.json().get("is_playing", False))

    def play(self, track_id: str) -> None:
        import httpx

        if self._device_id is None:
            raise DeviceNotResolvedError(
                "play() requires a resolved device_id -- call resolve_device_id() first"
            )

        resp = httpx.put(
            SPOTIFY_PLAYER_PLAY_URL,
            headers=self._headers(),
            params={"device_id": self._device_id},
            json={"uris": [spotify_track_uri(track_id)]},
            timeout=self._timeout,
        )
        if resp.status_code in (404, 403):
            raise SpotifyPlayRejected(resp.status_code)
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


@dataclass(frozen=True)
class PlayAttempt:
    """Outcome of one :func:`attempt_play` call (#159 AC2/AC3):

    - ``"played"`` -- the track played; ``reason`` is ``None``.
    - ``"deferred"`` -- the account was already busy (AC2); not counted
      against the track's retry budget, since it isn't the track's fault.
    - ``"requeued"`` -- Spotify rejected the play (404/403) and the track's
      retry count is still within ``max_retries``; the caller should
      re-queue it (AC3).
    - ``"abandoned"`` -- rejected again after exhausting ``max_retries``.
    """

    status: str
    reason: str | None


def attempt_play(
    client: SpotifyPlaybackClient,
    track: TrackRef,
    *,
    retry_counts: dict[str, int],
    max_retries: int = 3,
    journal: Callable[[TrackRef, str], None] | None = None,
) -> PlayAttempt:
    """Gate + play one track, journaling and bounding retries on rejection
    (#159 AC2/AC3) so a busy account or a 404/403 never crashes the run --
    the caller re-queues on ``"requeued"`` and moves on either way."""
    if client.account_is_busy():
        if journal is not None:
            journal(track, "account_busy")
        return PlayAttempt(status="deferred", reason="account_busy")

    try:
        client.play(track.spotify_id)
    except SpotifyPlayRejected as exc:
        reason = f"http_{exc.status_code}"
        if journal is not None:
            journal(track, reason)
        count = retry_counts.get(track.spotify_id, 0) + 1
        retry_counts[track.spotify_id] = count
        status = "abandoned" if count > max_retries else "requeued"
        return PlayAttempt(status=status, reason=reason)

    return PlayAttempt(status="played", reason=None)
