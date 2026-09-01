"""Backfill 'to-analyze' playlist automation (#127).

An opt-in, auto-managed Spotify playlist listing unanalyzed back-catalog
tracks (from the user's saved library, ``user-library-read``) so the passive
WASAPI capture pipeline (#124) gets analysis coverage just by playing through
it. This is the project's first *user*-OAuth (authorization-code + PKCE)
scope, on top of the existing client-credentials-only Spotify integration
(see :mod:`music_intel_mcp.spotify_user_auth`) — gated behind explicit opt-in
because it writes to the user's Spotify account (``playlist-modify-*``).

Pure selection/diff logic lives here, HTTP-free and independently testable;
:class:`SpotifyPlaylistClient` and :func:`sync_backfill_playlist` (the
network-touching half, mirroring :mod:`music_intel_mcp.spotify_api`'s
respx-mocked-in-CI convention) compose it against the live Spotify Web API.

- **AC4** (never re-add an already-played track): :func:`played_track_ids`
  is checked *before* the #126 analyzed-check in :func:`select_backfill_tracks`
  — a played-but-unanalyzed track is still excluded.
- **AC3** (remove-on-analyzed): the analyzed-check keeps a track out of the
  *desired* set; :func:`diff_playlist_membership` then removes anything
  currently on the playlist that has fallen out of that desired set on the
  next daily refresh — no special-cased "removal" path, just membership diff.
- **AC2** (10k cap): :data:`MAX_BACKFILL_TRACKS`, enforced in
  :func:`select_backfill_tracks`.
- **AC1** (opt-in gate): :func:`is_backfill_enabled`.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import ListenEvent, TrackRef
from .shared_store import canonical_track_id

DEFAULT_PLAYLIST_NAME = "music-intel: to-analyze"
MAX_BACKFILL_TRACKS = 10_000
DEFAULT_REFRESH_INTERVAL_S = 24 * 60 * 60  # AC2: daily cadence
_BACKFILL_ENABLED_ENV = "MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}

SPOTIFY_PLAYLISTS_URL = "https://api.spotify.com/v1/me/playlists"
SPOTIFY_SAVED_TRACKS_URL = "https://api.spotify.com/v1/me/tracks"
_ADD_REMOVE_BATCH = 100  # Spotify caps playlist add/remove at 100 uris per call.


def playlist_tracks_url(playlist_id: str) -> str:
    return f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"


def fetch_current_user_id(*, access_token: Callable[[], str], timeout: float = 15.0) -> str:
    """The authenticated user's Spotify id, required by the playlist-create
    endpoint (``POST /users/{user_id}/playlists``)."""
    import httpx

    resp = httpx.get(
        "https://api.spotify.com/v1/me",
        headers={"Authorization": f"Bearer {access_token()}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def fetch_saved_track_refs(
    *, access_token: Callable[[], str], timeout: float = 15.0
) -> list[TrackRef]:
    """The backfill candidate pool (decision 8dd6ef53): every track in the
    user's Spotify library, via ``user-library-read`` — followed to the end
    of pagination."""
    import httpx

    headers = {"Authorization": f"Bearer {access_token()}"}
    refs: list[TrackRef] = []
    url: str | None = SPOTIFY_SAVED_TRACKS_URL
    while url:
        resp = httpx.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("items", []):
            track = item.get("track") or {}
            artists = track.get("artists") or [{}]
            album = track.get("album") or {}
            refs.append(
                TrackRef(
                    spotify_id=track.get("id"),
                    name=track.get("name", ""),
                    artist=artists[0].get("name", ""),
                    album=album.get("name"),
                )
            )
        url = payload.get("next")
    return refs


def is_backfill_enabled(env: dict[str, str] | None = None) -> bool:
    """AC1's gate half: the backfill playlist is off unless explicitly opted
    into via ``MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED`` (writes to a user's
    Spotify account, so it must never activate by default)."""
    source = env if env is not None else os.environ
    return source.get(_BACKFILL_ENABLED_ENV, "").strip().lower() in _TRUTHY


def played_track_ids(events: Iterable[ListenEvent]) -> set[str]:
    """Canonical ids of every track already in history (AC4's exclusion set)."""
    return {canonical_track_id(event.track) for event in events}


def select_backfill_tracks(
    candidates: Iterable[TrackRef],
    *,
    played_ids: set[str],
    has_audio_analysis: Callable[[str], bool],
    limit: int = MAX_BACKFILL_TRACKS,
) -> list[TrackRef]:
    """The desired backfill set: unplayed, unanalyzed candidates, deduped by
    canonical id and capped at ``limit`` (AC2/AC3/AC4)."""
    selected: list[TrackRef] = []
    seen: set[str] = set()
    for track in candidates:
        if len(selected) >= limit:
            break
        cid = canonical_track_id(track)
        if cid in played_ids or cid in seen:
            continue  # AC4: played (even if unanalyzed) never enters the queue
        if has_audio_analysis(cid):
            continue  # AC3: already-analyzed tracks are never (re-)queued
        seen.add(cid)
        selected.append(track)
    return selected


@dataclass(frozen=True)
class PlaylistDiff:
    """What a daily refresh must change to reach the desired membership."""

    to_add: list[str]
    to_remove: list[str]


def diff_playlist_membership(
    current_ids: Iterable[str], desired_ids: Iterable[str]
) -> PlaylistDiff:
    """AC3's removal half: anything on the playlist that fell out of the
    desired set (e.g. it now has an analyzed entry) is queued for removal;
    anything newly desired is queued for addition. Order-independent,
    duplicate-tolerant on both inputs."""
    current = list(dict.fromkeys(current_ids))
    current_set = set(current)
    desired_ordered = list(dict.fromkeys(desired_ids))
    desired_set = set(desired_ordered)
    to_add = [tid for tid in desired_ordered if tid not in current_set]
    to_remove = [tid for tid in current if tid not in desired_set]
    return PlaylistDiff(to_add=to_add, to_remove=to_remove)


class SpotifyPlaylistClient:
    """Thin wrapper over the playlist-write half of the Spotify Web API
    (``playlist-modify-*`` / ``user-library-read`` — see
    :mod:`music_intel_mcp.spotify_user_auth`). ``access_token`` is a callable
    (typically :meth:`SpotifyUserAuth.access_token`) so a refreshed bearer is
    fetched fresh on every call rather than captured once at construction."""

    def __init__(
        self, *, access_token: Callable[[], str], user_id: str, timeout: float = 15.0
    ) -> None:
        self._access_token = access_token
        self.user_id = user_id
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token()}"}

    def get_or_create_playlist(self, *, name: str) -> str:
        """The backfill playlist's id — reused if a playlist with ``name``
        already exists among the user's playlists, else created (AC1)."""
        import httpx

        url: str | None = SPOTIFY_PLAYLISTS_URL
        while url:
            resp = httpx.get(url, headers=self._headers(), timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("items", []):
                if item.get("name") == name:
                    return item["id"]
            url = payload.get("next")

        resp = httpx.post(
            f"https://api.spotify.com/v1/users/{self.user_id}/playlists",
            headers=self._headers(),
            json={"name": name, "public": False},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()["id"]

    def get_playlist_track_ids(self, playlist_id: str) -> list[str]:
        """Every track id currently on the playlist, following pagination.

        Spotify's API returns bare ids (e.g. ``6y0igZArWVi6Iz0rj35c1Y``); these
        are re-prefixed to the canonical ``spotify:<id>`` form (matching
        :func:`~music_intel_mcp.shared_store.canonical_track_id`) so
        :func:`diff_playlist_membership` compares like-for-like against
        ``desired_ids``, which is always built via that same canonicalizer.
        """
        import httpx

        ids: list[str] = []
        url: str | None = playlist_tracks_url(playlist_id)
        while url:
            resp = httpx.get(url, headers=self._headers(), timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()
            for item in payload.get("items", []):
                track = item.get("track") or {}
                tid = track.get("id")
                if tid:
                    ids.append(f"spotify:{tid}")
            url = payload.get("next")
        return ids

    def add_tracks(self, playlist_id: str, track_ids: list[str]) -> None:
        """Add ``track_ids`` to the playlist, batched at Spotify's 100-uri cap."""
        import httpx

        for i in range(0, len(track_ids), _ADD_REMOVE_BATCH):
            batch = track_ids[i : i + _ADD_REMOVE_BATCH]
            resp = httpx.post(
                playlist_tracks_url(playlist_id),
                headers=self._headers(),
                json={"uris": [_track_uri(tid) for tid in batch]},
                timeout=self._timeout,
            )
            resp.raise_for_status()

    def remove_tracks(self, playlist_id: str, track_ids: list[str]) -> None:
        """Remove ``track_ids`` from the playlist (AC3), batched at 100 uris."""
        import httpx

        for i in range(0, len(track_ids), _ADD_REMOVE_BATCH):
            batch = track_ids[i : i + _ADD_REMOVE_BATCH]
            resp = httpx.request(
                "DELETE",
                playlist_tracks_url(playlist_id),
                headers=self._headers(),
                json={"tracks": [{"uri": _track_uri(tid)} for tid in batch]},
                timeout=self._timeout,
            )
            resp.raise_for_status()


def _track_uri(track_id: str) -> str:
    """A bare id becomes a full ``spotify:track:<id>`` uri; an id already
    carrying our internal ``spotify:<id>`` canonical prefix (see
    :func:`~music_intel_mcp.shared_store.canonical_track_id`) is passed
    through as-is — Spotify's API only cares that it's a resolvable uri."""
    if track_id.startswith("spotify:track:"):
        return track_id
    if track_id.startswith("spotify:"):
        return f"spotify:track:{track_id.split(':', 1)[1]}"
    return f"spotify:track:{track_id}"


def sync_backfill_playlist(
    client: SpotifyPlaylistClient,
    *,
    desired_ids: list[str],
    playlist_name: str = DEFAULT_PLAYLIST_NAME,
) -> PlaylistDiff:
    """One daily-refresh cycle (AC1/AC2/AC3's network half): resolve the
    playlist, diff its current membership against ``desired_ids`` (already
    capped/filtered by :func:`select_backfill_tracks` upstream), and apply the
    add/remove calls. Idempotent — re-running against an already-synced
    playlist issues no add/remove calls (empty diff)."""
    playlist_id = client.get_or_create_playlist(name=playlist_name)
    current_ids = client.get_playlist_track_ids(playlist_id)
    diff = diff_playlist_membership(current_ids, desired_ids)
    if diff.to_add:
        client.add_tracks(playlist_id, diff.to_add)
    if diff.to_remove:
        client.remove_tracks(playlist_id, diff.to_remove)
    return diff


def run_continuous_backfill(
    *,
    sync_once: Callable[[], PlaylistDiff],
    interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
    stop_event: threading.Event | None = None,
    on_result: Callable[[PlaylistDiff], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """AC2's daily-cadence half: keep re-running ``sync_once`` (typically
    :func:`sync_backfill_playlist` bound to a live client/desired-ids
    closure) every ``interval_s`` seconds until ``stop_event`` is set —
    mirrors :func:`~music_intel_mcp.continuous_capture.run_continuous_capture`'s
    ``stop_event``/``sleep``-injection convention so this is testable without
    a real day-long sleep. A single cycle's failure is reported via
    ``on_error`` rather than killing the loop, matching the always-keep-
    running behavior a daily background job needs."""
    stop_event = stop_event or threading.Event()
    while not stop_event.is_set():
        try:
            result = sync_once()
        except Exception as exc:  # noqa: BLE001 - reported, loop keeps running
            if on_error is not None:
                on_error(exc)
        else:
            if on_result is not None:
                on_result(result)
        sleep(interval_s)
