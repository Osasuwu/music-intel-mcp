"""Live Spotify Web API ISRC source — the ``spotify_id -> ISRC`` waterfall leg (#87 AC-A).

History carries Spotify track ids, not ISRCs; the MusicBrainz dump is keyed by
ISRC. This module bridges the gap with the one Spotify endpoint our (non-
grandfathered) app is allowed: ``GET /v1/tracks``, which returns
``external_ids.isrc`` per track. It plugs into the :class:`~music_intel_mcp.
identity.SpotifyIsrcSource` seam (``lookup(spotify_id) -> str | None``) and adds a
:meth:`SpotifyApiIsrcSource.warm` prefetch that batches the lookups 50-at-a-time.

Cost discipline (CLAUDE.md — the API is rate-limited and costs real money):

- **Client-credentials flow.** One token (POST ``accounts.spotify.com/api/token``,
  HTTP Basic on the client id/secret), cached in memory until just before expiry.
  Credentials are read from ``SPOTIFY_CLIENT_ID`` / ``SPOTIFY_CLIENT_SECRET`` and
  never logged (mirrors :class:`~music_intel_mcp.scene.LastfmTagSource`).
- **Persistent JSONL cache, positive *and* negative.** Every resolved id — hit
  (``{"spotify_id", "isrc"}``) or miss (``isrc: null``) — is appended to a local
  sidecar, so a re-run (or a resumed, interrupted warm) skips ids it already
  asked about rather than re-querying from zero (memory
  ``scene-warm-negative-cache-resume``). The cache is per-user-local, never
  cloud (history-never-leaves-the-machine).
- **Backoff on 429/503.** ``Retry-After`` honored when present, else exponential;
  capped at :data:`_MAX_RETRIES` attempts.

Network-only: ``httpx`` is imported lazily and tests mock it with ``respx`` — the
live endpoints are never touched in CI.
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from .store import resolve_data_root

_SPOTIFY_CLIENT_ID_ENV = "SPOTIFY_CLIENT_ID"
_SPOTIFY_CLIENT_SECRET_ENV = "SPOTIFY_CLIENT_SECRET"

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_TRACKS_URL = "https://api.spotify.com/v1/tracks"

_TRACK_URI_PREFIX = "spotify:track:"
_BATCH = 50  # Spotify caps GET /v1/tracks at 50 ids per call.
_MAX_RETRIES = 5
_TOKEN_EXPIRY_MARGIN_S = 60  # refresh a minute early to dodge edge-of-expiry 401s.
_DEFAULT_CACHE_FILENAME = "spotify_isrc_cache.jsonl"
_USER_AGENT = "music-intel-mcp/0.0.1 ( https://github.com/Osasuwu/music-intel-mcp )"


def _bare_id(spotify_id: str) -> str:
    """A bare track id from either a bare id or a ``spotify:track:<id>`` URI."""
    sid = spotify_id.strip()
    if sid.startswith(_TRACK_URI_PREFIX):
        sid = sid[len(_TRACK_URI_PREFIX) :]
    return sid


class SpotifyApiIsrcSource:
    """Production :class:`~music_intel_mcp.identity.SpotifyIsrcSource` over the
    Spotify Web API.

    Construct with explicit ``client_id``/``client_secret`` or leave them ``None``
    to read the ``SPOTIFY_CLIENT_ID``/``SPOTIFY_CLIENT_SECRET`` env vars. ``warm``
    is the batching prefetch (call it once with every history spotify id before
    resolution); ``lookup`` is a cache read that falls back to a single-id fetch
    when the id was never warmed. ``sleep`` is injectable so tests exercise the
    backoff without real delays. ``api_calls`` counts logical batch fetches (after
    retries) for telemetry/tests."""

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_path: str | Path | None = None,
        data_root: str | Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 15.0,
    ) -> None:
        self._client_id = client_id or os.environ.get(_SPOTIFY_CLIENT_ID_ENV)
        self._client_secret = client_secret or os.environ.get(_SPOTIFY_CLIENT_SECRET_ENV)
        self._cache_path = (
            Path(cache_path)
            if cache_path is not None
            else resolve_data_root(data_root) / _DEFAULT_CACHE_FILENAME
        )
        self._sleep = sleep
        self._timeout = timeout
        self._token: str | None = None
        self._token_expiry = 0.0
        self._cache: dict[str, str | None] | None = None
        self.api_calls = 0

    # -- credentials / token ------------------------------------------------- #

    def _require_creds(self) -> tuple[str, str]:
        if not self._client_id or not self._client_secret:
            raise RuntimeError(
                f"{_SPOTIFY_CLIENT_ID_ENV} and {_SPOTIFY_CLIENT_SECRET_ENV} must be set "
                "to use SpotifyApiIsrcSource."
            )
        return self._client_id, self._client_secret

    def _access_token(self, httpx) -> str:
        if self._token is not None and time.monotonic() < self._token_expiry:
            return self._token
        client_id, client_secret = self._require_creds()
        basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        resp = self._request(
            httpx,
            "POST",
            SPOTIFY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        payload = resp.json()
        self._token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 3600))
        self._token_expiry = time.monotonic() + max(0, expires_in - _TOKEN_EXPIRY_MARGIN_S)
        return self._token

    # -- HTTP with backoff --------------------------------------------------- #

    def _request(self, httpx, method: str, url: str, **kwargs):
        """One request with retry on 429/503. ``Retry-After`` (seconds) is honored
        when present, otherwise the delay grows exponentially. Any other 4xx/5xx
        raises immediately (not retryable)."""
        resp = None
        for attempt in range(_MAX_RETRIES):
            resp = httpx.request(method, url, timeout=self._timeout, **kwargs)
            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.strip().isdigit()
                    else 2.0**attempt
                )
                self._sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        # Retries exhausted — surface the last (still-throttled) response as an error.
        resp.raise_for_status()
        return resp

    # -- cache --------------------------------------------------------------- #

    def _load_cache(self) -> dict[str, str | None]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, str | None] = {}
        if self._cache_path.exists():
            with self._cache_path.open(encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line from an interrupted run
                    sid = row.get("spotify_id")
                    if sid:
                        cache[sid] = row.get("isrc")  # None encodes a known miss
        self._cache = cache
        return cache

    def _append_cache(self, entries: dict[str, str | None]) -> None:
        cache = self._load_cache()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            for sid, isrc in entries.items():
                fh.write(json.dumps({"spotify_id": sid, "isrc": isrc}) + "\n")
                cache[sid] = isrc

    # -- fetch --------------------------------------------------------------- #

    def _fetch_batch(self, httpx, ids: list[str]) -> dict[str, str | None]:
        """Fetch <=50 ids in one call; every requested id gets an entry (its ISRC
        or ``None`` for an unknown/ISRC-less track), so misses are cached too."""
        token = self._access_token(httpx)
        resp = self._request(
            httpx,
            "GET",
            SPOTIFY_TRACKS_URL,
            params={"ids": ",".join(ids)},
            headers={"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT},
        )
        self.api_calls += 1
        out: dict[str, str | None] = dict.fromkeys(ids)  # default every id to a miss
        for track in resp.json().get("tracks", []):
            if not track:
                continue  # Spotify returns null for an unknown id
            tid = track.get("id")
            isrc = (track.get("external_ids") or {}).get("isrc")
            if tid:
                out[tid] = isrc or None
        return out

    # -- public API ---------------------------------------------------------- #

    def warm(self, spotify_ids: Iterable[str]) -> int:
        """Prefetch ISRCs for ``spotify_ids`` in batches of 50, appending every
        result (hit or miss) to the persistent cache. Ids already in the cache are
        skipped, so a re-run — or a resumed, interrupted warm — only fetches the
        remainder. Returns the count of ids newly fetched this call."""
        import httpx

        cache = self._load_cache()
        unknown: list[str] = []
        seen: set[str] = set()
        for raw in spotify_ids:
            sid = _bare_id(raw)
            if sid and sid not in cache and sid not in seen:
                seen.add(sid)
                unknown.append(sid)
        for i in range(0, len(unknown), _BATCH):
            self._append_cache(self._fetch_batch(httpx, unknown[i : i + _BATCH]))
        return len(unknown)

    def lookup(self, spotify_id: str) -> str | None:
        """ISRC for ``spotify_id`` from the cache, fetching a single id on demand
        when it was never warmed. Returns ``None`` for an unknown/ISRC-less track
        (a known miss is cached, not re-queried)."""
        sid = _bare_id(spotify_id)
        if not sid:
            return None
        cache = self._load_cache()
        if sid in cache:
            return cache[sid]
        import httpx

        results = self._fetch_batch(httpx, [sid])
        self._append_cache(results)
        return results.get(sid)
