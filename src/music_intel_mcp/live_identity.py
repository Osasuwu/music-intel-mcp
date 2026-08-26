"""Live-capture identity waterfall (#139): AcoustID -> Spotify search -> ISRC
-> MBID -> MusicBrainz name search -> normalized name key.

The batch-import waterfall in ``identity.py`` (spotify_id -> ISRC -> MBID) stays
untouched — it resolves whatever identity a scrobble export already carries.
This module is the *live-capture* path: the OS media session only ever reports
a title/artist (and sometimes a spotify id), so the live path leads with an
audio fingerprint match and only falls through to string-based matching, which
is fragile (SoundCloud/regional titles have no AcoustID coverage, so the
fingerprint rung alone would silently under-resolve).

Bias: **fragmentation over false merge**. A low-score AcoustID match is treated
as no match (:data:`ACOUSTID_MIN_SCORE_DEFAULT`) rather than trusted, and the
normalized-name rung deliberately *keeps* Live/Remix/Radio Edit suffixes so
distinct recordings never collapse into one key — see :func:`normalize_track_name`.

Provenance (AC5): every resolution is paired with a :class:`ProvenanceSidecar`
(raw title/artist, source app id, capture timestamp, chromaprint fingerprint) so
a future identity-strategy change is a re-mapping job over the sidecars, never
a re-listen (CONTEXT.md invariant this issue adds).

Negative caching (AC6): a MusicBrainz-name-search or AcoustID miss is cached for
:data:`NEGATIVE_CACHE_TTL_DAYS` (~30 days). This is a deliberate, scoped revision
of the batch waterfall's persist-only-terminal-resolutions invariant (decision
dddc4d90) — that invariant still holds for the batch path; the live path caches
negatives because a live target is *repeatedly re-queried* (the same track plays
again) and the string-search legs are the ones actually costing network+ratelimit
budget on every repeat miss.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .identity import IsrcMbidIndex, SpotifyIsrcSource, disambiguate_mbids
from .shared_store import encode_cache_key
from .store import resolve_data_root

LiveResolutionLevel = Literal["acoustid", "spotify_search", "isrc", "mbid", "mb_name", "name"]

# Bump whenever the live-path cache schema (this module's ``LiveResolvedIdentity``
# or the negative-cache record shape) changes, so stale on-disk entries are
# ignored rather than trusted. Namespaces ``identity/live/v<N>/`` the same way
# ``identity.IDENTITY_CACHE_SCHEMA_VERSION`` namespaces the batch cache.
IDENTITY_CACHE_SCHEMA_VERSION = 1

# Below this AcoustID confidence, the match is not trusted (fragmentation over
# false merge, AC3) and resolution falls through to the string chain.
ACOUSTID_MIN_SCORE_DEFAULT = 0.5

NEGATIVE_CACHE_TTL_DAYS = 30

_ACOUSTID_API_KEY_ENV = "ACOUSTID_API_KEY"
ACOUSTID_LOOKUP_URL = "https://api.acoustid.org/v2/lookup"


# --------------------------------------------------------------------------- #
# Normalization (final fallback rung)
# --------------------------------------------------------------------------- #

_FEAT_RE = re.compile(r"\s*[\(\[]?\b(feat\.?|ft\.?|featuring)\b[^)\]]*[\)\]]?", re.IGNORECASE)
_OFFICIAL_VIDEO_RE = re.compile(r"\s*[\(\[]\s*official[^)\]]*[\)\]]", re.IGNORECASE)
_LYRICS_RE = re.compile(r"\s*[\(\[]\s*lyrics?\s*[\)\]]", re.IGNORECASE)
_REMASTER_RE = re.compile(r"\s*[\(\[-]\s*\d{4}\s*remaster(ed)?\s*[\)\]]?", re.IGNORECASE)
_DASH_RE = re.compile(r"[‐-―−]")
_QUOTE_RE = re.compile(r"[‘’“”]")
_WS_RE = re.compile(r"\s+")


def _clean(raw: str) -> str:
    text = _DASH_RE.sub("-", raw)
    text = _QUOTE_RE.sub("'", text)
    text = _FEAT_RE.sub("", text)
    text = _OFFICIAL_VIDEO_RE.sub("", text)
    text = _LYRICS_RE.sub("", text)
    text = _REMASTER_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.casefold()


def normalize_track_name(title: str, artist: str) -> str:
    """Final-fallback name key for the live waterfall (AC4).

    Casefolds; strips feat./ft./featuring credits, "(Official Video)"-style
    tags, "[Lyrics]" tags, and ``(<year> Remaster[ed])`` suffixes; normalizes
    unicode dashes/quotes and collapses whitespace. Deliberately does **not**
    strip Live/Remix/Radio Edit (or similar) suffixes — those denote a
    genuinely different recording, and merging them would be a false merge
    (this project's stated bias is fragmentation over false merge).

    This is independent of :func:`~music_intel_mcp.shared_store.canonical_track_id`
    and its ``name:<title>\\x1f<artist>`` fallback rung — that function and any
    keys it has already produced are untouched (AC4); this is a *new*, separate
    key used only by the live waterfall's final rung.
    """
    return f"{_clean(title)}\x1f{_clean(artist)}"


# --------------------------------------------------------------------------- #
# Lookup sources — protocols
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AcoustIdMatch:
    score: float
    mbid: str | None


@runtime_checkable
class AcoustIdSource(Protocol):
    """chromaprint fingerprint -> candidate AcoustID/MBID matches."""

    def match(self, fingerprint: str, duration_s: float) -> list[AcoustIdMatch]: ...


@runtime_checkable
class SpotifySearchSource(Protocol):
    """title/artist -> best-guess spotify track id (the "Spotify search" leg)."""

    def search(self, *, title: str, artist: str) -> str | None: ...


@runtime_checkable
class MusicBrainzNameSearchSource(Protocol):
    """title/artist -> best-guess recording MBID (the MB name-search leg)."""

    def search(self, *, title: str, artist: str) -> str | None: ...


class InMemoryAcoustIdSource:
    """Dict-backed :class:`AcoustIdSource` for tests, keyed by fingerprint."""

    def __init__(self, mapping: dict[str, list[AcoustIdMatch]] | None = None) -> None:
        self._map = dict(mapping or {})
        self.calls: list[str] = []

    def match(self, fingerprint: str, duration_s: float) -> list[AcoustIdMatch]:
        self.calls.append(fingerprint)
        return list(self._map.get(fingerprint, []))


class InMemorySpotifySearchSource:
    def __init__(self, mapping: dict[tuple[str, str], str] | None = None) -> None:
        self._map = dict(mapping or {})
        self.calls: list[tuple[str, str]] = []

    def search(self, *, title: str, artist: str) -> str | None:
        self.calls.append((title, artist))
        return self._map.get((title, artist))


class InMemoryMusicBrainzNameSearchSource:
    def __init__(self, mapping: dict[tuple[str, str], str] | None = None) -> None:
        self._map = dict(mapping or {})
        self.calls: list[tuple[str, str]] = []

    def search(self, *, title: str, artist: str) -> str | None:
        self.calls.append((title, artist))
        return self._map.get((title, artist))


class AcoustIdApiSource:
    """Production :class:`AcoustIdSource` over the AcoustID web API.

    ``httpx`` is imported lazily so the module stays importable without the
    dependency installed; tests mock it with ``respx`` (AC9 — no live AcoustID
    calls in CI). Reads the API key from ``ACOUSTID_API_KEY`` unless passed
    explicitly, declared in ``.env.example`` (AC7)."""

    def __init__(self, *, api_key: str | None = None, timeout: float = 15.0) -> None:
        self._api_key = api_key or os.environ.get(_ACOUSTID_API_KEY_ENV)
        self._timeout = timeout

    def match(self, fingerprint: str, duration_s: float) -> list[AcoustIdMatch]:
        if not self._api_key:
            raise RuntimeError(f"{_ACOUSTID_API_KEY_ENV} must be set to use AcoustIdApiSource.")
        import httpx

        resp = httpx.get(
            ACOUSTID_LOOKUP_URL,
            params={
                "client": self._api_key,
                "fingerprint": fingerprint,
                "duration": int(duration_s),
                "meta": "recordings",
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        matches: list[AcoustIdMatch] = []
        for result in payload.get("results", []):
            score = float(result.get("score", 0.0))
            recordings = result.get("recordings") or [{}]
            for rec in recordings:
                matches.append(AcoustIdMatch(score=score, mbid=rec.get("id")))
        return matches


# --------------------------------------------------------------------------- #
# Negative cache (AC6) — MB/AcoustID misses on the live path only
# --------------------------------------------------------------------------- #


class LiveNegativeCache:
    """~30-day TTL miss cache for the live waterfall's network-backed rungs.

    Scoped revision of the batch identity cache's persist-only-terminal
    invariant (decision dddc4d90): the live path repeatedly re-queries the
    *same* track (a song plays again), so caching "MusicBrainz/AcoustID had no
    match for this" avoids re-spending rate-limited API budget on a repeat miss
    — but only for :data:`NEGATIVE_CACHE_TTL_DAYS`, since a since-grown source
    database may resolve it later. ``now`` is injectable for tests."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        schema_version: int = IDENTITY_CACHE_SCHEMA_VERSION,
        ttl_days: int = NEGATIVE_CACHE_TTL_DAYS,
        now: object | None = None,
    ) -> None:
        self.root = resolve_data_root(root)
        self.schema_version = schema_version
        self.ttl_days = ttl_days
        self._now = now or (lambda: datetime.now(UTC))

    @property
    def cache_dir(self) -> Path:
        return self.root / "identity" / "live_negative" / f"v{self.schema_version}"

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{encode_cache_key(key)}.json"

    def get(self, key: str) -> bool:
        """True iff ``key`` has a still-live (unexpired) negative cache entry."""
        path = self._path(key)
        if not path.exists():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(record["cached_at"])
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return False
        return (self._now() - cached_at) <= timedelta(days=self.ttl_days)

    def put(self, key: str, *, reason: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        path.write_text(
            json.dumps({"cached_at": self._now().isoformat(), "reason": reason}),
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------- #
# Provenance sidecar (AC5)
# --------------------------------------------------------------------------- #


class ProvenanceSidecar(BaseModel):
    """Raw capture metadata stored alongside every live-capture embedding.

    Reversibility (CONTEXT.md invariant this issue adds): a future change to
    the identity strategy becomes a re-mapping job over these sidecars — no
    re-listening to the source audio required."""

    model_config = ConfigDict(extra="forbid")

    raw_title: str
    raw_artist: str
    app_id: str
    captured_at: str
    chromaprint_fingerprint: str | None = None


# --------------------------------------------------------------------------- #
# Resolved identity + resolver
# --------------------------------------------------------------------------- #


class LiveResolvedIdentity(BaseModel):
    """Outcome of one live-waterfall resolution. ``level`` is the rung reached
    (AC2). ``name_key`` is populated only when the waterfall bottoms out at the
    normalized-name rung."""

    model_config = ConfigDict(extra="forbid")

    spotify_id: str | None = None
    isrc: str | None = None
    mbid: str | None = None
    name: str
    artist: str
    level: LiveResolutionLevel
    name_key: str | None = None


class LiveIdentityResolver:
    """AcoustID -> Spotify search -> ISRC -> MBID -> MB name search -> name key.

    Every leg is optional except title/artist (always present from the OS media
    session) — an absent source simply skips its rung. ``min_acoustid_score``
    implements AC3: a low-confidence fingerprint match is discarded rather than
    trusted, and resolution falls through to the string chain.
    """

    def __init__(
        self,
        *,
        acoustid_source: AcoustIdSource | None = None,
        spotify_search: SpotifySearchSource | None = None,
        spotify_isrc: SpotifyIsrcSource | None = None,
        isrc_index: IsrcMbidIndex | None = None,
        mb_name_search: MusicBrainzNameSearchSource | None = None,
        negative_cache: LiveNegativeCache | None = None,
        min_acoustid_score: float = ACOUSTID_MIN_SCORE_DEFAULT,
    ) -> None:
        self.acoustid_source = acoustid_source
        self.spotify_search = spotify_search
        self.spotify_isrc = spotify_isrc
        self.isrc_index = isrc_index
        self.mb_name_search = mb_name_search
        self.negative_cache = negative_cache
        self.min_acoustid_score = min_acoustid_score

    def resolve(
        self,
        *,
        title: str,
        artist: str,
        fingerprint: str | None = None,
        duration_s: float = 0.0,
    ) -> LiveResolvedIdentity:
        mbid: str | None = None
        spotify_id: str | None = None
        isrc: str | None = None
        level: LiveResolutionLevel = "name"

        # Rung 1: AcoustID (high-score gated, AC3).
        if mbid is None and self.acoustid_source is not None and fingerprint:
            neg_key = f"acoustid:{fingerprint}"
            cached_miss = self.negative_cache is not None and self.negative_cache.get(neg_key)
            if not cached_miss:
                matches = self.acoustid_source.match(fingerprint, duration_s)
                best = max(matches, key=lambda m: m.score, default=None)
                if best is not None and best.score >= self.min_acoustid_score and best.mbid:
                    mbid, level = best.mbid, "acoustid"
                elif self.negative_cache is not None:
                    self.negative_cache.put(neg_key, reason="acoustid_no_high_score_match")

        # Rung 2: Spotify search.
        if mbid is None and self.spotify_search is not None:
            found_sid = self.spotify_search.search(title=title, artist=artist)
            if found_sid:
                spotify_id, level = found_sid, "spotify_search"

        # Rung 3: ISRC via the resolved spotify id.
        if mbid is None and spotify_id is not None and self.spotify_isrc is not None:
            found_isrc = self.spotify_isrc.lookup(spotify_id)
            if found_isrc:
                isrc, level = found_isrc, "isrc"

        # Rung 4: MBID via the ISRC -> MBID index (the existing MB-dump index).
        if mbid is None and isrc is not None and self.isrc_index is not None:
            candidates = self.isrc_index.lookup_all(isrc)
            found_mbid = disambiguate_mbids(candidates)
            if found_mbid:
                mbid, level = found_mbid, "mbid"

        # Rung 5: MusicBrainz name search (negative-cached, AC6).
        if mbid is None and self.mb_name_search is not None:
            neg_key = f"mbname:{title.casefold()}\x1f{artist.casefold()}"
            cached_miss = self.negative_cache is not None and self.negative_cache.get(neg_key)
            if not cached_miss:
                found_mbid = self.mb_name_search.search(title=title, artist=artist)
                if found_mbid:
                    mbid, level = found_mbid, "mb_name"
                elif self.negative_cache is not None:
                    self.negative_cache.put(neg_key, reason="mb_name_search_miss")

        # Rung 6: normalized name key (final fallback, AC4). Only overrides
        # ``level`` when nothing above matched — a rung that found a
        # spotify_id/isrc without reaching an mbid still counts as "the rung
        # reached" per AC2 and must not be reported as "name".
        name_key: str | None = None
        if mbid is None and level == "name":
            name_key = normalize_track_name(title, artist)

        return LiveResolvedIdentity(
            spotify_id=spotify_id,
            isrc=isrc,
            mbid=mbid,
            name=title,
            artist=artist,
            level=level,
            name_key=name_key,
        )
