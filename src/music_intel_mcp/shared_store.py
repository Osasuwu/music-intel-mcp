"""Shared metadata store + pull-and-cache (decision f7a9fcbd, issue #60).

The shared store is an **additive, multi-user-ready** database of anonymous
track-level facts (Supabase Postgres in production). Enrichers (#61/#63/#64)
write facts here; the analyser reads them. It holds *no* per-user data — no
``user_id``, no ``played_at``, no play-context — so it can be shared across all
users (more users -> more cache hits -> faster analysis for everyone).

**Pull-and-cache** is the access pattern every enricher/analyser step uses:
one *bulk* read of all tracks of interest, materialised to the local cache
(``data/cache/<track_id>.json``), then analysis runs fully offline. Per-track
round-trips to Supabase during analysis are forbidden — :func:`pull_and_cache`
issues at most one :meth:`SharedStore.get_tracks` call per invocation.

Freshness: each record carries ``fetched_at``; an entry older than the TTL
(~90 days) is *stale* and re-pulled from the store (which, once an enricher
runs, re-derives it from the source APIs).

CI uses :class:`InMemorySharedStore` exclusively — no live Supabase, no network.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from .models import TrackRef
from .store import resolve_data_root

DEFAULT_TTL_DAYS = 90

# Env metadata for the production Supabase client (values live in .env / host
# env, never in the repo). Listed here so the lazy client knows what to read.
_SUPABASE_URL_ENV = "SUPABASE_URL"
_SUPABASE_KEY_ENV = "SUPABASE_KEY"

# PostgREST inlines an ``in.()`` filter into the URL query string, so a bulk read
# of N ids becomes one URL holding all N — past httpx's 65536-char limit on a
# realistic library (tens of thousands of tracks). Split bulk reads into id
# chunks below that ceiling; writes (POST body, no URL limit) chunk only to keep
# request bodies and the per-row work bounded. Both are *internal* to the store —
# the caller still issues one logical ``get_tracks``/``upsert_tracks``, so the
# pull-and-cache no-round-trip invariant (caller-level, not HTTP-level) holds.
_READ_CHUNK = 100
_WRITE_CHUNK = 500


def _chunked(seq: Sequence, size: int):
    """Yield ``seq`` in contiguous chunks of at most ``size`` (size ≥ 1)."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


# --------------------------------------------------------------------------- #
# Canonical track identity
# --------------------------------------------------------------------------- #


def canonical_track_id(track: TrackRef) -> str:
    """Stable string key for a track across the shared store and local cache.

    Mirrors the identity waterfall (mbid > isrc > spotify_id > name/artist).
    String form (the analyser's ``_track_key`` returns a tuple for counting);
    #61 unifies the two when identity resolution lands.
    """
    if track.mbid:
        return f"mbid:{track.mbid}"
    if track.isrc:
        return f"isrc:{track.isrc}"
    if track.spotify_id:
        return f"spotify:{track.spotify_id}"
    return f"name:{track.name.casefold()}\x1f{track.artist.casefold()}"


def encode_cache_key(key: str) -> str:
    """Percent-encode a canonical id into a collision-free filename stem.

    Any canonical id round-trips — including the ``name:<n>\\x1f<a>`` fallback,
    whose ``\\x1f`` separator and spaces are not filesystem-safe. Shared by the
    metadata cache and the identity cache (#61)."""
    return quote(key, safe="")


# --------------------------------------------------------------------------- #
# Records — anonymous track-level facts (no per-user fields, ever)
# --------------------------------------------------------------------------- #


class AudioFeatures(BaseModel):
    """Scalar audio descriptors (AcousticBrainz / future enrichers). All
    nullable — a track may be only partially enriched."""

    model_config = ConfigDict(extra="forbid")

    bpm: float | None = None
    energy: float | None = None
    valence: float | None = None
    danceability: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    source: str | None = None


class TrackTag(BaseModel):
    """One scene/cultural tag for a track (Last.fm and similar)."""

    model_config = ConfigDict(extra="forbid")

    tag: str
    weight: float | None = None
    source: str | None = None


class TrackMetadataRecord(BaseModel):
    """A track's anonymous facts as mirrored from the shared store.

    Carries NO per-user data by construction (``extra="forbid"`` blocks any
    stray ``user_id``/``played_at`` from leaking in). ``fetched_at`` is the TTL
    anchor — tz-aware UTC; naive datetimes are treated as UTC by :func:`is_stale`.
    """

    model_config = ConfigDict(extra="forbid")

    track_id: str
    spotify_id: str | None = None
    isrc: str | None = None
    mbid: str | None = None
    name: str
    artist: str
    audio_features: AudioFeatures | None = None
    tags: list[TrackTag] = Field(default_factory=list)
    fetched_at: datetime


def is_stale(
    record: TrackMetadataRecord,
    *,
    now: datetime,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> bool:
    """True when ``record`` is older than the TTL and should be re-fetched."""
    fetched = record.fetched_at
    if fetched.tzinfo is None:
        fetched = fetched.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (now - fetched) > timedelta(days=ttl_days)


# --------------------------------------------------------------------------- #
# SharedStore — protocol + implementations
# --------------------------------------------------------------------------- #


@runtime_checkable
class SharedStore(Protocol):
    """Bulk-read / bulk-write interface to the shared metadata store.

    Only bulk operations exist on purpose: there is no ``get_track`` singular,
    so the pull-and-cache no-round-trip rule is enforced by the type, not by
    convention.
    """

    def get_tracks(self, track_ids: Sequence[str]) -> dict[str, TrackMetadataRecord]:
        """Return the records that exist for ``track_ids`` (missing ids omitted)."""
        ...

    def upsert_tracks(self, records: Sequence[TrackMetadataRecord]) -> None:
        """Insert-or-replace ``records`` by ``track_id`` (enricher write-back)."""
        ...


class InMemorySharedStore:
    """Dict-backed :class:`SharedStore` — the real store in tests and the
    offline fallback. ``get_calls`` records each bulk read so tests can assert
    the no-round-trip invariant."""

    def __init__(self, records: Sequence[TrackMetadataRecord] | None = None) -> None:
        self._rows: dict[str, TrackMetadataRecord] = {}
        self.get_calls: list[list[str]] = []
        if records:
            self.upsert_tracks(records)

    def get_tracks(self, track_ids: Sequence[str]) -> dict[str, TrackMetadataRecord]:
        ids = list(track_ids)
        self.get_calls.append(ids)
        return {tid: self._rows[tid].model_copy(deep=True) for tid in ids if tid in self._rows}

    def upsert_tracks(self, records: Sequence[TrackMetadataRecord]) -> None:
        for record in records:
            self._rows[record.track_id] = record.model_copy(deep=True)


class SupabaseSharedStore:  # pragma: no cover - network-only, never run in CI
    """Production :class:`SharedStore` over Supabase Postgres.

    Network-only: the ``supabase`` SDK is imported lazily (optional extra
    ``music-intel-mcp[supabase]``) and the client is built from ``SUPABASE_URL``
    / ``SUPABASE_KEY``. Never exercised in CI — tests use
    :class:`InMemorySharedStore`. Reads/writes the ``tracks`` + ``audio_features``
    + ``tags`` tables (see ``supabase/migrations``).
    """

    def __init__(self, client: object | None = None) -> None:
        self._client = client

    @property
    def client(self) -> object:
        if self._client is None:
            try:
                from supabase import create_client
            except ImportError as exc:
                raise RuntimeError(
                    "SupabaseSharedStore needs the 'supabase' extra: "
                    "pip install 'music-intel-mcp[supabase]'"
                ) from exc
            url = os.environ.get(_SUPABASE_URL_ENV)
            key = os.environ.get(_SUPABASE_KEY_ENV)
            if not url or not key:
                raise RuntimeError(
                    f"{_SUPABASE_URL_ENV}/{_SUPABASE_KEY_ENV} must be set "
                    "to use the live shared store."
                )
            self._client = create_client(url, key)
        return self._client

    def get_tracks(self, track_ids: Sequence[str]) -> dict[str, TrackMetadataRecord]:
        ids = list(track_ids)
        if not ids:
            return {}
        client = self.client
        tracks: list[dict] = []
        feats: list[dict] = []
        tags: list[dict] = []
        # One ``.in_()`` per id-chunk per table (chunk keeps each URL under the
        # httpx limit); rows accumulate across chunks before assembly.
        for chunk in _chunked(ids, _READ_CHUNK):
            tracks.extend(client.table("tracks").select("*").in_("id", chunk).execute().data)
            feats.extend(
                client.table("audio_features").select("*").in_("track_id", chunk).execute().data
            )
            tags.extend(client.table("tags").select("*").in_("track_id", chunk).execute().data)

        feat_by_id = {f["track_id"]: f for f in feats}
        tags_by_id: dict[str, list[dict]] = {}
        for row in tags:
            tags_by_id.setdefault(row["track_id"], []).append(row)

        out: dict[str, TrackMetadataRecord] = {}
        for row in tracks:
            tid = row["id"]
            feat = feat_by_id.get(tid)
            out[tid] = TrackMetadataRecord(
                track_id=tid,
                spotify_id=row.get("spotify_id"),
                isrc=row.get("isrc"),
                mbid=row.get("mbid"),
                name=row["name"],
                artist=row["artist"],
                audio_features=AudioFeatures(**_audio_cols(feat)) if feat else None,
                tags=[
                    TrackTag(tag=t["tag"], weight=t.get("weight"), source=t.get("source"))
                    for t in tags_by_id.get(tid, [])
                ],
                fetched_at=row["fetched_at"],
            )
        return out

    def upsert_tracks(self, records: Sequence[TrackMetadataRecord]) -> None:
        if not records:
            return
        client = self.client
        track_rows = [
            {
                "id": r.track_id,
                "spotify_id": r.spotify_id,
                "isrc": r.isrc,
                "mbid": r.mbid,
                "name": r.name,
                "artist": r.artist,
                "fetched_at": r.fetched_at.isoformat(),
            }
            for r in records
        ]
        for chunk in _chunked(track_rows, _WRITE_CHUNK):
            client.table("tracks").upsert(chunk).execute()
        feat_rows = [
            {"track_id": r.track_id, **_audio_cols(r.audio_features.model_dump())}
            for r in records
            if r.audio_features is not None
        ]
        for chunk in _chunked(feat_rows, _WRITE_CHUNK):
            client.table("audio_features").upsert(chunk).execute()
        tag_rows = [
            {"track_id": r.track_id, "tag": t.tag, "weight": t.weight, "source": t.source}
            for r in records
            for t in r.tags
        ]
        for chunk in _chunked(tag_rows, _WRITE_CHUNK):
            client.table("tags").upsert(chunk, on_conflict="track_id,tag,source").execute()


def _audio_cols(data: dict) -> dict:  # pragma: no cover - used only by Supabase path
    cols = (
        "bpm",
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "instrumentalness",
        "source",
    )
    return {c: data.get(c) for c in cols}


# --------------------------------------------------------------------------- #
# Local metadata cache (materialised pull target)
# --------------------------------------------------------------------------- #


class MetadataCache:
    """Local materialisation of pulled shared-store records.

    Lives under ``<data root>/cache/<track_id>.json``. Regenerable and
    gitignored; filenames are percent-encoded so any canonical id (including the
    ``name:<n>\\x1f<a>`` fallback) round-trips collision-free.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_data_root(root)

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    def _path(self, track_id: str) -> Path:
        return self.cache_dir / f"{encode_cache_key(track_id)}.json"

    def read(self, track_id: str) -> TrackMetadataRecord | None:
        path = self._path(track_id)
        if not path.exists():
            return None
        return TrackMetadataRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def read_many(self, track_ids: Sequence[str]) -> dict[str, TrackMetadataRecord]:
        out: dict[str, TrackMetadataRecord] = {}
        for tid in track_ids:
            record = self.read(tid)
            if record is not None:
                out[tid] = record
        return out

    def write(self, record: TrackMetadataRecord) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(record.track_id)
        path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return path

    def write_many(self, records: Sequence[TrackMetadataRecord]) -> None:
        for record in records:
            self.write(record)


# --------------------------------------------------------------------------- #
# Pull-and-cache orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class PullResult:
    """Outcome of a :func:`pull_and_cache` run.

    - ``records`` — all metadata available for analysis (fresh cache hits +
      freshly pulled + stale-but-present, keyed by ``track_id``).
    - ``cache_hits`` — served from the local cache without touching the store.
    - ``pulled`` — fresh records pulled from the shared store this run.
    - ``stale`` — present in the store but past TTL; surfaced for re-enrichment.
    - ``missing`` — no metadata anywhere; the enricher must derive these.
    """

    records: dict[str, TrackMetadataRecord]
    cache_hits: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def pull_and_cache(
    track_ids: Sequence[str],
    store: SharedStore,
    cache: MetadataCache,
    *,
    now: datetime,
    ttl_days: int = DEFAULT_TTL_DAYS,
) -> PullResult:
    """Materialise metadata for ``track_ids`` into the local cache, offline-first.

    Fresh cache entries are served without touching ``store``. Everything else
    (uncached or stale) is collected and fetched in a **single** bulk
    :meth:`SharedStore.get_tracks` call — the no-round-trip invariant. Freshly
    pulled records are written back to the cache.
    """
    unique_ids = list(dict.fromkeys(track_ids))
    result = PullResult(records={})

    cached = cache.read_many(unique_ids)
    to_pull: list[str] = []
    for tid in unique_ids:
        record = cached.get(tid)
        if record is not None and not is_stale(record, now=now, ttl_days=ttl_days):
            result.records[tid] = record
            result.cache_hits.append(tid)
        else:
            to_pull.append(tid)

    if to_pull:
        fetched = store.get_tracks(to_pull)
        to_write: list[TrackMetadataRecord] = []
        for tid in to_pull:
            record = fetched.get(tid)
            if record is None:
                result.missing.append(tid)
                continue
            result.records[tid] = record
            to_write.append(record)
            if is_stale(record, now=now, ttl_days=ttl_days):
                result.stale.append(tid)
            else:
                result.pulled.append(tid)
        cache.write_many(to_write)

    return result
