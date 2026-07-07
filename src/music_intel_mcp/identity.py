"""Track identity resolution — the spotify_id -> ISRC -> MBID waterfall (#61).

History events carry whatever identity their source happened to know: a Spotify
id from the IFTTT export, an ISRC or MBID from a Last.fm + MusicBrainz scrobble,
or nothing but name/artist. The downstream enrichers need one *stable* join key
— the MusicBrainz recording MBID — to look facts up in the AcousticBrainz /
MusicBrainz dumps (#63/#64). This module walks each :class:`TrackRef` up the
waterfall::

    mbid (already present)            -> done
    isrc -> MBID via the MB dump      -> done
    spotify_id -> ISRC (Spotify) -> MBID via the MB dump

Two CONTEXT.md invariants hold here:

- **Transparent rejection** — a track that cannot reach an MBID is *flagged and
  counted*, never silently dropped. :class:`ResolutionReport` records the level
  each track reached and lists the unresolved ones.
- **No re-resolution** — resolved identities are written to a local
  :class:`IdentityCache` keyed by the *input* identity and reused on re-run, so
  a second analysis never re-walks the dump for a track it already resolved.

The MusicBrainz dump lives OUTSIDE the repo (env-pointed); tests use the
in-memory index/source and a tiny synthetic TSV — never the live dump or any
API. A live Spotify ``spotify_id -> ISRC`` source plugs into the
:class:`SpotifyIsrcSource` seam in a later slice; without it, spotify-only
tracks are honestly flagged at the ``spotify`` level rather than dropped.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import TrackRef
from .shared_store import (
    TrackMetadataRecord,
    canonical_track_id,
    encode_cache_key,
)
from .store import resolve_data_root

# Env metadata for the production index (values/paths live in .env / host env).
# Explicit index path > MUSICBRAINZ_ISRC_INDEX > $MUSICBRAINZ_DUMP_DIR/<default>.
_MB_ISRC_INDEX_ENV = "MUSICBRAINZ_ISRC_INDEX"
_MB_DUMP_DIR_ENV = "MUSICBRAINZ_DUMP_DIR"
_DEFAULT_INDEX_FILENAME = "isrc_to_mbid.tsv"

ResolutionLevel = Literal["mbid", "isrc", "spotify", "name"]

# Bump whenever ``ResolvedIdentity``'s schema or the resolution semantics change,
# so on-disk caches written by an older schema are ignored rather than trusted.
# The version namespaces the cache directory (``identity/v<N>/``); old versions
# are simply never consulted (regenerable, gitignored — no migration needed).
IDENTITY_CACHE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Resolved identity record
# --------------------------------------------------------------------------- #


class ResolvedIdentity(BaseModel):
    """The outcome of resolving one track. ``level`` is the deepest waterfall
    rung reached; ``input_key`` is the canonical id of the *pre-resolution* ref
    (the cache key). Anonymous by construction — ``extra='forbid'`` blocks any
    per-user field from leaking into a shared identity record."""

    model_config = ConfigDict(extra="forbid")

    input_key: str
    spotify_id: str | None = None
    isrc: str | None = None
    mbid: str | None = None
    name: str
    artist: str
    level: ResolutionLevel

    @property
    def resolved(self) -> bool:
        """True once the waterfall reached an MBID (the cross-dump join key)."""
        return self.level == "mbid"

    def to_track_ref(self) -> TrackRef:
        return TrackRef(
            spotify_id=self.spotify_id,
            isrc=self.isrc,
            mbid=self.mbid,
            name=self.name,
            artist=self.artist,
        )


# --------------------------------------------------------------------------- #
# Lookup sources — protocols + implementations
# --------------------------------------------------------------------------- #

# A predicate answering "does this MBID have AcousticBrainz features?" — the
# disambiguation signal for a multi-valued ISRC. In production this wraps the
# AcousticBrainz dump (``lambda m: ab_source.lookup(m) is not None``); ``None``
# means no AB information, so disambiguation falls back to the smallest MBID.
AbCoverage = Callable[[str], bool]


def disambiguate_mbids(
    mbids: Iterable[str],
    *,
    ab_covered: AbCoverage | None = None,
) -> str | None:
    """Pick one recording MBID for an ISRC that maps to several (#87 AC-B).

    One ISRC legitimately maps to many MusicBrainz recordings (re-releases,
    per-region masters, remaster duplicates). The join we actually need is to
    AcousticBrainz features, so **prefer an AB-covered recording**; among equally
    covered (or, absent AB info, among all) candidates, **tie-break by the
    smallest MBID** so the choice is deterministic across runs. Returns ``None``
    only for an empty candidate list."""
    ordered = sorted(set(mbids))
    if not ordered:
        return None
    if ab_covered is not None:
        covered = [m for m in ordered if ab_covered(m)]
        if covered:
            return covered[0]
    return ordered[0]


@runtime_checkable
class IsrcMbidIndex(Protocol):
    """ISRC -> recording MBID lookup (the MusicBrainz dump leg).

    ``lookup`` returns the single best MBID (disambiguated when the ISRC maps to
    several, per :func:`disambiguate_mbids`); ``lookup_all`` exposes every
    candidate MBID for that ISRC (empty when unknown)."""

    def lookup(self, isrc: str) -> str | None: ...

    def lookup_all(self, isrc: str) -> list[str]: ...


@runtime_checkable
class SpotifyIsrcSource(Protocol):
    """spotify_id -> ISRC lookup (the Spotify-metadata leg)."""

    def lookup(self, spotify_id: str) -> str | None: ...


def _as_mbid_list(value: str | Iterable[str]) -> list[str]:
    """Normalise a mapping value to an MBID list. A bare string (older single-
    valued fixtures) becomes a one-element list; an iterable is materialised."""
    if isinstance(value, str):
        return [value]
    return list(value)


class InMemoryIsrcMbidIndex:
    """Dict-backed :class:`IsrcMbidIndex` for tests and small extracts.

    Values may be a single MBID (``{isrc: mbid}``, older fixtures) or a list
    (``{isrc: [mbid, ...]}``, a multi-valued ISRC) — both normalise to a list.
    ``lookups`` records each query so tests can assert the dump was not re-walked
    on a cache hit."""

    def __init__(self, mapping: dict[str, str | list[str]] | None = None) -> None:
        self._map = {isrc: _as_mbid_list(v) for isrc, v in (mapping or {}).items()}
        self.lookups: list[str] = []

    def lookup_all(self, isrc: str) -> list[str]:
        self.lookups.append(isrc)
        return list(self._map.get(isrc, []))

    def lookup(self, isrc: str) -> str | None:
        return disambiguate_mbids(self.lookup_all(isrc))


class InMemorySpotifyIsrcSource:
    """Dict-backed :class:`SpotifyIsrcSource` for tests."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._map = dict(mapping or {})
        self.lookups: list[str] = []

    def lookup(self, spotify_id: str) -> str | None:
        self.lookups.append(spotify_id)
        return self._map.get(spotify_id)


class MusicBrainzIsrcIndex:
    """ISRC -> recording MBID index read from a TSV derived from the MB dump.

    The TSV is a prebuilt ``<isrc>\\t<mbid>`` extract (one pair per line, ``#``
    comments allowed); the raw MusicBrainz dump is far too large to scan per
    analysis. One ISRC legitimately maps to several recordings, so repeated rows
    for an ISRC **accumulate into a candidate list** (dup MBIDs collapsed, first-
    seen order kept); :func:`disambiguate_mbids` picks one at lookup time. Path
    resolution: explicit arg > ``MUSICBRAINZ_ISRC_INDEX`` >
    ``$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv``. A missing file yields an empty
    index (every lookup misses) rather than an error — honest low coverage beats
    a crash when the dump isn't installed. Loaded once, lazily, on first lookup.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._explicit = Path(path) if path is not None else None
        self._map: dict[str, list[str]] | None = None

    def _resolve_path(self) -> Path | None:
        if self._explicit is not None:
            return self._explicit
        env = os.environ.get(_MB_ISRC_INDEX_ENV)
        if env:
            return Path(env)
        dump_dir = os.environ.get(_MB_DUMP_DIR_ENV)
        if dump_dir:
            return Path(dump_dir) / _DEFAULT_INDEX_FILENAME
        return None

    def _load(self) -> dict[str, list[str]]:
        if self._map is not None:
            return self._map
        mapping: dict[str, list[str]] = {}
        path = self._resolve_path()
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    isrc, mbid = parts[0].strip(), parts[1].strip()
                    if isrc and mbid:
                        bucket = mapping.setdefault(isrc, [])
                        if mbid not in bucket:
                            bucket.append(mbid)
        self._map = mapping
        return mapping

    def lookup_all(self, isrc: str) -> list[str]:
        return list(self._load().get(isrc, []))

    def lookup(self, isrc: str) -> str | None:
        return disambiguate_mbids(self.lookup_all(isrc))


# --------------------------------------------------------------------------- #
# Identity cache (no re-resolution on re-run)
# --------------------------------------------------------------------------- #


class IdentityCache:
    """Local cache of resolved identities (``<data root>/identity/v<N>/<key>.json``).

    Keyed by the *input* canonical id (what history knew) so a re-run with the
    same history skips the waterfall entirely. Regenerable and gitignored, like
    the metadata cache; filenames are percent-encoded for any canonical id.

    The ``v<N>`` segment (``schema_version``, #87 AC-D) namespaces the cache by
    the identity-record schema: bumping :data:`IDENTITY_CACHE_SCHEMA_VERSION`
    when ``ResolvedIdentity`` changes lands new writes in a fresh directory, so
    entries written by an older schema are never read. As a second guard, a file
    that no longer validates (partial write, or a drift the version bump missed)
    is treated as a cache miss — the resolver re-walks rather than crashing."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        schema_version: int = IDENTITY_CACHE_SCHEMA_VERSION,
    ) -> None:
        self.root = resolve_data_root(root)
        self.schema_version = schema_version

    @property
    def cache_dir(self) -> Path:
        return self.root / "identity" / f"v{self.schema_version}"

    def _path(self, input_key: str) -> Path:
        return self.cache_dir / f"{encode_cache_key(input_key)}.json"

    def get(self, input_key: str) -> ResolvedIdentity | None:
        path = self._path(input_key)
        if not path.exists():
            return None
        try:
            return ResolvedIdentity.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            # Stale/corrupt entry (schema drift within a version, or a truncated
            # write). Treat as a miss so the pipeline re-resolves instead of dying.
            return None

    def put(self, identity: ResolvedIdentity) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(identity.input_key)
        path.write_text(identity.model_dump_json(indent=2), encoding="utf-8")
        return path

    def resolved_mbids(self) -> list[str]:
        """Every distinct MBID this cache has resolved, in sorted order.

        The cache only ever ``put``s terminal MBID resolutions (the terminal-MBID-
        only invariant), so each stored file carries a resolved MBID; a file that
        no longer validates is skipped like any other stale entry. This is the
        MBID source for the AcousticBrainz feature build (#87 AC-E) — no re-run of
        the identity waterfall needed. Empty when the cache dir does not exist."""
        if not self.cache_dir.exists():
            return []
        mbids: set[str] = set()
        for path in self.cache_dir.glob("*.json"):
            try:
                ident = ResolvedIdentity.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValidationError, OSError):
                continue
            if ident.mbid:
                mbids.add(ident.mbid)
        return sorted(mbids)


# --------------------------------------------------------------------------- #
# Resolution report
# --------------------------------------------------------------------------- #


@dataclass
class ResolutionReport:
    """Outcome of resolving a batch of tracks. ``identities`` is keyed by input
    canonical id (deduplicated). Counts and coverage derive from it so there is
    one source of truth."""

    identities: dict[str, ResolvedIdentity]

    @property
    def n_unique(self) -> int:
        return len(self.identities)

    @property
    def counts(self) -> dict[str, int]:
        """Per-level breakdown (the transparency ledger)."""
        out: dict[str, int] = {"mbid": 0, "isrc": 0, "spotify": 0, "name": 0}
        for ident in self.identities.values():
            out[ident.level] += 1
        return out

    @property
    def mbid_coverage(self) -> float:
        """Fraction of unique tracks resolved to an MBID. Feeds
        ``generated_from.coverage_per_category``. 0.0 on empty input."""
        if not self.identities:
            return 0.0
        return self.counts["mbid"] / len(self.identities)

    @property
    def unresolved(self) -> list[str]:
        """Input keys that did not reach an MBID — flagged, never dropped."""
        return [key for key, ident in self.identities.items() if not ident.resolved]


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #


class IdentityResolver:
    """Walks tracks up the spotify_id -> ISRC -> MBID waterfall.

    ``isrc_index`` is required; ``spotify_source`` is optional (the spotify->ISRC
    leg is skipped when absent); ``cache`` is optional (memoises resolution
    across runs). ``ab_covered`` is an optional predicate marking MBIDs that have
    AcousticBrainz features — it steers :func:`disambiguate_mbids` toward the
    join we actually need when an ISRC maps to several recordings (#87 AC-B)."""

    def __init__(
        self,
        isrc_index: IsrcMbidIndex,
        *,
        spotify_source: SpotifyIsrcSource | None = None,
        cache: IdentityCache | None = None,
        ab_covered: AbCoverage | None = None,
    ) -> None:
        self.isrc_index = isrc_index
        self.spotify_source = spotify_source
        self.cache = cache
        self.ab_covered = ab_covered

    def resolve(self, track: TrackRef) -> ResolvedIdentity:
        key = canonical_track_id(track)
        if self.cache is not None:
            hit = self.cache.get(key)
            # Only a terminal (MBID-level) hit is trustworthy across runs. A cached
            # sub-MBID result records "the index/source I had couldn't resolve this"
            # — a negative that a since-grown index may now satisfy. Serving it would
            # freeze the miss (#87: an isrc-level cache written before the MB index
            # existed masked the whole 6M-pair join and pinned coverage at 0). Re-walk
            # partials; the expensive spotify->ISRC leg has its own durable cache, so
            # the retry is cheap in-memory lookups.
            if hit is not None and hit.resolved:
                return hit
        identity = self._waterfall(track, key)
        # Persist only terminal resolutions. A partial is provisional and must be
        # retried next run against whatever data is then available.
        if self.cache is not None and identity.resolved:
            self.cache.put(identity)
        return identity

    def _waterfall(self, track: TrackRef, key: str) -> ResolvedIdentity:
        spotify_id, isrc, mbid = track.spotify_id, track.isrc, track.mbid
        if mbid is None:
            if isrc is None and spotify_id is not None and self.spotify_source is not None:
                isrc = self.spotify_source.lookup(spotify_id) or None
            if isrc is not None:
                candidates = self.isrc_index.lookup_all(isrc)
                mbid = disambiguate_mbids(candidates, ab_covered=self.ab_covered)

        if mbid is not None:
            level: ResolutionLevel = "mbid"
        elif isrc is not None:
            level = "isrc"
        elif spotify_id is not None:
            level = "spotify"
        else:
            level = "name"

        return ResolvedIdentity(
            input_key=key,
            spotify_id=spotify_id,
            isrc=isrc,
            mbid=mbid,
            name=track.name,
            artist=track.artist,
            level=level,
        )

    def resolve_all(self, tracks: Sequence[TrackRef]) -> ResolutionReport:
        """Resolve every track, deduplicated by input canonical id."""
        identities: dict[str, ResolvedIdentity] = {}
        for track in tracks:
            key = canonical_track_id(track)
            if key in identities:
                continue
            identities[key] = self.resolve(track)
        return ResolutionReport(identities=identities)


def to_metadata_records(report: ResolutionReport, *, now: datetime) -> list[TrackMetadataRecord]:
    """Resolved identities as anonymous shared-store records (identity columns
    only — enrichers fill ``audio_features``/``tags`` later). Keyed by the
    *resolved* canonical id so the store deduplicates across users by MBID."""
    records: list[TrackMetadataRecord] = []
    for ident in report.identities.values():
        ref = ident.to_track_ref()
        records.append(
            TrackMetadataRecord(
                track_id=canonical_track_id(ref),
                spotify_id=ident.spotify_id,
                isrc=ident.isrc,
                mbid=ident.mbid,
                name=ident.name,
                artist=ident.artist,
                fetched_at=now,
            )
        )
    return records
