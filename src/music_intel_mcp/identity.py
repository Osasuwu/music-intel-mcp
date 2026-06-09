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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

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


@runtime_checkable
class IsrcMbidIndex(Protocol):
    """ISRC -> recording MBID lookup (the MusicBrainz dump leg)."""

    def lookup(self, isrc: str) -> str | None: ...


@runtime_checkable
class SpotifyIsrcSource(Protocol):
    """spotify_id -> ISRC lookup (the Spotify-metadata leg)."""

    def lookup(self, spotify_id: str) -> str | None: ...


class InMemoryIsrcMbidIndex:
    """Dict-backed :class:`IsrcMbidIndex` for tests and small extracts.
    ``lookups`` records each query so tests can assert the dump was not
    re-walked on a cache hit."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._map = dict(mapping or {})
        self.lookups: list[str] = []

    def lookup(self, isrc: str) -> str | None:
        self.lookups.append(isrc)
        return self._map.get(isrc)


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
    analysis. Path resolution: explicit arg > ``MUSICBRAINZ_ISRC_INDEX`` >
    ``$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv``. A missing file yields an empty
    index (every lookup misses) rather than an error — honest low coverage beats
    a crash when the dump isn't installed. Loaded once, lazily, on first lookup.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._explicit = Path(path) if path is not None else None
        self._map: dict[str, str] | None = None

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

    def _load(self) -> dict[str, str]:
        if self._map is not None:
            return self._map
        mapping: dict[str, str] = {}
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
                        mapping[isrc] = mbid
        self._map = mapping
        return mapping

    def lookup(self, isrc: str) -> str | None:
        return self._load().get(isrc)


# --------------------------------------------------------------------------- #
# Identity cache (no re-resolution on re-run)
# --------------------------------------------------------------------------- #


class IdentityCache:
    """Local cache of resolved identities (``<data root>/identity/<key>.json``).

    Keyed by the *input* canonical id (what history knew) so a re-run with the
    same history skips the waterfall entirely. Regenerable and gitignored, like
    the metadata cache; filenames are percent-encoded for any canonical id."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_data_root(root)

    @property
    def cache_dir(self) -> Path:
        return self.root / "identity"

    def _path(self, input_key: str) -> Path:
        return self.cache_dir / f"{encode_cache_key(input_key)}.json"

    def get(self, input_key: str) -> ResolvedIdentity | None:
        path = self._path(input_key)
        if not path.exists():
            return None
        return ResolvedIdentity.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, identity: ResolvedIdentity) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(identity.input_key)
        path.write_text(identity.model_dump_json(indent=2), encoding="utf-8")
        return path


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
    across runs)."""

    def __init__(
        self,
        isrc_index: IsrcMbidIndex,
        *,
        spotify_source: SpotifyIsrcSource | None = None,
        cache: IdentityCache | None = None,
    ) -> None:
        self.isrc_index = isrc_index
        self.spotify_source = spotify_source
        self.cache = cache

    def resolve(self, track: TrackRef) -> ResolvedIdentity:
        key = canonical_track_id(track)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        identity = self._waterfall(track, key)
        if self.cache is not None:
            self.cache.put(identity)
        return identity

    def _waterfall(self, track: TrackRef, key: str) -> ResolvedIdentity:
        spotify_id, isrc, mbid = track.spotify_id, track.isrc, track.mbid
        if mbid is None:
            if isrc is None and spotify_id is not None and self.spotify_source is not None:
                isrc = self.spotify_source.lookup(spotify_id) or None
            if isrc is not None:
                mbid = self.isrc_index.lookup(isrc) or None

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
