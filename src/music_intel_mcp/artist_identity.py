"""Artist identity resolution — the spotify artist URI -> artist MBID waterfall
(#102), the artist-level sibling of the track waterfall in :mod:`identity`.

The explicit-preference layer (#97) carries Spotify artist identity as it
comes off the export: a URI (``followed_artists`` / ``banned_artists``) or a
bare name only (Marquee, which the source never attaches a URI to). Anti-
bubble scoring needs a stable cross-source join key for artists the same way
track scoring needs an MBID — this module walks each :class:`~music_intel_mcp.
models.ArtistRef` (or bare Marquee name) up that waterfall::

    uri -> MBID via the MB dump (artist/url/l_artist_url join)   -> done
    name only, no measured-precision name-match rung available  -> unresolved

Two CONTEXT.md invariants carried over from the track waterfall:

- **Transparent rejection** — an artist that cannot reach an MBID is *flagged
  and counted*, never silently dropped. :class:`ArtistResolutionReport` records
  the level each artist reached.
- **No re-resolution** — resolved identities are written to a local
  :class:`ArtistIdentityCache` keyed by the *input* identity and reused on
  re-run (terminal-MBID-only, mirroring decision dddc4d90).

**Name-match rung — deliberately unshipped.** Marquee entries carry no URI, so
the only path to an MBID for them is a name join against the MB artist table.
Decision 2b262dc9 already rejected exactly this move at the track level as a
precision minefield (transliteration, aliases, "feat." variants); nothing
about the artist table makes that safer. :class:`ArtistIdentityResolver`
accepts an optional ``name_index`` seam for a future exact-match rung, but
defaults to ``None`` — a name-only ref is honestly reported at the ``name``
level (unresolved to MBID) rather than silently fuzzy-joined. Shipping a name-
match rung requires the same measured-precision evaluation the track-level
decision demanded, on real data this environment does not have (see the MB
dump note below).

The MusicBrainz dump lives OUTSIDE the repo (env-pointed), same as the track
index; tests use the in-memory index and a tiny synthetic TSV — never the live
dump or any API. No MusicBrainz dump was present in the environment this
module was built in, so unlike the track-level ISRC->MBID join (measured at
55.3% coverage on a July-2026 dump build — see CONTEXT.md), this module's
artist-level coverage has **no measured number of its own**; live measurement
is deferred to a follow-up issue once a dump is available.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import ArtistRef, MarqueeEntry
from .shared_store import encode_cache_key
from .store import resolve_data_root

# Env metadata for the production index (values/paths live in .env / host env).
# Explicit index path > MUSICBRAINZ_ARTIST_INDEX > $MUSICBRAINZ_DUMP_DIR/<default>.
_MB_ARTIST_INDEX_ENV = "MUSICBRAINZ_ARTIST_INDEX"
_MB_DUMP_DIR_ENV = "MUSICBRAINZ_DUMP_DIR"
_DEFAULT_INDEX_FILENAME = "artist_uri_to_mbid.tsv"

ArtistResolutionLevel = Literal["mbid", "name"]

# Bump whenever ``ResolvedArtist``'s schema or the resolution semantics change,
# so on-disk caches written by an older schema are ignored (mirrors
# IDENTITY_CACHE_SCHEMA_VERSION in identity.py).
ARTIST_IDENTITY_CACHE_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Canonical artist identity
# --------------------------------------------------------------------------- #


def canonical_artist_key(*, uri: str | None, name: str) -> str:
    """Stable string key for an artist ref, mirroring ``canonical_track_id``.

    URI (when present) beats name — it is the one identity the source itself
    guarantees is stable across a re-export; a bare name falls back to a
    casefolded key so re-runs of Marquee-only entries still cache-hit."""
    if uri:
        return f"uri:{uri}"
    return f"name:{name.casefold()}"


# --------------------------------------------------------------------------- #
# Resolved artist record
# --------------------------------------------------------------------------- #


class ResolvedArtist(BaseModel):
    """The outcome of resolving one artist ref. ``level`` is the deepest
    waterfall rung reached; ``input_key`` is the canonical id of the
    *pre-resolution* ref (the cache key). Anonymous by construction —
    ``extra='forbid'`` blocks any per-user field from leaking into a shared
    identity record, same discipline as ``ResolvedIdentity``."""

    model_config = ConfigDict(extra="forbid")

    input_key: str
    uri: str | None = None
    mbid: str | None = None
    name: str
    level: ArtistResolutionLevel

    @property
    def resolved(self) -> bool:
        """True once the waterfall reached an MBID."""
        return self.level == "mbid"


# --------------------------------------------------------------------------- #
# Lookup sources
# --------------------------------------------------------------------------- #


class InMemoryArtistUrlMbidIndex:
    """Dict-backed artist URI -> MBID index for tests and small extracts.

    ``lookups`` records each query so tests can assert the dump was not
    re-walked on a cache hit (mirrors ``InMemoryIsrcMbidIndex``)."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self._map = dict(mapping or {})
        self.lookups: list[str] = []

    def lookup(self, uri: str) -> str | None:
        self.lookups.append(uri)
        return self._map.get(uri)


class MusicBrainzArtistUrlIndex:
    """Spotify artist URI -> artist MBID index read from a TSV derived from the
    MB dump (built by :func:`music_intel_mcp.mb_dump.build_artist_mbid_tsv`).

    The TSV is a prebuilt ``<spotify:artist:id>\\t<mbid>`` extract (one pair
    per line, ``#`` comments allowed) — the raw dump's ``url``/``l_artist_url``
    /``artist`` tables are far too large to scan per analysis. Path
    resolution: explicit arg > ``MUSICBRAINZ_ARTIST_INDEX`` >
    ``$MUSICBRAINZ_DUMP_DIR/artist_uri_to_mbid.tsv``. A missing file yields an
    empty index (every lookup misses) rather than an error, mirroring
    ``MusicBrainzIsrcIndex``. Loaded once, lazily, on first lookup."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._explicit = Path(path) if path is not None else None
        self._map: dict[str, str] | None = None

    def _resolve_path(self) -> Path | None:
        if self._explicit is not None:
            return self._explicit
        env = os.environ.get(_MB_ARTIST_INDEX_ENV)
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
                    uri, mbid = parts[0].strip(), parts[1].strip()
                    if uri and mbid:
                        mapping[uri] = mbid
        self._map = mapping
        return mapping

    def lookup(self, uri: str) -> str | None:
        return self._load().get(uri)


# --------------------------------------------------------------------------- #
# Identity cache (no re-resolution on re-run)
# --------------------------------------------------------------------------- #


class ArtistIdentityCache:
    """Local cache of resolved artist identities
    (``<data root>/artist_identity/v<N>/<key>.json``). Mirrors
    :class:`~music_intel_mcp.identity.IdentityCache`'s terminal-MBID-only
    discipline (decision dddc4d90): only a resolved (MBID-level) identity is
    ever cached, so a since-grown index is never masked by a frozen miss."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        schema_version: int = ARTIST_IDENTITY_CACHE_SCHEMA_VERSION,
    ) -> None:
        self.root = resolve_data_root(root)
        self.schema_version = schema_version

    @property
    def cache_dir(self) -> Path:
        return self.root / "artist_identity" / f"v{self.schema_version}"

    def _path(self, input_key: str) -> Path:
        return self.cache_dir / f"{encode_cache_key(input_key)}.json"

    def get(self, input_key: str) -> ResolvedArtist | None:
        path = self._path(input_key)
        if not path.exists():
            return None
        try:
            return ResolvedArtist.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return None

    def put(self, identity: ResolvedArtist) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(identity.input_key)
        path.write_text(identity.model_dump_json(indent=2), encoding="utf-8")
        return path


# --------------------------------------------------------------------------- #
# Resolution report
# --------------------------------------------------------------------------- #


@dataclass
class ArtistResolutionReport:
    """Outcome of resolving a batch of artist refs. ``identities`` is keyed by
    input canonical id (deduplicated). Counts and coverage derive from it so
    there is one source of truth, mirroring ``ResolutionReport``."""

    identities: dict[str, ResolvedArtist]

    @property
    def n_unique(self) -> int:
        return len(self.identities)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"mbid": 0, "name": 0}
        for ident in self.identities.values():
            out[ident.level] += 1
        return out

    @property
    def mbid_coverage(self) -> float:
        """Fraction of unique artists resolved to an MBID. 0.0 on empty input.

        No measured baseline exists for this number (see module docstring) —
        it is report-only, printed for observability, never treated as a
        validated coverage figure the way the track-level 55.3% is."""
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


class ArtistIdentityResolver:
    """Walks artist refs up the spotify-uri -> MBID waterfall.

    ``url_index`` is required. ``cache`` is optional (memoises resolution
    across runs, terminal-MBID-only). ``name_index`` is an optional exact-
    casefold-match seam for a future, precision-measured name rung (see module
    docstring) — left unset, every name-only ref is honestly reported at the
    ``name`` level rather than fuzzy-joined."""

    def __init__(
        self,
        url_index: MusicBrainzArtistUrlIndex | InMemoryArtistUrlMbidIndex,
        *,
        cache: ArtistIdentityCache | None = None,
        name_index: dict[str, str] | None = None,
    ) -> None:
        self.url_index = url_index
        self.cache = cache
        self.name_index = name_index

    def resolve(self, *, uri: str | None, name: str) -> ResolvedArtist:
        key = canonical_artist_key(uri=uri, name=name)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None and hit.resolved:
                return hit
        identity = self._waterfall(uri=uri, name=name, key=key)
        if self.cache is not None and identity.resolved:
            self.cache.put(identity)
        return identity

    def _waterfall(self, *, uri: str | None, name: str, key: str) -> ResolvedArtist:
        mbid: str | None = None
        if uri is not None:
            mbid = self.url_index.lookup(uri)
        if mbid is None and self.name_index is not None:
            mbid = self.name_index.get(name.casefold())

        level: ArtistResolutionLevel = "mbid" if mbid is not None else "name"
        return ResolvedArtist(input_key=key, uri=uri, mbid=mbid, name=name, level=level)

    def resolve_ref(self, artist: ArtistRef) -> ResolvedArtist:
        return self.resolve(uri=artist.uri, name=artist.name)

    def resolve_marquee(self, entry: MarqueeEntry) -> ResolvedArtist:
        return self.resolve(uri=None, name=entry.artist_name)

    def resolve_all(
        self,
        artists: Sequence[ArtistRef],
        marquee: Sequence[MarqueeEntry] = (),
    ) -> ArtistResolutionReport:
        """Resolve every artist ref plus every Marquee entry, deduplicated by
        input canonical id (a followed artist that also appears in Marquee
        resolves once)."""
        identities: dict[str, ResolvedArtist] = {}
        for artist in artists:
            key = canonical_artist_key(uri=artist.uri, name=artist.name)
            if key in identities:
                continue
            identities[key] = self.resolve_ref(artist)
        for entry in marquee:
            key = canonical_artist_key(uri=None, name=entry.artist_name)
            if key in identities:
                continue
            identities[key] = self.resolve_marquee(entry)
        return ArtistResolutionReport(identities=identities)
