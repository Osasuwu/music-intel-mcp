"""Metadata cross-walk alias generation (#178).

The near-dup merge system (#140) can only merge two records that both have
audio embeddings. The pilot's actually-broken case is a **history-only**
key (``spotify:<id>``/``youtube:<id>``, never resolved to an MBID at
ingestion) meeting a pool record keyed ``mbid:<uuid>`` from organic
live-capture -- no audio comparison can bridge that gap.

This module re-runs the existing ``spotify: -> ISRC -> MBID`` resolution
(the same waterfall leg :mod:`identity` already implements for live
capture) over history keys that still lack an ``mbid:`` form, using only
**already-materialized indexes** -- the MusicBrainz ISRC->MBID dump and the
Spotify ISRC cache -- never a live API call at run time. Resolvable keys
get an alias line appended to the same ``aliases.jsonl`` sidecar #140
defines (``{loser, winner, tier}``, tier ``"metadata_crosswalk"``), so the
existing :func:`music_intel_mcp.store.resolve_key` machinery -- and every
consumer built on it -- picks them up for free. An ISRC resolving to more
than one MBID is reported, never aliased (ambiguity is not this step's call
to make).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .identity import IsrcMbidIndex
from .shared_store import canonical_track_id
from .store import UserStore, load_aliases, resolve_key

TIER = "metadata_crosswalk"


@dataclass(frozen=True)
class AmbiguousMatch:
    """An ISRC that resolved to more than one MBID -- reported, not aliased."""

    spotify_id: str
    isrc: str
    mbids: list[str]


@dataclass(frozen=True)
class CrosswalkResult:
    aliases_written: list[dict] = field(default_factory=list)
    ambiguous: list[AmbiguousMatch] = field(default_factory=list)


def _history_only_spotify_keys(store: UserStore, *, root_aliases: dict[str, str]) -> list[str]:
    """Distinct ``spotify:<id>`` canonical keys in history that resolve (through
    existing aliases) to something other than an ``mbid:`` key -- the set this
    step's resolution attempt is scoped to (#178 AC1)."""
    keys: set[str] = set()
    for event in store.load_history():
        cid = canonical_track_id(event.track)
        resolved = resolve_key(cid, root_aliases=root_aliases)
        if resolved.startswith("spotify:"):
            keys.add(resolved)
    return sorted(keys)


def run_metadata_crosswalk(
    root: str | Path,
    *,
    isrc_index: IsrcMbidIndex,
    spotify_isrc_lookup: Callable[[str], str | None],
) -> CrosswalkResult:
    """Emit ``spotify:<id> -> mbid:<uuid>`` alias lines for history-only keys
    resolvable through ``isrc_index`` (#178 AC1/AC2). ``spotify_isrc_lookup``
    is a cache-only ``spotify_id -> ISRC`` reader -- e.g.
    :meth:`music_intel_mcp.spotify_api.SpotifyApiIsrcSource.lookup_cached` --
    never a live network call.

    Idempotent: a ``loser`` already present in ``aliases.jsonl`` is never
    re-aliased (AC2). Ambiguous ISRC->MBID matches (``lookup_all`` returning
    more than one candidate) are reported in ``CrosswalkResult.ambiguous``
    and left unaliased (AC4).
    """
    store = UserStore(root=root)
    aliases_path = store.aliases_path
    existing = load_aliases(aliases_path)

    new_lines: list[dict] = []
    ambiguous: list[AmbiguousMatch] = []

    for key in _history_only_spotify_keys(store, root_aliases=existing):
        spotify_id = key.split(":", 1)[1]
        isrc = spotify_isrc_lookup(spotify_id)
        if not isrc:
            continue
        mbids = isrc_index.lookup_all(isrc)
        if not mbids:
            continue
        if len(mbids) > 1:
            ambiguous.append(AmbiguousMatch(spotify_id=spotify_id, isrc=isrc, mbids=sorted(mbids)))
            continue
        winner = f"mbid:{mbids[0]}"
        line = {"loser": key, "winner": winner, "tier": TIER}
        new_lines.append(line)
        existing[key] = winner

    if new_lines:
        aliases_path.parent.mkdir(parents=True, exist_ok=True)
        with aliases_path.open("a", encoding="utf-8") as fh:
            for line in new_lines:
                fh.write(json.dumps(line) + "\n")

    return CrosswalkResult(aliases_written=new_lines, ambiguous=ambiguous)
