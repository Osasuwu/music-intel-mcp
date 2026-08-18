"""Spotify **Account Data** ``StreamingHistory_music_*.json`` ingestion (#103).

The Account Data export (distinct from the Extended Streaming History export
handled by :mod:`.spotify_extended`) ships a lightweight play log alongside the
explicit-preference signals :mod:`.account_data` already imports. Row shape::

    endTime      "YYYY-MM-DD HH:MM"   (see timezone note below)
    artistName   str
    trackName    str
    msPlayed     int

This importer exists to extend history *past* the tail of an existing
``spotify_extended`` (ESH) import — Account Data typically covers the last ~12
months and is available immediately, while a fresh ESH export takes weeks to
arrive from Spotify's privacy dashboard. It is deliberately narrow: only rows
after the newest existing ``spotify_extended`` play are imported (time-cutoff
dedup), since Account Data carries no per-play context (no ``skipped``,
``incognito``, ``conn_country``) and duplicating the ESH range would just
downgrade already-rich rows.

**Timezone (verified against the real export, not assumed):** ``endTime`` was
checked against 8,197 unambiguous plays matched by (track, artist) against the
same user's ``spotify_extended`` ``ts`` across all 12 overlap months — every
pair agreed within 60 seconds with no seasonal (DST-like) drift. ``endTime`` is
**already UTC**, minute-truncated. It is parsed directly as UTC; no timezone
lookup or offset conversion is applied (supersedes the earlier zone-reuse
assumption floated before this was checked against real data).

**Identity (Spotify-only scope, decision — see CONTEXT.md):** Account Data rows
carry no ``spotify_track_uri``, only ``trackName``/``artistName``. To let these
rows share a canonical id with the richer ESH rows for the same play (rather
than permanently forking onto the ``name:`` identity rung), each row is looked
up by ``(name.casefold(), artist.casefold())`` against a map built from the
user's existing ``spotify_extended`` history. A name/artist pair that maps to
more than one ``spotify_id`` in that history is dropped from the lookup
(ambiguous — resolving it either way could misattribute a play) rather than
guessed. This resolution is deliberately scoped to Spotify-sourced history
only; generalizing identity resolution across arbitrary sources (YouTube
Music, Apple Music, ...) is out of scope for this importer.

**Lossless projection, drop-accounting (same discipline as #89 decisions
83bd6f76 / 9576bde1):** no validity filtering at import time; every row this
importer does not import (unparseable timestamp, no identity, before the
cutoff) is counted on :class:`AccountHistoryStats`, never silent.

**No reconciliation on a future ESH re-import (deferred, follow-up filed):**
if the user later imports a new/extended ESH export whose range overlaps rows
already imported by this module, the ``spotify_account_history`` source is not
covered by :data:`~.spotify_extended.SUPERSEDES` and the two will coexist as
separate events for the same play.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .ingest import dedup_events
from .models import ListenEvent, PlayContext, TrackRef

SOURCE = "spotify_account_history"

_MAX_SKIP_SAMPLES = 5
_FILE_GLOB = "StreamingHistory_music_*.json"
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
_SPOTIFY_EXTENDED_SOURCE = "spotify_extended"


@dataclass
class AccountHistoryStats:
    """What ``load_account_history_*`` dropped or resolved, surfaced by the CLI
    rather than left silent."""

    skipped_no_identity: int = 0
    skipped_unparseable: int = 0
    unparseable_samples: list[str] = field(default_factory=list)
    dropped_before_cutoff: int = 0
    resolved_via_spotify_lookup: int = 0
    ambiguous_lookup_keys: int = 0

    def _note_unparseable(self, raw: str) -> None:
        self.skipped_unparseable += 1
        if len(self.unparseable_samples) < _MAX_SKIP_SAMPLES:
            self.unparseable_samples.append(raw)

    @property
    def total_skipped(self) -> int:
        return self.skipped_no_identity + self.skipped_unparseable + self.dropped_before_cutoff


def parse_account_history_timestamp(raw: str) -> datetime:
    """Parse one ``endTime`` cell (``"2026-07-03 09:30"``) to a UTC-aware
    datetime. ``endTime`` is already UTC (verified against the real export —
    see module docstring), so this is a direct parse with no offset applied.
    Raises ``ValueError`` on any other shape; callers catch it and skip+count
    the row."""
    return datetime.strptime(raw.strip(), _TIMESTAMP_FORMAT).replace(tzinfo=UTC)


def _str(value: object) -> str:
    """Normalise a JSON value to a stripped string (``None`` -> ``""``)."""
    return "" if value is None else str(value).strip()


def _lookup_key(name: str, artist: str) -> tuple[str, str]:
    return (name.casefold(), artist.casefold())


def build_spotify_id_lookup(
    events: Iterable[ListenEvent],
) -> tuple[dict[tuple[str, str], str], int]:
    """Build a ``(name, artist) -> spotify_id`` map from ``spotify_extended``
    events, for resolving Account Data rows onto the richer ``spotify:<id>``
    canonical rung instead of the ``name:`` fallback. A key that maps to more
    than one ``spotify_id`` across the source history is dropped (ambiguous)
    rather than guessed — see module docstring. Returns ``(lookup,
    ambiguous_key_count)``."""
    lookup: dict[tuple[str, str], str] = {}
    ambiguous: set[tuple[str, str]] = set()
    for event in events:
        if event.source != _SPOTIFY_EXTENDED_SOURCE or not event.track.spotify_id:
            continue
        key = _lookup_key(event.track.name, event.track.artist)
        if key in ambiguous:
            continue
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = event.track.spotify_id
        elif existing != event.track.spotify_id:
            del lookup[key]
            ambiguous.add(key)
    return lookup, len(ambiguous)


def max_spotify_extended_played_at(events: Iterable[ListenEvent]) -> datetime | None:
    """Newest ``played_at`` across ``spotify_extended`` events, or ``None`` if
    there is no such history yet (in which case nothing is cut off)."""
    timestamps = [e.played_at for e in events if e.source == _SPOTIFY_EXTENDED_SOURCE]
    return max(timestamps) if timestamps else None


def _row_to_event(
    row: dict,
    *,
    lookup: dict[tuple[str, str], str],
    cutoff: datetime | None,
    stats: AccountHistoryStats,
) -> ListenEvent | None:
    """Map one Account Data streaming-history row to a ``ListenEvent``; return
    ``None`` (and tally the reason on ``stats``) to skip it."""
    name = _str(row.get("trackName"))
    artist = _str(row.get("artistName"))
    if not name and not artist:
        stats.skipped_no_identity += 1
        return None

    try:
        played_at = parse_account_history_timestamp(_str(row.get("endTime")))
    except ValueError:
        stats._note_unparseable(_str(row.get("endTime")))
        return None

    if cutoff is not None and played_at <= cutoff:
        stats.dropped_before_cutoff += 1
        return None

    spotify_id = lookup.get(_lookup_key(name, artist))
    if spotify_id is not None:
        stats.resolved_via_spotify_lookup += 1

    ms_played = row.get("msPlayed")
    return ListenEvent(
        track=TrackRef(spotify_id=spotify_id, name=name, artist=artist),
        played_at=played_at,
        source=SOURCE,
        context=PlayContext(ms_played=ms_played if isinstance(ms_played, int) else None),
    )


def load_account_history_file(
    path: str | Path,
    *,
    lookup: dict[tuple[str, str], str] | None = None,
    cutoff: datetime | None = None,
    stats: AccountHistoryStats | None = None,
) -> list[ListenEvent]:
    """Convert every element of one ``StreamingHistory_music_*.json`` array to a
    ``ListenEvent``. Pass a shared ``stats`` to accumulate counts across many
    files."""
    stats = stats if stats is not None else AccountHistoryStats()
    lookup = lookup if lookup is not None else {}
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        event
        for row in rows
        if (event := _row_to_event(row, lookup=lookup, cutoff=cutoff, stats=stats))
    ]


def load_account_history_dir(
    directory: str | Path,
    *,
    existing_events: Iterable[ListenEvent] = (),
    pattern: str = _FILE_GLOB,
    stats: AccountHistoryStats | None = None,
) -> list[ListenEvent]:
    """Load and merge every Account Data streaming-history JSON under
    ``directory`` into one deduped, time-sorted history extending past the
    newest existing ``spotify_extended`` play. ``existing_events`` (typically
    the current ``history.jsonl``) supplies both the spotify_id lookup and the
    cutoff; pass none to import everything (no ESH history yet). Idempotent:
    re-running against the same ``existing_events`` yields the same list."""
    root = Path(directory)
    stats = stats if stats is not None else AccountHistoryStats()
    existing = list(existing_events)
    lookup, ambiguous_count = build_spotify_id_lookup(existing)
    stats.ambiguous_lookup_keys = ambiguous_count
    cutoff = max_spotify_extended_played_at(existing)
    events: list[ListenEvent] = []
    for jsonfile in sorted(root.glob(pattern)):
        events.extend(
            load_account_history_file(jsonfile, lookup=lookup, cutoff=cutoff, stats=stats)
        )
    return dedup_events(events)
