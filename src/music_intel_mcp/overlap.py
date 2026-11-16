"""Cross-user overlap measurement (#266) -- the number that gates #265.

The operator exports salted hashes of their tracks with >=``min_valid_plays``
valid plays; the friend compares that file with their own local history and
sends back only the printed counts. Two identity columns are compared:

- ``spotify_id`` -- the Spotify track id (the column #265 is judged on);
- ``name_artist`` -- normalised track name + album artist, the bridge for
  tracks whose id differs between releases.

The salt gives no protection against the friend: the file carries it, so a
friend can hash guesses. It only keeps the file from being a plain track list.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from .live_identity import normalize_track_name
from .models import ListenEvent, TrackRef
from .replay_queue import MIN_VALID_PLAYS, _index_events

COLUMN_SPOTIFY_ID = "spotify_id"
COLUMN_NAME_ARTIST = "name_artist"
COLUMNS = (COLUMN_SPOTIFY_ID, COLUMN_NAME_ARTIST)
COLUMN_IDENTITY = {
    COLUMN_SPOTIFY_ID: "Spotify track id",
    COLUMN_NAME_ARTIST: "normalised track name + album artist",
}
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ColumnHashes:
    hashes: frozenset[str]
    not_comparable: int


@dataclass(frozen=True)
class ColumnOverlap:
    column: str
    size_a: int
    size_b: int
    intersection: int
    denominator: int
    coefficient: float | None
    not_comparable_a: int
    not_comparable_b: int


def new_salt() -> str:
    return secrets.token_hex(16)


def _hash(salt: str, column: str, value: str) -> str:
    return hashlib.sha256(f"{salt}\x1f{column}\x1f{value}".encode()).hexdigest()


def _column_value(column: str, track: TrackRef) -> str | None:
    if column == COLUMN_SPOTIFY_ID:
        return track.spotify_id or None
    if not track.name.strip() or not track.artist.strip():
        return None
    return normalize_track_name(track.name, track.artist)


def hash_columns(
    events: list[ListenEvent], *, salt: str, min_valid_plays: int = MIN_VALID_PLAYS
) -> dict[str, ColumnHashes]:
    """Per column: salted hashes of the eligible tracks, and how many eligible
    tracks have no value on that column (counted, never dropped silently)."""
    counts, reps, _ = _index_events(events)
    eligible = [reps[cid] for cid, n in counts.items() if n >= min_valid_plays]
    result: dict[str, ColumnHashes] = {}
    for column in COLUMNS:
        hashes: set[str] = set()
        missing = 0
        for track in eligible:
            value = _column_value(column, track)
            if value is None:
                missing += 1
            else:
                hashes.add(_hash(salt, column, value))
        result[column] = ColumnHashes(frozenset(hashes), missing)
    return result


def write_export(
    path: Path, events: list[ListenEvent], *, salt: str, min_valid_plays: int = MIN_VALID_PLAYS
) -> dict[str, ColumnHashes]:
    columns = hash_columns(events, salt=salt, min_valid_plays=min_valid_plays)
    payload = {
        "version": FORMAT_VERSION,
        "salt": salt,
        "min_valid_plays": min_valid_plays,
        "columns": {
            c: {"not_comparable": columns[c].not_comparable, "hashes": sorted(columns[c].hashes)}
            for c in COLUMNS
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return columns


def read_export(path: Path) -> tuple[str, int, dict[str, ColumnHashes]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported export version {data['version']!r}")
        columns = {
            c: ColumnHashes(
                frozenset(data["columns"][c]["hashes"]), int(data["columns"][c]["not_comparable"])
            )
            for c in COLUMNS
        }
        return str(data["salt"]), int(data["min_valid_plays"]), columns
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not a valid overlap export: {path}") from exc


def compare(own: dict[str, ColumnHashes], other: dict[str, ColumnHashes]) -> list[ColumnOverlap]:
    """``own`` is side A (the file), ``other`` side B (local history). The
    coefficient is |A ∩ B| / min(|A|, |B|); ``None`` when a set is empty."""
    out: list[ColumnOverlap] = []
    for column in COLUMNS:
        a, b = own[column], other[column]
        inter = len(a.hashes & b.hashes)
        denom = min(len(a.hashes), len(b.hashes))
        out.append(
            ColumnOverlap(
                column=column,
                size_a=len(a.hashes),
                size_b=len(b.hashes),
                intersection=inter,
                denominator=denom,
                coefficient=inter / denom if denom else None,
                not_comparable_a=a.not_comparable,
                not_comparable_b=b.not_comparable,
            )
        )
    return out


def format_report(results: list[ColumnOverlap]) -> str:
    lines: list[str] = []
    for r in results:
        coeff = "n/a (empty set)" if r.coefficient is None else f"{r.coefficient:.4f}"
        lines.append(
            f"[{r.column}] identity: {COLUMN_IDENTITY[r.column]}\n"
            f"  overlap coefficient: {coeff}\n"
            f"  |A| (file): {r.size_a}  |B| (local): {r.size_b}\n"
            f"  intersection: {r.intersection}  denominator min(|A|,|B|): {r.denominator}\n"
            f"  not comparable on this column: A {r.not_comparable_a}, B {r.not_comparable_b}"
        )
    return "\n".join(lines)
