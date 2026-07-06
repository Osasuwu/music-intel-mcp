"""Spotify **Extended Streaming History** JSON ingestion (#89).

The official Spotify data export (requested from the privacy dashboard, delivered
weeks later) is a directory of ``Streaming_History_Audio_*.json`` files — each a
JSON array where every element is one play. Unlike the thin IFTTT xlsx path
(#76), every row carries per-play context at **second resolution**: ``ms_played``,
``skipped``, ``incognito_mode``, ``conn_country``. This is the V1 data foundation
— the real ~138k-play library the locked-not-calibrated thresholds of #87 finally
get measured against.

Row shape (fields this adapter reads; the export carries more)::

    ts                                ISO-8601 UTC, second resolution, "…Z" suffix
    ms_played                         int, milliseconds the track played
    master_metadata_track_name        track title      (null on non-track rows)
    master_metadata_album_artist_name artist
    spotify_track_uri                 "spotify:track:<id>"  (null on non-track rows)
    spotify_episode_uri               set on podcast/video rows
    audiobook_uri                     set on audiobook rows
    skipped / incognito_mode / conn_country   per-play context

**Lossless projection (decision 83bd6f76):** no ``ms_played`` / ``skipped`` /
``incognito`` filtering at import — validity is an analysis-time per-pillar policy
(#90), never an import-time filter. The *only* rows dropped are non-audio-track
rows (podcast/video episodes, audiobooks) and rows carrying no identity at all;
every drop is **counted** on :class:`SpotifyExtendedStats`, never silent
(decision 9576bde1).

**Timezone (decision d2c1ff60):** ``ts`` is parsed UTC-aware and stored as-is.
``conn_country`` is carried onto :class:`~.models.PlayContext` but the
local-wall-clock conversion for temporal day-part/season buckets is deferred to
#90 (it is unobservable until enrichment produces a surviving temporal member).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .ingest import dedup_events
from .models import ListenEvent, PlayContext, TrackRef

SOURCE = "spotify_extended"
# import-spotify rebuilds history from scratch for these sources only — a Spotify
# play logged by IFTTT and by this export is the *same* play, so re-importing the
# authoritative export must supersede the thin IFTTT rows (and any prior run of
# this importer). Events from other sources (e.g. a future Last.fm scrobble) are
# preserved untouched (source-scoped supersede, decision 23fcf92c).
SUPERSEDES = frozenset({"ifttt", SOURCE})

_MAX_SKIP_SAMPLES = 5
_FILE_GLOB = "Streaming_History_Audio_*.json"
_TRACK_URI_PREFIX = "spotify:track:"


@dataclass
class SpotifyExtendedStats:
    """What ``load_spotify_extended_*`` dropped, surfaced by the CLI rather than
    left silent. ``skipped_episode`` / ``skipped_audiobook`` count the non-audio
    rows the lossless projection deliberately excludes; a spike in
    ``skipped_unparseable`` is the loud signal that the export's ``ts`` shape
    drifted (vs aborting on the first odd row)."""

    skipped_episode: int = 0
    skipped_audiobook: int = 0
    skipped_no_identity: int = 0
    skipped_unparseable: int = 0
    unparseable_samples: list[str] = field(default_factory=list)

    def _note_unparseable(self, raw: str) -> None:
        self.skipped_unparseable += 1
        if len(self.unparseable_samples) < _MAX_SKIP_SAMPLES:
            self.unparseable_samples.append(raw)

    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_episode
            + self.skipped_audiobook
            + self.skipped_no_identity
            + self.skipped_unparseable
        )


def parse_spotify_timestamp(raw: str) -> datetime:
    """Parse one export ``ts`` (``"2022-10-21T20:19:07Z"``) to a UTC-aware
    datetime. The ``Z`` zulu suffix is normalised to ``+00:00`` for
    ``fromisoformat``; a value that already carries a numeric offset is honoured
    and coerced to UTC. Raises ``ValueError`` (with the offending string) on any
    other shape — callers catch it and skip+count the row rather than fabricate a
    time the source never recorded."""
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _str(value: object) -> str:
    """Normalise a JSON value to a stripped string (``None`` -> ``""``)."""
    return "" if value is None else str(value).strip()


def _row_to_event(row: dict, stats: SpotifyExtendedStats) -> ListenEvent | None:
    """Map one export row to a ``ListenEvent``; return ``None`` (and tally the
    reason on ``stats``) to skip it.

    Skipped: podcast/video episode rows, audiobook rows (non-audio-track types),
    rows carrying neither a track uri nor a track name (no identity), and rows
    whose ``ts`` does not parse (unplaceable in time — never fabricated).
    """
    if _str(row.get("spotify_episode_uri")):
        stats.skipped_episode += 1
        return None
    if _str(row.get("audiobook_uri")):
        stats.skipped_audiobook += 1
        return None

    track_uri = _str(row.get("spotify_track_uri"))
    name = _str(row.get("master_metadata_track_name"))
    artist = _str(row.get("master_metadata_album_artist_name"))
    if not track_uri and not name:
        stats.skipped_no_identity += 1
        return None

    try:
        played_at = parse_spotify_timestamp(_str(row.get("ts")))
    except ValueError:
        stats._note_unparseable(_str(row.get("ts")))
        return None

    spotify_id = (
        track_uri[len(_TRACK_URI_PREFIX) :] if track_uri.startswith(_TRACK_URI_PREFIX) else None
    )
    ms_played = row.get("ms_played")
    return ListenEvent(
        track=TrackRef(spotify_id=spotify_id or None, name=name, artist=artist),
        played_at=played_at,
        source=SOURCE,
        context=PlayContext(
            ms_played=ms_played if isinstance(ms_played, int) else None,
            skipped=row.get("skipped"),
            incognito=row.get("incognito_mode"),
            conn_country=_str(row.get("conn_country")) or None,
        ),
    )


def load_spotify_extended_file(
    path: str | Path, *, stats: SpotifyExtendedStats | None = None
) -> list[ListenEvent]:
    """Convert every element of one ``Streaming_History_Audio_*.json`` array to a
    ``ListenEvent``. Pass a shared ``stats`` to accumulate skip counts across many
    files."""
    stats = stats if stats is not None else SpotifyExtendedStats()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [event for row in rows if (event := _row_to_event(row, stats))]


def load_spotify_extended_dir(
    directory: str | Path,
    *,
    pattern: str = _FILE_GLOB,
    stats: SpotifyExtendedStats | None = None,
) -> list[ListenEvent]:
    """Load and merge every audio-history JSON under ``directory`` into one
    deduped, time-sorted history. Idempotent: re-running yields the same list.
    Pass a ``stats`` to receive skip counts across the whole directory."""
    root = Path(directory)
    stats = stats if stats is not None else SpotifyExtendedStats()
    events: list[ListenEvent] = []
    for jsonfile in sorted(root.glob(pattern)):
        events.extend(load_spotify_extended_file(jsonfile, stats=stats))
    return dedup_events(events)
