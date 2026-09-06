"""Google Takeout ``watch-history.json`` importer for YouTube Music (#164).

One pilot participant's history lives in YouTube Music, not Spotify. The
browser-URL replay driver is deferred post-pilot (decision d6c9ffe0); this
importer ships only the import side so the participant gets a
history-derived profile plus passive-capture timbre.

Takeout ships the whole watch history as a single JSON array — one entry per
watched video, oldest first (Google's real export order is newest-first; this
importer does not assume an order and dedups/sorts via ``dedup_events``).
Only entries with ``header == "YouTube Music"`` are music; everything else
(regular YouTube video/short watches) is a different ``header`` value and is
skipped and counted, never imported.

Row shape (fields this adapter reads; the export carries a few more)::

    header       "YouTube Music" for a music play; anything else is skipped
    title        "Watched <title>", or exactly "Watched a video that has
                 been removed" for a since-removed video (no titleUrl)
    titleUrl     "https://music.youtube.com/watch?v=<id>" — video id lives in
                 the "v" query param; absent for a removed video
    subtitles    [{"name": "<Artist> - Topic", "url": ...}] — the channel
                 name, which for an Art Track / auto-generated upload carries
                 a trailing " - Topic" suffix (stripped here)
    time         ISO-8601 UTC "...Z" timestamp, same shape as Spotify's ts

**No-duration validity rule (documented in CONTEXT.md):** Takeout carries no
play-duration signal at all — a YouTube Music entry counts as a play with no
``ms_played``, so ``ListenEvent.context`` is always ``None`` for this source.

**Self-referential supersede:** no prior source overlaps YouTube Music watch
history, so ``SUPERSEDES`` only ever displaces a prior run of this importer
(re-import is idempotent via ``dedup_events``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .ingest import dedup_events
from .models import ListenEvent, TrackRef

SOURCE = "youtube_music"
SUPERSEDES = frozenset({SOURCE})

_MAX_SKIP_SAMPLES = 5
_MUSIC_HEADER = "YouTube Music"
_REMOVED_TITLE = "Watched a video that has been removed"
_WATCHED_PREFIX = "Watched "
_TOPIC_SUFFIX = " - Topic"


@dataclass
class YoutubeMusicStats:
    """What ``load_watch_history_file`` dropped, surfaced by the CLI rather
    than left silent. Every row the lossless projection excludes is counted
    (never silently dropped): non-music entries (any other ``header``),
    removed videos (no ``titleUrl`` to extract a video id from), entries
    whose ``titleUrl`` carries no ``v`` query param, and unparseable
    timestamps."""

    skipped_non_music: int = 0
    skipped_removed: int = 0
    skipped_no_video_id: int = 0
    skipped_unparseable: int = 0
    unparseable_samples: list[str] = field(default_factory=list)

    def _note_unparseable(self, raw: str) -> None:
        self.skipped_unparseable += 1
        if len(self.unparseable_samples) < _MAX_SKIP_SAMPLES:
            self.unparseable_samples.append(raw)

    @property
    def total_skipped(self) -> int:
        return (
            self.skipped_non_music
            + self.skipped_removed
            + self.skipped_no_video_id
            + self.skipped_unparseable
        )


def parse_takeout_timestamp(raw: str) -> datetime:
    """Parse one Takeout ``time`` (``"2022-10-21T20:19:07.123Z"``) to a
    UTC-aware datetime. Mirrors ``spotify_extended.parse_spotify_timestamp``:
    the ``Z`` zulu suffix is normalised to ``+00:00`` for ``fromisoformat``; a
    value that already carries a numeric offset is honoured and coerced to
    UTC. Raises ``ValueError`` on any other shape — callers catch it and
    skip+count the row rather than fabricate a time the source never
    recorded."""
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _video_id(title_url: str | None) -> str | None:
    """Extract the ``v`` query param from a ``titleUrl``, or ``None`` if
    absent/unparseable."""
    if not title_url:
        return None
    values = parse_qs(urlparse(title_url).query).get("v")
    return values[0] if values else None


def _artist_name(row: dict) -> str:
    """The channel name from the first ``subtitles`` entry, with the
    auto-generated-upload ``" - Topic"`` suffix stripped. Empty string when
    the row carries no ``subtitles`` at all."""
    subtitles = row.get("subtitles") or []
    if not subtitles:
        return ""
    name = str(subtitles[0].get("name") or "").strip()
    if name.endswith(_TOPIC_SUFFIX):
        name = name[: -len(_TOPIC_SUFFIX)]
    return name


def _track_name(title: str) -> str:
    """Strip the ``"Watched "`` prefix Takeout puts on every title."""
    if title.startswith(_WATCHED_PREFIX):
        return title[len(_WATCHED_PREFIX) :]
    return title


def _row_to_event(row: dict, stats: YoutubeMusicStats) -> ListenEvent | None:
    """Map one Takeout row to a ``ListenEvent``; return ``None`` (and tally
    the reason on ``stats``) to skip it.

    Skipped: non-music entries (``header`` != "YouTube Music"), removed
    videos (title is the removed-video sentinel or there is no ``titleUrl``
    at all), entries whose ``titleUrl`` carries no video id, and entries
    whose ``time`` does not parse (unplaceable in time — never fabricated).
    """
    if row.get("header") != _MUSIC_HEADER:
        stats.skipped_non_music += 1
        return None

    title = str(row.get("title") or "")
    title_url = row.get("titleUrl")
    if title == _REMOVED_TITLE or not title_url:
        stats.skipped_removed += 1
        return None

    video_id = _video_id(title_url)
    if not video_id:
        stats.skipped_no_video_id += 1
        return None

    try:
        played_at = parse_takeout_timestamp(str(row.get("time") or ""))
    except ValueError:
        stats._note_unparseable(str(row.get("time")))
        return None

    return ListenEvent(
        track=TrackRef(youtube_id=video_id, name=_track_name(title), artist=_artist_name(row)),
        played_at=played_at,
        source=SOURCE,
    )


def load_watch_history_file(
    path: str | Path, *, stats: YoutubeMusicStats | None = None
) -> list[ListenEvent]:
    """Convert every element of one ``watch-history.json`` array to a
    ``ListenEvent``, deduped and time-sorted. Idempotent: re-running over the
    same export yields the same list. Pass a shared ``stats`` to receive skip
    counts."""
    stats = stats if stats is not None else YoutubeMusicStats()
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    events = [event for row in rows if (event := _row_to_event(row, stats))]
    return dedup_events(events)
