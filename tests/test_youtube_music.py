"""Google Takeout ``watch-history.json`` importer for YouTube Music (#164).

The participant's history lives in YouTube Music, not Spotify. Takeout entries
carry title, video id, channel/artist and time but no duration — a YouTube
Music entry counts as a play with no ``ms_played`` (no duration is available).
The browser-URL replay driver is deferred post-pilot (decision d6c9ffe0); this
importer ships only the import side. Fixtures are built in-test as JSON arrays
so no real Takeout export is committed and the exact source shape is visible
here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, TrackRef
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore
from music_intel_mcp.youtube_music import (
    YoutubeMusicStats,
    load_watch_history_file,
    parse_takeout_timestamp,
)


def _music_row(
    time: str,
    title: str,
    video_id: str | None,
    artist: str,
) -> dict:
    """One YouTube Music Takeout row. ``titleUrl`` carries the video id as a
    ``v=`` query param; omit ``video_id`` to model a removed video (no
    ``titleUrl`` at all)."""
    row: dict = {
        "header": "YouTube Music",
        "title": f"Watched {title}",
        "subtitles": [{"name": f"{artist} - Topic", "url": "https://music.youtube.com/x"}],
        "time": time,
    }
    if video_id is not None:
        row["titleUrl"] = f"https://music.youtube.com/watch?v={video_id}"
    return row


def _write_export(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# timestamp parsing
# --------------------------------------------------------------------------- #


def test_parse_zulu_timestamp_to_utc():
    assert parse_takeout_timestamp("2022-10-21T20:19:07.123Z") == datetime(
        2022, 10, 21, 20, 19, 7, 123000, tzinfo=UTC
    )


# --------------------------------------------------------------------------- #
# row -> ListenEvent mapping (AC1)
# --------------------------------------------------------------------------- #


def test_music_row_maps_to_listen_event_with_youtube_id(tmp_path):
    path = _write_export(
        tmp_path / "watch-history.json",
        [_music_row("2022-10-21T20:19:07.000Z", "Maniac", "abc123", "Flower Face")],
    )
    events = load_watch_history_file(path)

    assert len(events) == 1
    e = events[0]
    assert e.track.youtube_id == "abc123"
    assert e.track.name == "Maniac"
    assert e.track.artist == "Flower Face"
    assert e.played_at == datetime(2022, 10, 21, 20, 19, 7, tzinfo=UTC)
    assert e.source == "youtube_music"
    assert e.context is None
    assert canonical_track_id(e.track) == "youtube:abc123"


def test_non_music_entries_are_dropped_and_counted(tmp_path):
    rows = [
        _music_row("2023-01-01T10:00:00.000Z", "Kept", "trackKEEP", "Artist"),
        {  # plain YouTube video watch, not YouTube Music
            "header": "YouTube",
            "title": "Watched Some Video",
            "titleUrl": "https://www.youtube.com/watch?v=vid123",
            "time": "2023-01-01T11:00:00.000Z",
        },
    ]
    path = _write_export(tmp_path / "watch-history.json", rows)
    stats = YoutubeMusicStats()
    events = load_watch_history_file(path, stats=stats)

    assert len(events) == 1
    assert events[0].track.name == "Kept"
    assert stats.skipped_non_music == 1
    assert stats.total_skipped == 1


def test_removed_video_entries_are_dropped_and_counted(tmp_path):
    rows = [
        _music_row("2023-02-01T10:00:00.000Z", "Kept", "trackKEEP", "Artist"),
        {  # removed video: sentinel title, no titleUrl at all
            "header": "YouTube Music",
            "title": "Watched a video that has been removed",
            "subtitles": [{"name": "Artist - Topic"}],
            "time": "2023-02-01T11:00:00.000Z",
        },
    ]
    path = _write_export(tmp_path / "watch-history.json", rows)
    stats = YoutubeMusicStats()
    events = load_watch_history_file(path, stats=stats)

    assert len(events) == 1
    assert stats.skipped_removed == 1


def test_entries_without_video_id_are_dropped_and_counted(tmp_path):
    rows = [
        _music_row("2023-03-01T10:00:00.000Z", "Kept", "trackKEEP", "Artist"),
        {  # titleUrl present but carries no "v" query param
            "header": "YouTube Music",
            "title": "Watched Weird",
            "titleUrl": "https://music.youtube.com/watch",
            "subtitles": [{"name": "Artist - Topic"}],
            "time": "2023-03-01T11:00:00.000Z",
        },
    ]
    path = _write_export(tmp_path / "watch-history.json", rows)
    stats = YoutubeMusicStats()
    events = load_watch_history_file(path, stats=stats)

    assert len(events) == 1
    assert stats.skipped_no_video_id == 1


def test_unparseable_timestamp_is_skipped_and_counted(tmp_path):
    rows = [
        _music_row("2023-04-01T10:00:00.000Z", "Good", "trackGOOD", "Artist"),
        _music_row("not-a-timestamp", "Bad", "trackBAD", "Artist"),
    ]
    path = _write_export(tmp_path / "watch-history.json", rows)
    stats = YoutubeMusicStats()
    events = load_watch_history_file(path, stats=stats)

    assert len(events) == 1
    assert events[0].track.name == "Good"
    assert stats.skipped_unparseable == 1
    assert stats.unparseable_samples == ["not-a-timestamp"]


def test_artist_topic_suffix_is_stripped(tmp_path):
    path = _write_export(
        tmp_path / "watch-history.json",
        [_music_row("2023-05-01T10:00:00.000Z", "Song", "trackX", "Cool Band")],
    )
    events = load_watch_history_file(path)
    assert events[0].track.artist == "Cool Band"


# --------------------------------------------------------------------------- #
# file load — dedup + idempotency (AC3)
# --------------------------------------------------------------------------- #


def test_file_load_dedups_same_track_same_second(tmp_path):
    same = _music_row("2023-06-01T15:30:45.000Z", "Dup", "trackDUP", "Artist")
    path = _write_export(tmp_path / "watch-history.json", [same, same])
    events = load_watch_history_file(path)
    assert len(events) == 1


def test_file_load_is_idempotent(tmp_path):
    row = _music_row("2023-07-01T00:00:00.000Z", "Once", "trackONCE", "Artist")
    path = _write_export(tmp_path / "watch-history.json", [row])
    once = load_watch_history_file(path)
    twice = load_watch_history_file(path)
    assert len(once) == 1
    assert once == twice


# --------------------------------------------------------------------------- #
# CLI import-youtube — self-referential supersede + idempotency (AC3)
# --------------------------------------------------------------------------- #


def test_cli_import_youtube_preserves_other_sources(tmp_path):
    seed = [
        ListenEvent(
            track=TrackRef(mbid="mbid-lastfm", name="Scrobble", artist="B"),
            played_at=datetime(2021, 2, 2, tzinfo=UTC),
            source="lastfm",
        ),
    ]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in seed), encoding="utf-8"
    )
    export = _write_export(
        tmp_path / "watch-history.json",
        [_music_row("2022-10-21T20:19:07.000Z", "Maniac", "trackNEW", "Flower Face")],
    )

    rc = main(["import-youtube", "--from", str(export), "--data-dir", str(tmp_path)])
    assert rc == 0

    history = UserStore(root=tmp_path).load_history()
    sources = sorted(e.source for e in history)
    assert sources == ["lastfm", "youtube_music"]
    assert any(e.track.youtube_id == "trackNEW" for e in history)


def test_cli_import_youtube_supersedes_prior_run(tmp_path, capsys):
    """A self-referential supersede (#164): no prior source overlaps YouTube Music
    history, so re-import only ever displaces a prior run of this importer."""
    export1 = _write_export(
        tmp_path / "watch-history-1.json",
        [_music_row("2022-10-21T20:19:07.000Z", "Old", "trackOLD", "Artist")],
    )
    main(["import-youtube", "--from", str(export1), "--data-dir", str(tmp_path)])

    export2 = _write_export(
        tmp_path / "watch-history-2.json",
        [_music_row("2022-10-21T20:19:07.000Z", "Old", "trackOLD", "Artist")],
    )
    rc = main(["import-youtube", "--from", str(export2), "--data-dir", str(tmp_path)])
    assert rc == 0

    history = UserStore(root=tmp_path).load_history()
    assert len(history) == 1  # superseded, not duplicated
    assert "superseded 1" in capsys.readouterr().out


def test_cli_import_youtube_is_idempotent(tmp_path):
    export = _write_export(
        tmp_path / "watch-history.json",
        [_music_row("2022-10-21T20:19:07.000Z", "Maniac", "trackNEW", "Flower Face")],
    )
    main(["import-youtube", "--from", str(export), "--data-dir", str(tmp_path)])
    first = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    main(["import-youtube", "--from", str(export), "--data-dir", str(tmp_path)])
    second = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    assert first == second
