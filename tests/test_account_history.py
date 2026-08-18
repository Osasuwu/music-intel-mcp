"""Spotify Account Data StreamingHistory importer (#103 — extend history past
an existing spotify_extended tail without a timezone conversion or a live
export in the repo).

Pins: direct-UTC ``endTime`` parse (no zone-map — corrected via real-data
verification during the #103 grill, see account_history.py docstring), the
name/artist -> spotify_id lookup built from existing spotify_extended history
(with ambiguous keys dropped rather than guessed), the time-cutoff dedup
against the newest existing spotify_extended play, drop-accounting, and the
``import-account-history`` CLI. Fixtures are built in-test as JSON arrays so
no export is committed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.account_history import (
    AccountHistoryStats,
    build_spotify_id_lookup,
    load_account_history_dir,
    load_account_history_file,
    max_spotify_extended_played_at,
    parse_account_history_timestamp,
)
from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore


def _row(end_time: str, track: str, artist: str, *, ms_played: int = 200_000) -> dict:
    return {"endTime": end_time, "artistName": artist, "trackName": track, "msPlayed": ms_played}


def _write_export(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _esh_event(name: str, artist: str, spotify_id: str, played_at: datetime) -> ListenEvent:
    return ListenEvent(
        track=TrackRef(spotify_id=spotify_id, name=name, artist=artist),
        played_at=played_at,
        source="spotify_extended",
    )


# --------------------------------------------------------------------------- #
# timestamp parsing — direct UTC, no offset
# --------------------------------------------------------------------------- #


def test_parse_account_history_timestamp_is_direct_utc():
    assert parse_account_history_timestamp("2026-07-03 09:30") == datetime(
        2026, 7, 3, 9, 30, tzinfo=UTC
    )


# --------------------------------------------------------------------------- #
# row -> ListenEvent mapping
# --------------------------------------------------------------------------- #


def test_row_maps_with_minimal_play_context(tmp_path):
    path = _write_export(
        tmp_path / "StreamingHistory_music_0.json",
        [_row("2026-07-03 09:30", "Stop Thinking (Pt. 3)", "Reflection", ms_played=568_000)],
    )
    events = load_account_history_file(path)
    assert len(events) == 1
    e = events[0]
    assert e.track.name == "Stop Thinking (Pt. 3)"
    assert e.track.artist == "Reflection"
    assert e.track.spotify_id is None
    assert e.played_at == datetime(2026, 7, 3, 9, 30, tzinfo=UTC)
    assert e.source == "spotify_account_history"
    assert e.context == PlayContext(ms_played=568_000)
    assert canonical_track_id(e.track) == "name:stop thinking (pt. 3)\x1freflection"


def test_no_identity_row_is_dropped_and_counted(tmp_path):
    rows = [
        _row("2026-07-01 10:00", "Kept", "Artist"),
        {"endTime": "2026-07-01 11:00", "artistName": "", "trackName": "", "msPlayed": 1000},
    ]
    path = _write_export(tmp_path / "StreamingHistory_music_0.json", rows)
    stats = AccountHistoryStats()
    events = load_account_history_file(path, stats=stats)
    assert len(events) == 1
    assert stats.skipped_no_identity == 1


def test_unparseable_timestamp_is_skipped_and_counted(tmp_path):
    rows = [
        _row("2026-07-01 10:00", "Good", "Artist"),
        _row("not-a-timestamp", "Bad", "Artist"),
    ]
    path = _write_export(tmp_path / "StreamingHistory_music_0.json", rows)
    stats = AccountHistoryStats()
    events = load_account_history_file(path, stats=stats)
    assert len(events) == 1
    assert events[0].track.name == "Good"
    assert stats.skipped_unparseable == 1
    assert stats.unparseable_samples == ["not-a-timestamp"]


# --------------------------------------------------------------------------- #
# spotify_id lookup from existing spotify_extended history
# --------------------------------------------------------------------------- #


def test_lookup_resolves_matching_name_artist_case_insensitively():
    existing = [_esh_event("Maniac", "Flower Face", "trackID", datetime(2026, 1, 1, tzinfo=UTC))]
    lookup, ambiguous = build_spotify_id_lookup(existing)
    assert lookup[("maniac", "flower face")] == "trackID"
    assert ambiguous == 0


def test_lookup_drops_ambiguous_key_mapping_to_two_different_ids():
    existing = [
        _esh_event("Same Name", "Same Artist", "idOne", datetime(2026, 1, 1, tzinfo=UTC)),
        _esh_event("Same Name", "Same Artist", "idTwo", datetime(2026, 1, 2, tzinfo=UTC)),
    ]
    lookup, ambiguous = build_spotify_id_lookup(existing)
    assert ("same name", "same artist") not in lookup
    assert ambiguous == 1


def test_row_resolves_to_spotify_id_via_lookup(tmp_path):
    existing = [_esh_event("Maniac", "Flower Face", "trackID", datetime(2026, 1, 1, tzinfo=UTC))]
    _write_export(
        tmp_path / "StreamingHistory_music_0.json",
        [_row("2026-07-03 09:30", "Maniac", "Flower Face")],
    )
    stats = AccountHistoryStats()
    events = load_account_history_dir(tmp_path, existing_events=existing, stats=stats)
    assert len(events) == 1
    assert events[0].track.spotify_id == "trackID"
    assert canonical_track_id(events[0].track) == "spotify:trackID"
    assert stats.resolved_via_spotify_lookup == 1


# --------------------------------------------------------------------------- #
# time-cutoff dedup against the newest existing spotify_extended play
# --------------------------------------------------------------------------- #


def test_max_spotify_extended_played_at_ignores_other_sources():
    events = [
        _esh_event("A", "B", "id1", datetime(2026, 1, 1, tzinfo=UTC)),
        _esh_event("C", "D", "id2", datetime(2026, 6, 1, tzinfo=UTC)),
        ListenEvent(
            track=TrackRef(name="E", artist="F"),
            played_at=datetime(2026, 12, 1, tzinfo=UTC),
            source="spotify_account_history",
        ),
    ]
    assert max_spotify_extended_played_at(events) == datetime(2026, 6, 1, tzinfo=UTC)


def test_rows_at_or_before_cutoff_are_dropped_and_counted(tmp_path):
    existing = [_esh_event("Old", "Artist", "oldID", datetime(2026, 7, 1, 12, 0, tzinfo=UTC))]
    rows = [
        _row("2026-07-01 11:00", "Before Cutoff", "Artist"),  # dropped
        _row("2026-07-01 12:00", "At Cutoff", "Artist"),  # dropped (<=, not <)
        _row("2026-07-01 13:00", "After Cutoff", "Artist"),  # kept
    ]
    _write_export(tmp_path / "StreamingHistory_music_0.json", rows)
    stats = AccountHistoryStats()
    events = load_account_history_dir(tmp_path, existing_events=existing, stats=stats)
    assert len(events) == 1
    assert events[0].track.name == "After Cutoff"
    assert stats.dropped_before_cutoff == 2


def test_no_existing_spotify_extended_history_imports_everything(tmp_path):
    rows = [_row("2020-01-01 00:00", "Anything", "Artist")]
    _write_export(tmp_path / "StreamingHistory_music_0.json", rows)
    events = load_account_history_dir(tmp_path, existing_events=[])
    assert len(events) == 1


# --------------------------------------------------------------------------- #
# directory load — concat + dedup + idempotency
# --------------------------------------------------------------------------- #


def test_dir_load_is_idempotent(tmp_path):
    row = _row("2026-07-03 09:30", "Once", "Artist")
    _write_export(tmp_path / "StreamingHistory_music_0.json", [row, row])
    once = load_account_history_dir(tmp_path, existing_events=[])
    twice = load_account_history_dir(tmp_path, existing_events=[])
    assert len(once) == 1
    assert once == twice


# --------------------------------------------------------------------------- #
# CLI import-account-history
# --------------------------------------------------------------------------- #


def test_cli_import_account_history_extends_past_esh_tail(tmp_path, capsys):
    seed = [_esh_event("Old", "Artist", "oldID", datetime(2026, 7, 1, 12, 0, tzinfo=UTC))]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in seed), encoding="utf-8"
    )

    export = tmp_path / "export"
    export.mkdir()
    _write_export(
        export / "StreamingHistory_music_0.json",
        [
            _row("2026-07-01 11:00", "Before Cutoff", "Artist"),
            _row("2026-07-05 09:30", "New Play", "Artist"),
        ],
    )

    rc = main(["import-account-history", "--from", str(export), "--data-dir", str(tmp_path)])
    assert rc == 0

    history = UserStore(root=tmp_path).load_history()
    sources = sorted({e.source for e in history})
    assert sources == ["spotify_account_history", "spotify_extended"]
    assert any(e.track.name == "New Play" for e in history)
    assert not any(e.track.name == "Before Cutoff" for e in history)
    assert "dropped 1 rows" in capsys.readouterr().out


def test_cli_import_account_history_is_idempotent(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    _write_export(
        export / "StreamingHistory_music_0.json",
        [_row("2026-07-05 09:30", "New Play", "Artist")],
    )
    main(["import-account-history", "--from", str(export), "--data-dir", str(tmp_path)])
    first = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    main(["import-account-history", "--from", str(export), "--data-dir", str(tmp_path)])
    second = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    assert first == second
