"""IFTTT xlsx history importer (#76 — enables #66 criterion 1).

The owner's listening history arrives as a directory of IFTTT "Spotify ->
spreadsheet" ``.xlsx`` exports. These tests pin the column mapping, the
``"Month D, YYYY at H:MMAM"`` timestamp parse, blank-row tolerance, and the
idempotent merge+dedup of ``load_ifttt_dir``. Fixtures are built in-test with
openpyxl so no binary workbook is committed and the exact source shape is
visible in the test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from openpyxl import Workbook

from music_intel_mcp.ingest import (
    IngestStats,
    load_ifttt_dir,
    load_ifttt_workbook,
    parse_ifttt_timestamp,
)
from music_intel_mcp.shared_store import canonical_track_id

# (played_at, track, artist, spotify_id, url) — IFTTT row order, no header.
_ROW_A = (
    "December 3, 2024 at 12:31AM",
    "Maniac",
    "Flower Face",
    "56O4WlGoUJiwfwoyRYqTe9",
    "https://open.spotify.com/track/56O4WlGoUJiwfwoyRYqTe9",
)
_ROW_B = (
    "July 14, 2025 at 2:05PM",
    "Cellophane",
    "FKA twigs",
    "1234567890abcdefABCDEF",
    "https://open.spotify.com/track/1234567890abcdefABCDEF",
)


def _write_workbook(path: Path, rows: list[tuple]) -> Path:
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(list(row))
    wb.save(path)
    return path


# --------------------------------------------------------------------------- #
# timestamp parsing — the one non-mechanical bit of the mapping
# --------------------------------------------------------------------------- #


def test_parse_midnight_and_noon_and_pm():
    # 12:31AM is 00:31, not 12:31; 2:05PM is 14:05; 12:00PM is noon.
    assert parse_ifttt_timestamp("December 3, 2024 at 12:31AM") == datetime(
        2024, 12, 3, 0, 31, tzinfo=UTC
    )
    assert parse_ifttt_timestamp("July 14, 2025 at 2:05PM") == datetime(
        2025, 7, 14, 14, 5, tzinfo=UTC
    )
    assert parse_ifttt_timestamp("January 1, 2025 at 12:00PM") == datetime(
        2025, 1, 1, 12, 0, tzinfo=UTC
    )


def test_parse_tolerates_surrounding_whitespace():
    assert parse_ifttt_timestamp("  March 9, 2025 at 6:07AM  ") == datetime(
        2025, 3, 9, 6, 7, tzinfo=UTC
    )


# --------------------------------------------------------------------------- #
# workbook -> ListenEvent column mapping
# --------------------------------------------------------------------------- #


def test_workbook_maps_columns_to_listen_events(tmp_path):
    path = _write_workbook(tmp_path / "Spotify_data.xlsx", [_ROW_A, _ROW_B])
    events = load_ifttt_workbook(path)

    assert len(events) == 2
    a = events[0]
    assert a.track.spotify_id == "56O4WlGoUJiwfwoyRYqTe9"
    assert a.track.name == "Maniac"
    assert a.track.artist == "Flower Face"
    assert a.played_at == datetime(2024, 12, 3, 0, 31, tzinfo=UTC)
    assert a.source == "ifttt"
    # bare id, not URI-prefixed -> canonical id is spotify:<bare id>.
    assert canonical_track_id(a.track) == "spotify:56O4WlGoUJiwfwoyRYqTe9"


def test_blank_and_untimestamped_rows_are_skipped(tmp_path):
    rows = [
        _ROW_A,
        ("", "", "", "", ""),  # trailing blank row
        (None, None, None, None, None),  # openpyxl all-None row
        ("", "Ghost", "Artist", "", ""),  # has data but no timestamp -> unplaceable
        _ROW_B,
    ]
    path = _write_workbook(tmp_path / "Spotify_data1.xlsx", rows)
    stats = IngestStats()
    events = load_ifttt_workbook(path, stats=stats)
    # only the two well-formed rows survive; nothing crashes.
    assert len(events) == 2
    assert {e.track.name for e in events} == {"Maniac", "Cellophane"}
    assert stats.skipped_empty == 2  # the "" row and the all-None row


def test_date_only_timestamp_is_skipped_and_counted(tmp_path):
    """The real export carried one date-only cell ("June 7, 2026") with no
    time-of-day. It is skipped (never coerced to a fabricated clock time) and
    tallied so the CLI can surface it — drift shows as a count, not a crash."""
    rows = [_ROW_A, ("June 7, 2026", "Daylist", "Various", "ZZZ123", "")]
    path = _write_workbook(tmp_path / "Spotify_data3.xlsx", rows)
    stats = IngestStats()
    events = load_ifttt_workbook(path, stats=stats)
    assert len(events) == 1  # only the WITH_TIME row promotes
    assert events[0].track.name == "Maniac"
    assert stats.skipped_unparseable == 1
    assert stats.unparseable_samples == ["June 7, 2026"]


def test_row_without_spotify_id_keeps_name_artist_fallback(tmp_path):
    row = ("April 2, 2025 at 9:15PM", "No Id Track", "Some Artist", "", "")
    path = _write_workbook(tmp_path / "Spotify_data2.xlsx", [row])
    events = load_ifttt_workbook(path)
    assert len(events) == 1
    assert events[0].track.spotify_id is None
    assert canonical_track_id(events[0].track) == "name:no id track\x1fsome artist"


# --------------------------------------------------------------------------- #
# directory load — concat across workbooks + idempotent dedup
# --------------------------------------------------------------------------- #


def test_dir_concats_dedups_and_sorts(tmp_path):
    # _ROW_A appears in both workbooks (same id + timestamp) -> one event.
    _write_workbook(tmp_path / "Spotify_data.xlsx", [_ROW_B, _ROW_A])
    _write_workbook(tmp_path / "Spotify_data1.xlsx", [_ROW_A])

    events = load_ifttt_dir(tmp_path)

    assert len(events) == 2  # deduped
    # sorted ascending by played_at: Dec 2024 before Jul 2025.
    assert [e.played_at for e in events] == sorted(e.played_at for e in events)
    assert events[0].track.name == "Maniac"
    assert events[1].track.name == "Cellophane"


def test_dir_load_is_idempotent_over_a_played_at_and_track(tmp_path):
    _write_workbook(tmp_path / "Spotify_data.xlsx", [_ROW_A, _ROW_A])  # dup within file
    once = load_ifttt_dir(tmp_path)
    twice = load_ifttt_dir(tmp_path)
    assert len(once) == 1
    assert once == twice
