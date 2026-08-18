"""Spotify Extended Streaming History importer (#89 — the V1 data foundation).

These tests pin the JSON row -> ``ListenEvent`` mapping, the ``ts`` parse, the
lossless projection (non-audio rows dropped + counted; nothing else filtered),
second-resolution dedup/idempotency, the ``import-spotify`` CLI with source-scoped
supersede, and ``PlayContext`` schema back-compat. Fixtures are built in-test as
JSON arrays so no export is committed and the exact source shape is visible here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.spotify_extended import (
    SpotifyExtendedStats,
    load_spotify_extended_dir,
    load_spotify_extended_file,
    parse_spotify_timestamp,
)
from music_intel_mcp.store import UserStore


def _audio_row(
    ts: str,
    track: str,
    artist: str,
    track_id: str,
    *,
    ms_played: int = 200_000,
    skipped: bool = False,
    incognito: bool = False,
    conn_country: str = "KZ",
) -> dict:
    """One audio-track export row (only the fields the adapter reads + a couple of
    ignored ones, to prove ``extra`` on the raw JSON is tolerated)."""
    return {
        "ts": ts,
        "ms_played": ms_played,
        "master_metadata_track_name": track,
        "master_metadata_album_artist_name": artist,
        "master_metadata_album_album_name": "Some Album",
        "spotify_track_uri": f"spotify:track:{track_id}",
        "spotify_episode_uri": None,
        "audiobook_uri": None,
        "skipped": skipped,
        "incognito_mode": incognito,
        "conn_country": conn_country,
        "reason_start": "trackdone",
        "reason_end": "trackdone",
        "shuffle": True,
        "offline": False,
    }


def _write_export(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# timestamp parsing
# --------------------------------------------------------------------------- #


def test_parse_zulu_timestamp_to_utc():
    assert parse_spotify_timestamp("2022-10-21T20:19:07Z") == datetime(
        2022, 10, 21, 20, 19, 7, tzinfo=UTC
    )


def test_parse_numeric_offset_coerced_to_utc():
    # a +05:00 wall-clock is the same instant as 07:19:07Z.
    assert parse_spotify_timestamp("2022-10-21T12:19:07+05:00") == datetime(
        2022, 10, 21, 7, 19, 7, tzinfo=UTC
    )


# --------------------------------------------------------------------------- #
# row -> ListenEvent mapping + lossless PlayContext
# --------------------------------------------------------------------------- #


def test_audio_row_maps_with_full_play_context(tmp_path):
    path = _write_export(
        tmp_path / "Streaming_History_Audio_2022.json",
        [_audio_row("2022-10-21T20:19:07Z", "Maniac", "Flower Face", "56O4WlGoUJiwfwoyRYqTe9")],
    )
    events = load_spotify_extended_file(path)

    assert len(events) == 1
    e = events[0]
    assert e.track.spotify_id == "56O4WlGoUJiwfwoyRYqTe9"
    assert e.track.name == "Maniac"
    assert e.track.artist == "Flower Face"
    assert e.played_at == datetime(2022, 10, 21, 20, 19, 7, tzinfo=UTC)
    assert e.source == "spotify_extended"
    assert e.context == PlayContext(
        ms_played=200_000, skipped=False, incognito=False, conn_country="KZ"
    )
    assert canonical_track_id(e.track) == "spotify:56O4WlGoUJiwfwoyRYqTe9"


def test_episode_and_audiobook_rows_are_dropped_and_counted(tmp_path):
    rows = [
        _audio_row("2023-01-01T10:00:00Z", "Real Track", "Real Artist", "trackAAA"),
        {  # podcast/video episode
            "ts": "2023-01-01T11:00:00Z",
            "ms_played": 500_000,
            "master_metadata_track_name": None,
            "spotify_track_uri": None,
            "spotify_episode_uri": "spotify:episode:podcastXYZ",
        },
        {  # audiobook chapter
            "ts": "2023-01-01T12:00:00Z",
            "ms_played": 800_000,
            "master_metadata_track_name": None,
            "spotify_track_uri": None,
            "audiobook_uri": "spotify:show:bookABC",
        },
    ]
    path = _write_export(tmp_path / "Streaming_History_Audio_2023.json", rows)
    stats = SpotifyExtendedStats()
    events = load_spotify_extended_file(path, stats=stats)

    assert len(events) == 1
    assert events[0].track.name == "Real Track"
    assert stats.skipped_episode == 1
    assert stats.skipped_audiobook == 1
    assert stats.total_skipped == 2


def test_no_identity_row_is_dropped_and_counted(tmp_path):
    rows = [
        _audio_row("2023-02-01T10:00:00Z", "Kept", "Artist", "trackKEEP"),
        {  # neither a track uri nor a track name -> no identity
            "ts": "2023-02-01T11:00:00Z",
            "ms_played": 1_000,
            "master_metadata_track_name": None,
            "spotify_track_uri": None,
        },
    ]
    path = _write_export(tmp_path / "Streaming_History_Audio_2023.json", rows)
    stats = SpotifyExtendedStats()
    events = load_spotify_extended_file(path, stats=stats)
    assert len(events) == 1
    assert stats.skipped_no_identity == 1


def test_row_without_track_uri_keeps_name_artist_fallback(tmp_path):
    row = {
        "ts": "2023-03-01T09:15:00Z",
        "ms_played": 60_000,
        "master_metadata_track_name": "No Uri Track",
        "master_metadata_album_artist_name": "Some Artist",
        "spotify_track_uri": None,
    }
    path = _write_export(tmp_path / "Streaming_History_Audio_2023.json", [row])
    events = load_spotify_extended_file(path)
    assert len(events) == 1
    assert events[0].track.spotify_id is None
    assert canonical_track_id(events[0].track) == "name:no uri track\x1fsome artist"


def test_unparseable_timestamp_is_skipped_and_counted(tmp_path):
    rows = [
        _audio_row("2023-04-01T10:00:00Z", "Good", "Artist", "trackGOOD"),
        _audio_row("not-a-timestamp", "Bad", "Artist", "trackBAD"),
    ]
    path = _write_export(tmp_path / "Streaming_History_Audio_2023.json", rows)
    stats = SpotifyExtendedStats()
    events = load_spotify_extended_file(path, stats=stats)
    assert len(events) == 1
    assert events[0].track.name == "Good"
    assert stats.skipped_unparseable == 1
    assert stats.unparseable_samples == ["not-a-timestamp"]


def test_lossless_no_validity_filtering_at_import(tmp_path):
    """A <5s play, a skipped play, and an incognito play are all KEPT — validity is
    an analysis-time policy (#90), never an import-time filter (decision 83bd6f76)."""
    rows = [
        _audio_row("2023-05-01T10:00:00Z", "Blip", "A", "t1", ms_played=1_500),
        _audio_row("2023-05-01T11:00:00Z", "Skip", "A", "t2", skipped=True),
        _audio_row("2023-05-01T12:00:00Z", "Private", "A", "t3", incognito=True),
    ]
    path = _write_export(tmp_path / "Streaming_History_Audio_2023.json", rows)
    events = load_spotify_extended_file(path)
    assert len(events) == 3
    assert events[0].context.ms_played == 1_500
    assert events[1].context.skipped is True
    assert events[2].context.incognito is True


# --------------------------------------------------------------------------- #
# directory load — concat + second-resolution dedup + idempotency
# --------------------------------------------------------------------------- #


def test_dir_concats_dedups_at_second_resolution_and_sorts(tmp_path):
    same = _audio_row("2023-06-01T15:30:45Z", "Dup", "Artist", "trackDUP")
    later = _audio_row("2024-06-01T15:30:45Z", "Later", "Artist", "trackLATE")
    _write_export(tmp_path / "Streaming_History_Audio_2023.json", [later, same])
    _write_export(tmp_path / "Streaming_History_Audio_2024.json", [same])  # same second

    events = load_spotify_extended_dir(tmp_path)
    assert len(events) == 2  # the two identical (track, second) rows collapse to one
    assert [e.played_at for e in events] == sorted(e.played_at for e in events)
    assert events[0].track.name == "Dup"


def test_dir_load_is_idempotent(tmp_path):
    row = _audio_row("2023-07-01T00:00:00Z", "Once", "Artist", "trackONCE")
    _write_export(tmp_path / "Streaming_History_Audio_2023.json", [row, row])
    once = load_spotify_extended_dir(tmp_path)
    twice = load_spotify_extended_dir(tmp_path)
    assert len(once) == 1
    assert once == twice


# --------------------------------------------------------------------------- #
# CLI import-spotify — source-scoped supersede
# --------------------------------------------------------------------------- #


def test_cli_import_spotify_supersedes_ifttt_preserves_others(tmp_path, capsys):
    """import-spotify replaces the thin IFTTT rows (same plays, richer source) but
    leaves a Last.fm event untouched (source-scoped supersede, decision 23fcf92c)."""
    seed = [
        ListenEvent(
            track=TrackRef(spotify_id="oldIFTTT", name="Old", artist="A"),
            played_at=datetime(2021, 1, 1, tzinfo=UTC),
            source="ifttt",
        ),
        ListenEvent(
            track=TrackRef(mbid="mbid-lastfm", name="Scrobble", artist="B"),
            played_at=datetime(2021, 2, 2, tzinfo=UTC),
            source="lastfm",
        ),
    ]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in seed), encoding="utf-8"
    )

    export = tmp_path / "export"
    export.mkdir()
    _write_export(
        export / "Streaming_History_Audio_2022.json",
        [_audio_row("2022-10-21T20:19:07Z", "Maniac", "Flower Face", "trackNEW")],
    )

    rc = main(["import-spotify", "--from", str(export), "--data-dir", str(tmp_path)])
    assert rc == 0

    history = UserStore(root=tmp_path).load_history()
    sources = sorted(e.source for e in history)
    assert sources == ["lastfm", "spotify_extended"]  # ifttt superseded, lastfm kept
    assert any(e.track.spotify_id == "trackNEW" for e in history)
    assert any(e.track.mbid == "mbid-lastfm" for e in history)
    assert "superseded 1" in capsys.readouterr().out


def test_cli_import_spotify_is_idempotent(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    _write_export(
        export / "Streaming_History_Audio_2022.json",
        [_audio_row("2022-10-21T20:19:07Z", "Maniac", "Flower Face", "trackNEW")],
    )
    main(["import-spotify", "--from", str(export), "--data-dir", str(tmp_path)])
    first = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    main(["import-spotify", "--from", str(export), "--data-dir", str(tmp_path)])
    second = (tmp_path / "history.jsonl").read_text(encoding="utf-8")
    assert first == second


# --------------------------------------------------------------------------- #
# PlayContext schema back-compat
# --------------------------------------------------------------------------- #


def test_play_context_back_compat_old_history_line_validates():
    """A history.jsonl line written before #89 (no incognito/conn_country keys)
    still validates; the new fields default to None."""
    old_line = (
        '{"track":{"name":"X","artist":"Y"},"played_at":"2024-01-01T00:00:00Z",'
        '"source":"ifttt","context":{"ms_played":123000,"skipped":false}}'
    )
    e = ListenEvent.model_validate_json(old_line)
    assert e.context.ms_played == 123_000
    assert e.context.incognito is None
    assert e.context.conn_country is None
