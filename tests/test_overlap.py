"""Cross-user overlap measurement (#266): export, compare, CLI. Synthetic data only."""

from __future__ import annotations

import json
from pathlib import Path

from music_intel_mcp import overlap
from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef


def _plays(name, artist, n, *, spotify_id=None, ms=180_000):
    return [
        ListenEvent(
            track=TrackRef(name=name, artist=artist, spotify_id=spotify_id),
            played_at=f"2026-01-{i + 1:02d}T00:00:00Z",
            source="test",
            context=PlayContext(ms_played=ms),
        )
        for i in range(n)
    ]


def _write_history(root: Path, events) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8"
    )


def test_only_tracks_with_three_valid_plays_are_exported():
    events = (
        _plays("In", "A", 3, spotify_id="s1")
        + _plays("TwoPlays", "A", 2, spotify_id="s2")
        + _plays("ShortPlays", "A", 3, spotify_id="s3", ms=5_000)
    )
    cols = overlap.hash_columns(events, salt="x")
    assert len(cols[overlap.COLUMN_SPOTIFY_ID].hashes) == 1
    assert len(cols[overlap.COLUMN_NAME_ARTIST].hashes) == 1


def test_known_overlap_is_computed_per_column_with_min_denominator():
    a = _plays("Shared", "X", 3, spotify_id="s1") + _plays("OnlyA", "X", 3, spotify_id="s2")
    b = (
        _plays("Shared", "X", 3, spotify_id="s1")
        + _plays("OnlyB", "Y", 4, spotify_id="s3")
        + _plays("OnlyB2", "Y", 4, spotify_id="s4")
    )
    res = {
        r.column: r
        for r in overlap.compare(
            overlap.hash_columns(a, salt="k"), overlap.hash_columns(b, salt="k")
        )
    }
    r = res[overlap.COLUMN_SPOTIFY_ID]
    assert (r.size_a, r.size_b, r.intersection, r.denominator) == (2, 3, 1, 2)
    assert r.coefficient == 0.5


def test_columns_differ_when_same_song_has_different_spotify_ids():
    a = _plays("Song", "Artist", 3, spotify_id="idA")
    b = _plays("Song", "Artist", 3, spotify_id="idB")
    res = {
        r.column: r
        for r in overlap.compare(
            overlap.hash_columns(a, salt="k"), overlap.hash_columns(b, salt="k")
        )
    }
    assert res[overlap.COLUMN_SPOTIFY_ID].intersection == 0
    assert res[overlap.COLUMN_NAME_ARTIST].intersection == 1


def test_name_column_normalises_case_and_feat_credit():
    a = _plays("Song (feat. Guest)", "ARTIST", 3, spotify_id="i1")
    b = _plays("song", "artist", 3, spotify_id="i2")
    res = overlap.compare(overlap.hash_columns(a, salt="k"), overlap.hash_columns(b, salt="k"))
    assert res[1].intersection == 1


def test_tracks_without_a_value_are_counted_not_silently_dropped():
    events = _plays("NoId", "A", 3) + _plays("HasId", "A", 3, spotify_id="s1")
    cols = overlap.hash_columns(events, salt="k")
    assert cols[overlap.COLUMN_SPOTIFY_ID].not_comparable == 1
    assert len(cols[overlap.COLUMN_SPOTIFY_ID].hashes) == 1
    assert cols[overlap.COLUMN_NAME_ARTIST].not_comparable == 0


def test_different_salt_breaks_the_match():
    events = _plays("S", "A", 3, spotify_id="s1")
    res = overlap.compare(
        overlap.hash_columns(events, salt="one"), overlap.hash_columns(events, salt="two")
    )
    assert res[0].intersection == 0


def test_empty_side_gives_no_coefficient():
    res = overlap.compare(overlap.hash_columns([], salt="k"), overlap.hash_columns([], salt="k"))
    assert res[0].coefficient is None
    assert "n/a" in overlap.format_report(res)


def test_export_contains_no_plaintext_and_roundtrips(tmp_path):
    events = _plays("Secret Title", "Secret Artist", 3, spotify_id="spotifyid123")
    out = tmp_path / "e.json"
    overlap.write_export(out, events, salt="s", min_valid_plays=3)
    text = out.read_text(encoding="utf-8")
    for leaked in ("Secret", "secret", "spotifyid123", "2026-01"):
        assert leaked not in text
    salt, min_plays, cols = overlap.read_export(out)
    assert (salt, min_plays) == ("s", 3)
    assert cols == overlap.hash_columns(events, salt="s")


def test_cli_end_to_end_prints_counts_and_identities(tmp_path, capsys):
    op, friend = tmp_path / "op", tmp_path / "friend"
    _write_history(op, _plays("Shared", "X", 3, spotify_id="s1") + _plays("Own", "X", 3))
    _write_history(friend, _plays("Shared", "X", 5, spotify_id="s1"))
    f = tmp_path / "export.json"

    assert main(["overlap-export", "--data-dir", str(op), "--out", str(f)]) == 0
    capsys.readouterr()
    assert main(["overlap-compare", "--data-dir", str(friend), "--file", str(f)]) == 0
    out = capsys.readouterr().out
    assert "Spotify track id" in out
    assert "normalised track name + album artist" in out
    assert "overlap coefficient: 1.0000" in out
    assert "intersection: 1  denominator min(|A|,|B|): 1" in out
    assert "not comparable on this column: A 1, B 0" in out


def test_cli_rejects_a_file_that_is_not_an_export(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"x": 1}), encoding="utf-8")
    _write_history(tmp_path / "d", _plays("S", "A", 3, spotify_id="s"))
    assert main(["overlap-compare", "--data-dir", str(tmp_path / "d"), "--file", str(bad)]) == 2
