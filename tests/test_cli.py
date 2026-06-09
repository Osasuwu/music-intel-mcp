"""CLI `analyze` + `resolve` entrypoints."""

from __future__ import annotations

import shutil
from pathlib import Path

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore

FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index.tsv"


def test_cli_analyze_writes_snapshot(tmp_path, history_sample_path, capsys):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "snapshot:" in out
    assert "events=5" in out
    assert "unique_tracks=3" in out
    assert "roots=0" in out  # honest-empty

    # a valid snapshot landed in the store and re-validates
    profile = UserStore(root=tmp_path).latest_profile()
    assert profile is not None
    assert profile.user_id == "petr"
    assert profile.roots == []


def test_cli_analyze_empty_history(tmp_path, capsys):
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "events=0" in out
    assert "roots=0" in out


def test_cli_resolve_reports_coverage(tmp_path, history_sample_path, capsys):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["resolve", "--data-dir", str(tmp_path), "--mb-index", str(FIXTURE_INDEX)])
    assert rc == 0

    out = capsys.readouterr().out
    # 3 unique tracks: spotify AAA (no source -> spotify), isrc USABC (-> mbid
    # via fixture), mbid 1111 (passthrough). 2 reach MBID.
    assert "resolved 2/3 unique tracks to MBID" in out
    assert "unresolved (flagged, not dropped): 1" in out

    # identities are cached for re-runs
    assert (tmp_path / "identity").is_dir()
