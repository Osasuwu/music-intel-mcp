"""CLI `analyze` entrypoint."""

from __future__ import annotations

import shutil

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore


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
