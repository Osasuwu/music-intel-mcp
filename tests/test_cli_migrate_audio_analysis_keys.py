"""CLI `migrate-audio-analysis-keys` entrypoint (#158 AC4) -- the one-shot
bare-key-to-prefixed-key rename, wired into the argparse surface. Core rename
logic (format classification, conflict handling, idempotency) is covered in
``test_store.py``; this file exercises the CLI wiring and report printing.
"""

from __future__ import annotations

import json

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore


def test_migrate_audio_analysis_keys_renames_and_reports(tmp_path, capsys):
    store = UserStore(root=tmp_path)
    bare = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})

    rc = main(["migrate-audio-analysis-keys", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "migrated 1" in out
    assert "conflicts 0" in out
    new_path = store.audio_analysis_path(f"mbid:{bare}")
    assert new_path.exists()
    assert json.loads(new_path.read_text(encoding="utf-8"))["track_id"] == f"mbid:{bare}"


def test_migrate_audio_analysis_keys_reports_conflicts_in_output(tmp_path, capsys):
    store = UserStore(root=tmp_path)
    bare = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})
    store.write_audio_analysis(track_id=f"mbid:{bare}", embedding=[0.9], tags={})

    rc = main(["migrate-audio-analysis-keys", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "migrated 0" in out
    assert "conflicts 1" in out
    assert bare in out
