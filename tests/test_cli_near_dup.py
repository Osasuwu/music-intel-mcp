"""CLI `near-dup scan`/`near-dup apply` entrypoints (#140 AC2/AC4) -- wiring
the offline embedding-space near-duplicate batch-merge tool into the
argparse surface. Core scan/apply logic (tier assignment, hub guard, winner
ranking) is covered in ``test_near_dup_scan.py``/``test_near_dup_apply.py``;
this file exercises the CLI wiring and report/alias printing only.
"""

from __future__ import annotations

import json

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore


def _write(store: UserStore, track_id: str, embedding: list[float]) -> None:
    store.write_audio_analysis(track_id=track_id, embedding=embedding, tags={})


def test_near_dup_scan_writes_report_and_prints_path(tmp_path, capsys):
    store = UserStore(root=tmp_path)
    _write(store, "spotify:dup", [1.0, 0.0, 0.0, 0.0])
    _write(store, "name:dup|artist", [1.0, 0.0, 0.0, 0.0])

    rc = main(["near-dup", "scan", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    reports = sorted((store.root / "near_dup").glob("report-*.json"))
    assert len(reports) == 1
    assert str(reports[0]) in out


def test_near_dup_apply_writes_aliases_and_prints_count(tmp_path, capsys):
    store = UserStore(root=tmp_path)
    _write(store, "spotify:dup", [1.0, 0.0, 0.0, 0.0])
    _write(store, "name:dup|artist", [1.0, 0.0, 0.0, 0.0])
    dup_fp = list(range(300))
    store.write_fingerprint(track_id="spotify:dup", fingerprint=dup_fp, duration_s=30.0)
    store.write_fingerprint(track_id="name:dup|artist", fingerprint=dup_fp, duration_s=30.0)

    from music_intel_mcp.near_dup import scan

    json_path = scan(store)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    for pair in report["pairs"]:
        pair["accepted"] = True
    json_path.write_text(json.dumps(report), encoding="utf-8")

    rc = main(["near-dup", "apply", str(json_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "wrote 1 alias" in out
    aliases_path = store.root / "aliases.jsonl"
    assert aliases_path.exists()


def test_near_dup_apply_with_no_accepted_pairs_prints_zero(tmp_path, capsys):
    """A freshly generated report has `accepted: null` on every pair -- e.g.
    `scan` then `apply` before any manual review. `apply()` never creates
    `aliases.jsonl` when there is nothing to write, so the CLI must not
    assume the path exists."""
    store = UserStore(root=tmp_path)
    _write(store, "spotify:dup", [1.0, 0.0, 0.0, 0.0])
    _write(store, "name:dup|artist", [1.0, 0.0, 0.0, 0.0])
    dup_fp = list(range(300))
    store.write_fingerprint(track_id="spotify:dup", fingerprint=dup_fp, duration_s=30.0)
    store.write_fingerprint(track_id="name:dup|artist", fingerprint=dup_fp, duration_s=30.0)

    from music_intel_mcp.near_dup import scan

    json_path = scan(store)

    rc = main(["near-dup", "apply", str(json_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "wrote 0 alias" in out
    assert not (store.root / "aliases.jsonl").exists()
