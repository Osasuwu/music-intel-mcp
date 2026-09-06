"""CLI `replay-queue` entrypoint (#163 AC4) — prints the AC5.1 coverage
statistic (share of valid plays covered by already-analysed + newly-queued
tracks). Pure selection/coverage logic is covered in ``test_replay_queue.py``;
this file exercises the CLI wiring only: reading local history via
``UserStore``, calling ``replay_queue_coverage``, and printing the stat.
"""

from __future__ import annotations

from pathlib import Path

from music_intel_mcp.cli import main
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef


def _write_history(data_dir: Path, events: list[ListenEvent]) -> None:
    history_path = data_dir / "history.jsonl"
    history_path.write_text(
        "\n".join(e.model_dump_json() for e in events) + ("\n" if events else ""),
        encoding="utf-8",
    )


def test_replay_queue_prints_coverage_stat_for_a_fully_covered_track(tmp_path, capsys):
    track = TrackRef(name="Real Listen", artist="Artist", spotify_id="s1")
    events = [
        ListenEvent(
            track=track,
            played_at=f"2026-01-0{i}T00:00:00Z",
            source="test",
            context=PlayContext(ms_played=180_000),
        )
        for i in range(1, 4)
    ]
    _write_history(tmp_path, events)

    rc = main(["replay-queue", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 tracks queued" in out
    assert "1 eligible" in out
    assert "valid-play coverage: 100.0%" in out


def test_replay_queue_handles_empty_history(tmp_path, capsys):
    rc = main(["replay-queue", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "0 tracks queued" in out
    assert "valid-play coverage: 0.0%" in out
