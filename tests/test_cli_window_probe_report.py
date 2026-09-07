"""CLI surface for the #169 pre-pilot window-bias gate.

The probe journal is written passively during replay (see
``test_window_probe.py``); this command is how the owner reads it back and
judges the gate.
"""

from __future__ import annotations

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore
from music_intel_mcp.window_probe import WindowPair, append_window_pair, window_probe_path


def _write_pairs(tmp_path, n: int) -> None:
    path = window_probe_path(UserStore(root=tmp_path))
    for i in range(n):
        append_window_pair(
            path,
            WindowPair(
                track_id=f"mbid:{i}",
                long_window_s=120.0,
                short_window_s=30.0,
                long_embedding=[1.0, 0.0],
                short_embedding=[1.0, 0.0],
            ),
        )


def test_window_probe_report_prints_the_gate_result(tmp_path, capsys):
    _write_pairs(tmp_path, 100)

    rc = main(["window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tracks: 100" in out
    assert "cosine distance" in out
    assert "capture-to-capture" in out
    assert "under-powered" not in out


def test_window_probe_report_flags_an_under_powered_sample(tmp_path, capsys):
    """#169 AC1 wants >=100 tracks. A thin run must not print as a passed gate."""
    _write_pairs(tmp_path, 5)

    rc = main(["window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert "under-powered" in capsys.readouterr().out.lower()


def test_window_probe_report_on_an_empty_journal_says_nothing_was_measured(tmp_path, capsys):
    rc = main(["window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert "no window-probe pairs recorded yet" in capsys.readouterr().out
