"""CLI surface for the #201 stream-decode pre-pilot window-bias gate.

Mirrors ``test_cli_window_probe_report.py``'s three tests, but against the
stream-decode leg's own journal/command so the two legs' numbers are never
read back through the same command (#201 AC6).
"""

from __future__ import annotations

from music_intel_mcp.cli import main
from music_intel_mcp.store import UserStore
from music_intel_mcp.window_probe import (
    WindowPair,
    append_window_pair,
    stream_decode_window_probe_path,
)


def _write_pairs(tmp_path, n: int) -> None:
    path = stream_decode_window_probe_path(UserStore(root=tmp_path))
    for i in range(n):
        append_window_pair(
            path,
            WindowPair(
                track_id=f"youtube:{i}",
                long_window_s=200.0,
                short_window_s=30.0,
                long_embedding=[1.0, 0.0],
                short_embedding=[1.0, 0.0],
            ),
        )


def test_stream_decode_window_probe_report_prints_the_gate_result(tmp_path, capsys):
    _write_pairs(tmp_path, 100)

    rc = main(["stream-decode-window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tracks: 100" in out
    assert "whole-track leg" in out
    assert "#201" in out
    assert "#169" not in out
    assert "under-powered" not in out


def test_stream_decode_window_probe_report_flags_an_under_powered_sample(tmp_path, capsys):
    _write_pairs(tmp_path, 5)

    rc = main(["stream-decode-window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert "under-powered" in capsys.readouterr().out.lower()


def test_stream_decode_window_probe_report_on_an_empty_journal_says_nothing_was_measured(
    tmp_path, capsys
):
    rc = main(["stream-decode-window-probe-report", "--data-dir", str(tmp_path)])

    assert rc == 0
    assert "no window-probe pairs recorded yet" in capsys.readouterr().out
