"""CLI `replay-journal-summary` entrypoint (#166 AC5) -- the weekly-checkpoint
per-outcome count printout, wired into the argparse surface. Journal read/count
logic is covered in ``test_replay_capture.py``; this file exercises the CLI
wiring and report printing.
"""

from __future__ import annotations

from music_intel_mcp.cli import main
from music_intel_mcp.replay_capture import (
    ReplayJournalEntry,
    append_replay_journal_entry,
    replay_journal_path,
)
from music_intel_mcp.store import UserStore


def test_replay_journal_summary_prints_per_outcome_counts(tmp_path, capsys):
    store = UserStore(root=tmp_path)
    path = replay_journal_path(store)
    for outcome in ("ok", "silent", "ok"):
        append_replay_journal_entry(
            path,
            ReplayJournalEntry(
                track_id="mbid:abc",
                outcome=outcome,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:02:00+00:00",
            ),
        )

    rc = main(["replay-journal-summary", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "3 attempts" in out
    assert "ok: 2" in out
    assert "silent: 1" in out


def test_replay_journal_summary_excludes_requeue_bookkeeping_from_attempt_count(tmp_path, capsys):
    """A requeue writes a "requeued" bookkeeping entry on top of the original
    silent/short outcome entry for the *same* physical capture attempt --
    counting it as an additional attempt overstates the weekly-checkpoint
    total (issue #166 AC5 review finding)."""
    store = UserStore(root=tmp_path)
    path = replay_journal_path(store)
    for outcome in ("silent", "requeued", "ok"):
        append_replay_journal_entry(
            path,
            ReplayJournalEntry(
                track_id="mbid:abc",
                outcome=outcome,
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:02:00+00:00",
            ),
        )

    rc = main(["replay-journal-summary", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "2 attempts" in out
    assert "ok: 1" in out
    assert "silent: 1" in out
    assert "requeued: 1" in out


def test_replay_journal_summary_handles_missing_journal(tmp_path, capsys):
    rc = main(["replay-journal-summary", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no attempts recorded" in out
