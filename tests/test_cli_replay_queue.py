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


# --- #170 AC8: coverage ceiling measured and reported before AC5.1's -- #
# >=80% criterion is applied, stating whether --cap or --min-valid-plays
# is the binding constraint on reaching it.


def _valid_events(track: TrackRef, *, n: int = 3, month: str = "01") -> list[ListenEvent]:
    return [
        ListenEvent(
            track=track,
            played_at=f"2026-{month}-0{i}T00:00:00Z",
            source="test",
            context=PlayContext(ms_played=180_000),
        )
        for i in range(1, n + 1)
    ]


def test_replay_queue_prints_ceiling_with_no_recommendation_when_target_already_met(
    tmp_path, capsys
):
    track = TrackRef(name="Real Listen", artist="Artist", spotify_id="s1")
    _write_history(tmp_path, _valid_events(track))

    rc = main(["replay-queue", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "coverage ceiling: 100.0% (target 80%)" in out
    assert "recommendation" not in out


def test_replay_queue_recommends_raising_the_cap_when_it_is_the_binding_constraint(
    tmp_path, capsys
):
    tracks = [
        TrackRef(name=f"Track {i}", artist=f"Artist {i}", spotify_id=f"s{i}") for i in range(3)
    ]
    events = [e for t in tracks for e in _valid_events(t)]
    _write_history(tmp_path, events)

    rc = main(["replay-queue", "--data-dir", str(tmp_path), "--cap", "1"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "coverage ceiling: 100.0% (target 80%)" in out
    assert "recommendation: raise --cap" in out


def test_replay_queue_recommends_lowering_min_valid_plays_when_ceiling_itself_is_below_target(
    tmp_path, capsys
):
    eligible = TrackRef(name="Eligible", artist="Artist E", spotify_id="e1")
    ineligible = TrackRef(name="Too Few Plays", artist="Artist F", spotify_id="f1")
    events = _valid_events(eligible) + _valid_events(ineligible, n=2, month="02")
    _write_history(tmp_path, events)

    rc = main(["replay-queue", "--data-dir", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "coverage ceiling: 60.0% (target 80%)" in out
    assert "recommendation: lower --min-valid-plays" in out
