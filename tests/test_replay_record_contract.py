"""Replay record contract (#276): yielded outcome, per-attempt ledger, exact
consent scope, capture route on the per-user record.

The window-probe leg of the contract (every accepted capture appends one line
to ``window_probe.jsonl``) is pinned in ``test_window_probe.py``. Everything
here runs against injected fakes, through ``process_replay_queue`` where the
AC names the queue, so the contract is checkable before S5's Spotify adapter.
"""

from __future__ import annotations

import json
from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

from music_intel_mcp.automated_playback import TrackSkipped
from music_intel_mcp.capture import AudioFrame
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import TrackRef
from music_intel_mcp.replay_capture import (
    ReplayLedgerEntry,
    ReplayOutcome,
    process_replay_queue,
    replay_journal_path,
    replay_ledger_path,
    run_replay_capture,
)
from music_intel_mcp.replay_control import NullKeepAwake, RunControl
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore

_EMPTY = np.zeros((0, 1), dtype=np.float32)


def _tone(n: int = 800) -> np.ndarray:
    t = np.arange(n) / 16000
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32).reshape(-1, 1)


def _quiet(n: int = 800) -> np.ndarray:
    return np.zeros((n, 1), dtype=np.float32)


class _Capture:
    """Pops one scripted frame per ``read()``; an exhausted script reads empty."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = list(frames)
        self.starts = 0
        self.reads = 0

    def start(self) -> None:
        self.starts += 1

    def read(self, duration_s: float) -> AudioFrame:
        self.reads += 1
        samples = self._frames.pop(0) if self._frames else _EMPTY
        return AudioFrame(samples=samples, sample_rate=16000)

    def stop(self) -> None: ...


class _Driver:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.played: list[str] = []
        self.pauses = 0

    def play(self, track: TrackRef) -> None:
        self.played.append(track.name)
        if track.name in self.fail_on:
            raise TrackSkipped("device busy")

    def pause(self) -> None:
        self.pauses += 1


def _scripted(script: list[str]):
    """A control source playing ``script``, then ``continue`` for ever. Since
    #279 a yield is a pause from run control (``RunControl``), not #276's
    ``should_yield`` callable; the source that detects an organic session and
    the 30 s wait are S4's, out of scope here."""
    script = list(script)
    return lambda: script.pop(0) if script else "continue"


def _yield_once(driver, *, after: int = 0) -> RunControl:
    """Run control that pauses once, on the poll after ``after`` continues
    (an organic session appears, then goes away)."""
    return RunControl(
        _scripted(["continue"] * after + ["pause", "continue"]),
        driver=driver,
        keep_awake=NullKeepAwake(),
        pause_poll_s=0.0,
        sleep=lambda s: None,
    )


def _track(name: str = "Around the World") -> TrackRef:
    return TrackRef(mbid=f"M-{name}", name=name, artist="Daft Punk")


def _models():
    return {
        "embedding_model": InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        "classifier": InMemoryClassifier(result=ClassifierResult(tags={"g": 0.9})),
    }


def _fast():
    return {"max_window_s": 0.05, "poll_interval_s": 0.01, "max_silence_wait_s": 0.02}


def _journal(store: UserStore) -> list[dict]:
    path = replay_journal_path(store)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- AC1: yielded -------------------------------------------------------- #


def test_yielded_is_a_replay_outcome():
    assert "yielded" in get_args(ReplayOutcome)


def test_a_yield_discards_the_capture_and_journals_yielded(tmp_path):
    store = UserStore(root=tmp_path)
    driver = _Driver()

    outcome = run_replay_capture(
        track=_track(),
        duration_s=0.05,
        capture=_Capture([_tone()]),
        driver=driver,
        store=store,
        capture_route="per_app_output",
        journal_path=replay_journal_path(store),
        control=_yield_once(driver),
        **_models(),
        **_fast(),
    )

    assert outcome.outcome == "yielded"
    assert outcome.analysis_path is None
    assert store.list_audio_analyses() == []
    assert driver.pauses == 1
    assert [e["outcome"] for e in _journal(store)] == ["yielded"]


def test_a_yield_mid_window_discards_the_partial_buffer(tmp_path):
    """The poll also runs while the window fills: a session appearing after
    the anchor frame still throws the partial buffer away."""
    store = UserStore(root=tmp_path)
    driver = _Driver()

    outcome = run_replay_capture(
        track=_track(),
        duration_s=0.1,
        capture=_Capture([_tone(), _tone()]),
        driver=driver,
        store=store,
        capture_route="per_app_output",
        control=_yield_once(driver, after=1),
        **_models(),
        max_window_s=0.1,
        poll_interval_s=0.01,
    )

    assert outcome.outcome == "yielded"
    assert store.list_audio_analyses() == []


def test_a_yielded_track_requeues_without_spending_its_requeue(tmp_path):
    """requeue_limit=1: the yield must not use the one requeue, or the later
    silent attempt would not be retried and the track would never reach ok."""
    store = UserStore(root=tmp_path)
    track = _track()
    queue = [track]

    results = process_replay_queue(
        queue=queue,
        track_duration_s=lambda t: 0.05,
        # attempt 1 yields before any read; attempt 2 reads two quiet frames
        # (silent); attempt 3 gets the tone (ok).
        capture=_Capture([_quiet(), _quiet(), _tone()]),
        driver=_Driver(),
        store=store,
        capture_route="per_app_output",
        journal_path=replay_journal_path(store),
        requeue_limit=1,
        # queue poll, then a pause at the first capture poll
        control_source=_scripted(["continue", "pause", "continue"]),
        pause_poll_s=0.0,
        sleep=lambda s: None,
        **_models(),
        **_fast(),
    )

    assert [r.outcome for r in results] == ["yielded", "silent", "ok"]
    assert queue == []
    journal = _journal(store)
    assert [(e["outcome"], e["reason"]) for e in journal if e["outcome"] == "requeued"] == [
        ("requeued", "yielded"),
        ("requeued", "silent"),
    ]


def test_resume_replays_a_track_whose_only_attempt_yielded(tmp_path):
    """#277 resume after a run that actually yielded: ``yielded`` does not
    settle the track, so the resumed run plays it again."""
    store = UserStore(root=tmp_path)
    journal = replay_journal_path(store)
    first = _Driver()
    run_replay_capture(
        track=_track(),
        duration_s=0.05,
        capture=_Capture([_tone()]),
        driver=first,
        store=store,
        capture_route="per_app_output",
        journal_path=journal,
        control=_yield_once(first),
        **_models(),
        **_fast(),
    )

    driver = _Driver()
    results = process_replay_queue(
        queue=[_track()],
        track_duration_s=lambda t: 0.05,
        capture=_Capture([_tone()]),
        driver=driver,
        store=store,
        capture_route="per_app_output",
        journal_path=journal,
        resume=True,
        **_models(),
        **_fast(),
    )

    assert driver.played == ["Around the World"]
    assert [r.outcome for r in results] == ["ok"]


# --- AC2: ledger per attempt, account mandatory, not applied to exports --- #


def test_the_ledger_gets_one_entry_per_play_attempt(tmp_path):
    """Two attempts (an ok and a play_failed) -> two entries; a track already
    analysed is skipped unplayed and gets none."""
    store = UserStore(root=tmp_path)
    analysed = _track("Already")
    store.write_audio_analysis(
        track_id=canonical_track_id(analysed), embedding=[0.1], tags={"g": 0.5}
    )

    process_replay_queue(
        queue=[_track("Played"), analysed, _track("Busy")],
        track_duration_s=lambda t: 0.05,
        capture=_Capture([_tone(), _tone()]),
        driver=_Driver(fail_on={"Busy"}),
        store=store,
        capture_route="per_app_output",
        account="replay-acct-1",
        ledger_path=replay_ledger_path(store),
        **_models(),
        **_fast(),
    )

    lines = replay_ledger_path(store).read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines]
    assert len(entries) == 2
    assert {e["account"] for e in entries} == {"replay-acct-1"}
    assert all(e["started_at"] <= e["ended_at"] for e in entries)
    assert entries[0]["ended_at"] <= entries[1]["started_at"]


def test_a_ledger_entry_without_an_account_is_refused():
    with pytest.raises(ValidationError):
        ReplayLedgerEntry(
            account="",
            data_root="data",
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
        )


def test_run_replay_capture_refuses_a_ledger_without_an_account_before_capture(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture([_tone()])

    with pytest.raises(ValueError, match="account"):
        run_replay_capture(
            track=_track(),
            duration_s=0.05,
            capture=capture,
            driver=_Driver(),
            store=store,
            capture_route="per_app_output",
            ledger_path=replay_ledger_path(store),
            **_models(),
            **_fast(),
        )

    assert capture.starts == 0
    assert not replay_ledger_path(store).exists()


def test_process_replay_queue_refuses_a_ledger_without_an_account_even_when_empty(tmp_path):
    store = UserStore(root=tmp_path)

    with pytest.raises(ValueError, match="account"):
        process_replay_queue(
            queue=[],
            track_duration_s=lambda t: 0.05,
            capture=_Capture([]),
            driver=_Driver(),
            store=store,
            capture_route="per_app_output",
            ledger_path=replay_ledger_path(store),
            **_models(),
            **_fast(),
        )


# --- AC3: exact consent scope --------------------------------------------- #


@pytest.mark.parametrize(
    ("granted", "checked", "expected"),
    [
        ("automated-playback", "automated-playback", True),
        ("automated-playback", "replay-capture", False),
        ("replay-capture", "automated-playback", False),
        ("replay-capture", "replay-capture", True),
    ],
)
def test_consent_is_checked_against_an_exact_scope(tmp_path, granted, checked, expected):
    store = UserStore(root=tmp_path)
    store.grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope=granted
    )

    assert store.has_automated_playback_consent(checked) is expected


def test_the_consent_scope_is_required(tmp_path):
    store = UserStore(root=tmp_path)
    store.grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="replay-capture"
    )

    with pytest.raises(TypeError):
        store.has_automated_playback_consent()  # type: ignore[call-arg]


# --- AC4: capture route on the per-user record, never on the pool -------- #


@pytest.mark.parametrize("route", ["per_app_output", "muted_endpoint", "cable"])
def test_the_per_user_record_carries_its_capture_route(tmp_path, route):
    store = UserStore(root=tmp_path)

    results = process_replay_queue(
        queue=[_track()],
        track_duration_s=lambda t: 0.05,
        capture=_Capture([_tone()]),
        driver=_Driver(),
        store=store,
        capture_route=route,
        journal_path=replay_journal_path(store),
        **_models(),
        **_fast(),
    )

    payload = json.loads(results[0].analysis_path.read_text(encoding="utf-8"))
    assert payload["capture_route"] == route
    assert store.list_audio_analyses()[0].capture_route == route
    assert [e["capture_route"] for e in _journal(store) if e["outcome"] == "ok"] == [route]


def test_a_pool_record_gains_no_capture_route_and_the_journal_keeps_it(tmp_path):
    """With a pool configured the analysis is written to the pool only
    (decision ``1fe8e95f``), and pool records stay provenance-free
    (``446d7d0a``, #197 open): the key is absent, not ``null``. The per-user
    record of the route is then the journal's ``ok`` entry."""
    store = UserStore(root=tmp_path / "root", pool_root=tmp_path / "pool")

    results = process_replay_queue(
        queue=[_track()],
        track_duration_s=lambda t: 0.05,
        capture=_Capture([_tone()]),
        driver=_Driver(),
        store=store,
        capture_route="cable",
        journal_path=replay_journal_path(store),
        **_models(),
        **_fast(),
    )

    pool_record = json.loads(results[0].analysis_path.read_text(encoding="utf-8"))
    assert results[0].analysis_path.is_relative_to(tmp_path / "pool")
    assert "capture_route" not in pool_record
    ok = [e for e in _journal(store) if e["outcome"] == "ok"]
    assert [e["capture_route"] for e in ok] == ["cable"]


def test_an_unknown_capture_route_is_refused_before_capture(tmp_path):
    capture = _Capture([_tone()])

    with pytest.raises(ValueError, match="capture_route"):
        run_replay_capture(
            track=_track(),
            duration_s=0.05,
            capture=capture,
            driver=_Driver(),
            store=UserStore(root=tmp_path),
            capture_route="speakers",  # type: ignore[arg-type]
            **_models(),
            **_fast(),
        )

    assert capture.starts == 0


def test_the_capture_route_is_required(tmp_path):
    with pytest.raises(TypeError):
        process_replay_queue(  # type: ignore[call-arg]
            queue=[],
            track_duration_s=lambda t: 0.05,
            capture=_Capture([]),
            driver=_Driver(),
            store=UserStore(root=tmp_path),
            **_models(),
        )
