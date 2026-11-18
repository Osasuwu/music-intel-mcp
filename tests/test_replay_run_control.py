"""Run control inside ``process_replay_queue`` (#279): one ``control()``
source polled in the queue and capture loops, stop rules over attempt
outcomes, exceptions as journaled outcomes, keep-awake held only while a
capture is in progress, and a journaled stop reason.

Fakes only: a scripted control source, a capture whose signal the driver
picks per track, and a keep-awake fake that records whether it is held.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from music_intel_mcp.automated_playback import TrackSkipped
from music_intel_mcp.capture import AudioFrame
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import TrackRef
from music_intel_mcp.replay_capture import (
    journal_handled_track_ids,
    process_replay_queue,
    replay_journal_path,
    replay_ledger_path,
    summarize_replay_journal,
)
from music_intel_mcp.replay_control import (
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    FAILURE_STREAK_LIMIT,
    SILENT_STREAK_LIMIT,
    RunControl,
    WindowsKeepAwake,
)
from music_intel_mcp.store import UserStore

_EMPTY = np.zeros((0, 1), dtype=np.float32)


def _tone(n: int = 800) -> np.ndarray:
    t = np.arange(n) / 16000
    return (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32).reshape(-1, 1)


class _KeepAwake:
    def __init__(self) -> None:
        self.held = False
        self.events: list[str] = []

    def acquire(self) -> None:
        self.held = True
        self.events.append("acquire")

    def release(self) -> None:
        self.held = False
        self.events.append("release")


class _Capture:
    """Signal per track, chosen by the driver on ``play``: ``loud`` gives a
    0.05 s tone per read, ``quiet`` zeros, ``short`` one tone frame then
    nothing, ``rate`` an 8 kHz frame, ``read_error`` raises. Records whether
    keep-awake was held at every read."""

    def __init__(self, keep_awake: _KeepAwake | None = None, *, start_error: bool = False):
        self.mode = "loud"
        self.keep_awake = keep_awake
        self.held_at_read: list[bool] = []
        self.start_error = start_error
        self._short_sent = False

    def start(self) -> None:
        self._short_sent = False
        if self.start_error:
            raise OSError("loopback device gone")

    def read(self, duration_s: float) -> AudioFrame:
        if self.keep_awake is not None:
            self.held_at_read.append(self.keep_awake.held)
        if self.mode == "read_error":
            raise RuntimeError("WASAPI read failed")
        if self.mode == "rate":
            return AudioFrame(samples=_tone(400), sample_rate=8000)
        if self.mode == "quiet":
            return AudioFrame(samples=np.zeros((800, 1), dtype=np.float32), sample_rate=16000)
        if self.mode == "short":
            if self._short_sent:
                return AudioFrame(samples=_EMPTY, sample_rate=16000)
            self._short_sent = True
            return AudioFrame(samples=_tone(400), sample_rate=16000)
        return AudioFrame(samples=_tone(), sample_rate=16000)

    def stop(self) -> None: ...


class _Driver:
    """Track name prefix picks the behaviour: ``busy`` -> TrackSkipped,
    ``boom`` -> RuntimeError from play, otherwise sets the capture mode named
    by the prefix (``loud``/``quiet``/``short``/``rate``/``read_error``)."""

    def __init__(self, capture: _Capture, *, pause_error: bool = False) -> None:
        self.capture = capture
        self.pause_error = pause_error
        self.played: list[str] = []
        self.pauses = 0

    def play(self, track: TrackRef) -> None:
        self.played.append(track.name)
        kind = track.name.split("-")[0]
        if kind == "busy":
            raise TrackSkipped("device busy")
        if kind == "boom":
            raise RuntimeError("player crashed")
        self.capture.mode = kind

    def pause(self) -> None:
        self.pauses += 1
        if self.pause_error:
            raise ConnectionError("pause call failed")


class _Source:
    """Scripted control source; ``continue`` once the script runs out. Logs
    whether keep-awake was held at every call."""

    def __init__(self, script: list[str], keep_awake: _KeepAwake | None = None) -> None:
        self.script = list(script)
        self.keep_awake = keep_awake
        self.calls: list[tuple[str, bool | None]] = []

    def __call__(self) -> str:
        value = self.script.pop(0) if self.script else "continue"
        self.calls.append((value, self.keep_awake.held if self.keep_awake else None))
        return value


def _t(name: str, i: int = 0) -> TrackRef:
    return TrackRef(mbid=f"M-{name}-{i}", name=f"{name}-{i}", artist="A")


def _run(store, queue, *, capture, driver, keep_awake=None, source=None, **kw):
    duration = kw.pop("duration", 0.05)
    return process_replay_queue(
        queue=queue,
        track_duration_s=lambda t: duration,
        capture=capture,
        driver=driver,
        store=store,
        capture_route="per_app_output",
        journal_path=replay_journal_path(store),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"g": 0.9})),
        max_window_s=kw.pop("max_window_s", 0.05),
        poll_interval_s=0.01,
        max_silence_wait_s=0.02,
        keep_awake=keep_awake or _KeepAwake(),
        control_source=source,
        pause_poll_s=0.0,
        sleep=lambda s: None,
        **kw,
    )


def _journal(store: UserStore) -> list[dict]:
    path = replay_journal_path(store)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _stop_line(store: UserStore) -> dict:
    runs = [e for e in _journal(store) if e["track_id"] == "__run__"]
    assert len(runs) == 1
    assert runs[0]["outcome"] == "run_stopped"
    return runs[0]


# --- AC1: one control() source; pause handled inside it ------------------ #


def test_run_control_hides_pause_from_the_loops():
    driver = _Driver(_Capture())
    keep_awake = _KeepAwake()
    keep_awake.acquire()
    sleeps: list[float] = []
    control = RunControl(
        _Source(["pause", "pause", "continue"]),
        driver=driver,
        keep_awake=keep_awake,
        pause_poll_s=2.0,
        sleep=sleeps.append,
    )

    assert control.poll() == "continue"
    assert control.pause_epoch == 1
    assert driver.pauses == 1
    assert keep_awake.held is False
    assert sleeps == [2.0, 2.0]


def test_a_stop_during_a_pause_comes_out_as_stop():
    control = RunControl(
        _Source(["pause", "stop"]),
        driver=_Driver(_Capture()),
        keep_awake=_KeepAwake(),
        pause_poll_s=0.0,
        sleep=lambda s: None,
    )

    assert control.poll() == "stop"


def test_a_pause_mid_capture_yields_releases_keep_awake_and_requeues(tmp_path):
    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()
    capture = _Capture(keep_awake)
    # queue poll, silence-wait poll, then pause twice during the window fill
    source = _Source(["continue", "continue", "pause", "pause", "continue"], keep_awake)

    results = _run(
        store,
        [_t("loud")],
        capture=capture,
        driver=_Driver(capture),
        keep_awake=keep_awake,
        source=source,
        duration=0.1,
        max_window_s=0.1,
    )

    assert [r.outcome for r in results] == ["yielded", "ok"]
    # every source call made while paused (the 2nd "pause" and the
    # "continue" that ends it) saw keep-awake released
    assert [held for _, held in source.calls[3:5]] == [False, False]
    # and every capture read, before and after the pause, held it
    assert capture.held_at_read and all(capture.held_at_read)
    assert keep_awake.held is False
    assert [(e["outcome"], e["reason"]) for e in _journal(store) if e["outcome"] == "requeued"] == [
        ("requeued", "yielded")
    ]


def test_a_stop_from_control_mid_capture_ends_the_attempt_and_the_run(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("loud", 1), _t("loud", 2)]

    results = _run(
        store,
        queue,
        capture=capture,
        driver=_Driver(capture),
        source=_Source(["continue", "stop"]),
        account="replay-acct",
        ledger_path=replay_ledger_path(store),
    )

    assert [r.outcome for r in results] == ["stopped"]
    assert store.list_audio_analyses() == []
    assert queue == [_t("loud", 2)]  # the stopped track is not requeued
    assert _stop_line(store)["reason"] == "control_stop"
    assert len(replay_ledger_path(store).read_text(encoding="utf-8").splitlines()) == 1


def test_a_stop_stays_a_stop_when_the_driver_pause_fails(tmp_path):
    """Review finding: ``driver.pause()`` raising on the stop path is logged,
    like in ``RunControl.poll()``; the attempt still ends ``stopped`` and the
    run halts, even though the source reports ``stop`` only once."""
    store = UserStore(root=tmp_path)
    capture = _Capture()
    driver = _Driver(capture, pause_error=True)
    queue = [_t("loud", 1), _t("loud", 2)]

    results = _run(
        store,
        queue,
        capture=capture,
        driver=driver,
        source=_Source(["continue", "stop"]),
        requeue_limit=1,
    )

    assert [r.outcome for r in results] == ["stopped"]
    assert driver.played == ["loud-1"]
    assert driver.pauses == 1  # attempted once on the stop path, not again in finally
    assert queue == [_t("loud", 2)]
    assert [e["outcome"] for e in _journal(store) if e["outcome"] == "error"] == []
    assert _stop_line(store)["reason"] == "control_stop"


def test_a_failing_keep_awake_release_does_not_end_the_run(tmp_path):
    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()

    def _boom() -> None:
        raise OSError("SetThreadExecutionState failed")

    keep_awake.release = _boom  # type: ignore[method-assign]
    capture = _Capture()

    results = _run(
        store,
        [_t("loud", 1), _t("loud", 2)],
        capture=capture,
        driver=_Driver(capture),
        keep_awake=keep_awake,
    )

    assert [r.outcome for r in results] == ["ok", "ok"]
    assert _stop_line(store)["reason"] == "queue_empty"


def test_a_stop_before_a_track_plays_nothing(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    driver = _Driver(capture)

    results = _run(store, [_t("loud")], capture=capture, driver=driver, source=_Source(["stop"]))

    assert results == []
    assert driver.played == []
    assert _stop_line(store)["reason"] == "control_stop"


# --- AC2: five consecutive silent captures stop the run ------------------ #


def test_five_consecutive_silent_captures_stop_the_run(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("quiet", i) for i in range(SILENT_STREAK_LIMIT + 1)]

    results = _run(store, queue, capture=capture, driver=_Driver(capture), requeue_limit=0)

    assert SILENT_STREAK_LIMIT == 5
    assert [r.outcome for r in results] == ["silent"] * 5
    assert len(queue) == 1
    assert _stop_line(store)["reason"] == "silent_streak"


def test_an_ok_resets_the_silent_streak(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    quiet = [_t("quiet", i) for i in range(8)]
    queue = quiet[:4] + [_t("loud")] + quiet[4:]

    results = _run(store, queue, capture=capture, driver=_Driver(capture), requeue_limit=0)

    assert [r.outcome for r in results].count("silent") == 8
    assert _stop_line(store)["reason"] == "queue_empty"


# --- AC3: three consecutive non-silent failures stop the run ------------- #


def test_three_mixed_non_silent_failures_stop_the_run(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("busy"), _t("boom"), _t("rate"), _t("loud")]

    results = _run(
        store,
        queue,
        capture=capture,
        driver=_Driver(capture),
        requeue_limit=0,
        expected_sample_rate=16000,
    )

    assert FAILURE_STREAK_LIMIT == 3
    assert [r.outcome for r in results] == ["play_failed", "error", "sample_rate_mismatch"]
    assert queue == [_t("loud")]
    assert _stop_line(store)["reason"] == "failure_streak"


def test_short_captures_count_as_non_silent_failures(tmp_path):
    """Decision (Пётр, 16.11): a short capture is not a silent failure -- it
    counts toward the three-failure streak, not the five-silent one."""
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("short", i) for i in range(4)]

    results = _run(store, queue, capture=capture, driver=_Driver(capture), requeue_limit=0)

    assert [r.outcome for r in results] == ["short"] * 3
    assert _stop_line(store)["reason"] == "failure_streak"


def test_an_ok_resets_the_failure_streak(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("busy", 1), _t("boom", 1), _t("loud"), _t("busy", 2), _t("boom", 2)]

    results = _run(store, queue, capture=capture, driver=_Driver(capture), requeue_limit=0)

    assert [r.outcome for r in results] == ["play_failed", "error", "ok", "play_failed", "error"]
    assert _stop_line(store)["reason"] == "queue_empty"


# --- AC4: exceptions become journaled outcomes; pause in finally --------- #


@pytest.mark.parametrize(
    ("name", "reason"),
    [("boom", "RuntimeError: player crashed"), ("read_error", "RuntimeError: WASAPI read failed")],
)
def test_a_driver_or_capture_exception_is_a_journaled_error_and_the_run_goes_on(
    tmp_path, name, reason
):
    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()
    capture = _Capture(keep_awake)
    driver = _Driver(capture)

    results = _run(
        store,
        [_t(name), _t("loud")],
        capture=capture,
        driver=driver,
        keep_awake=keep_awake,
        requeue_limit=0,
    )

    assert [r.outcome for r in results] == ["error", "ok"]
    errors = [e for e in _journal(store) if e["outcome"] == "error"]
    assert [e["reason"] for e in errors] == [reason]
    assert driver.pauses >= 2  # the errored attempt's pause in finally, and the ok one
    assert keep_awake.events[:2] == ["acquire", "release"]
    assert keep_awake.held is False


@pytest.mark.parametrize("step", ["run_inference", "write_audio_analysis"])
def test_a_post_capture_exception_is_a_journaled_error_and_the_queue_moves_on(
    tmp_path, monkeypatch, step
):
    """#279 AC4 (review finding): a failure after the capture (inference,
    the analysis write) is that track's ``error``, not an escape from the run."""
    import music_intel_mcp.replay_capture as rc

    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()
    capture = _Capture(keep_awake)
    driver = _Driver(capture)
    calls = {"n": 0}
    real_inference = rc.run_inference
    real_write = UserStore.write_audio_analysis

    def _flaky_inference(*a, **kw):
        calls["n"] += 1
        if step == "run_inference" and calls["n"] == 1:
            raise RuntimeError("inference blew up")
        return real_inference(*a, **kw)

    def _flaky_write(self, **kw):
        if step == "write_audio_analysis" and calls["n"] == 1:
            raise OSError("disk full")
        return real_write(self, **kw)

    monkeypatch.setattr(rc, "run_inference", _flaky_inference)
    monkeypatch.setattr(UserStore, "write_audio_analysis", _flaky_write)

    results = _run(
        store,
        [_t("loud", 0), _t("loud", 1)],
        capture=capture,
        driver=driver,
        keep_awake=keep_awake,
        requeue_limit=0,
    )

    assert [r.outcome for r in results] == ["error", "ok"]
    reason = "RuntimeError: inference blew up" if step == "run_inference" else "OSError: disk full"
    assert [e["reason"] for e in _journal(store) if e["outcome"] == "error"] == [reason]
    assert keep_awake.held is False


def test_a_capture_start_exception_is_a_journaled_error(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture(start_error=True)

    results = _run(store, [_t("loud")], capture=capture, driver=_Driver(capture), requeue_limit=0)

    assert [(r.outcome, r.reason) for r in results] == [("error", "OSError: loopback device gone")]


def test_pause_is_attempted_in_finally_and_its_own_failure_does_not_escape(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()
    driver = _Driver(capture, pause_error=True)

    results = _run(store, [_t("boom")], capture=capture, driver=driver, requeue_limit=0)

    assert [r.outcome for r in results] == ["error"]
    assert driver.pauses == 1


def test_an_error_spends_a_requeue(tmp_path):
    """Decision (Пётр, 16.11): ``error`` spends a requeue, like silent/short."""
    store = UserStore(root=tmp_path)
    capture = _Capture()
    queue = [_t("boom")]

    results = _run(store, queue, capture=capture, driver=_Driver(capture), requeue_limit=1)

    assert [r.outcome for r in results] == ["error", "error"]
    assert [(e["outcome"], e["reason"]) for e in _journal(store) if e["outcome"] == "requeued"] == [
        ("requeued", "error")
    ]


def test_keyboard_interrupt_still_propagates_after_cleanup(tmp_path):
    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()
    capture = _Capture(keep_awake)

    class _Interrupting(_Driver):
        def play(self, track: TrackRef) -> None:
            super().play(track)
            raise KeyboardInterrupt

    driver = _Interrupting(capture)
    with pytest.raises(KeyboardInterrupt):
        _run(store, [_t("loud")], capture=capture, driver=driver, keep_awake=keep_awake)

    assert driver.pauses == 1
    assert keep_awake.held is False


# --- AC5: keep-awake only while a capture is in progress ----------------- #


def test_keep_awake_is_held_only_while_capturing(tmp_path):
    store = UserStore(root=tmp_path)
    keep_awake = _KeepAwake()
    capture = _Capture(keep_awake)
    source = _Source([], keep_awake)

    _run(
        store,
        [_t("loud", 1), _t("loud", 2)],
        capture=capture,
        driver=_Driver(capture),
        keep_awake=keep_awake,
        source=source,
    )

    assert capture.held_at_read and all(capture.held_at_read)
    # the queue-loop polls (before each track) run outside a capture
    queue_polls = [held for value, held in source.calls if value == "continue"]
    assert False in queue_polls
    assert keep_awake.events == ["acquire", "release", "acquire", "release", "release"]
    assert keep_awake.held is False


def test_windows_keep_awake_sets_and_clears_the_execution_state():
    calls: list[int] = []
    keep_awake = WindowsKeepAwake(set_state=calls.append)

    keep_awake.acquire()
    keep_awake.acquire()
    keep_awake.release()
    keep_awake.release()

    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


# --- AC6: the stop reason is journaled ----------------------------------- #


def test_a_drained_queue_journals_queue_empty_and_resume_ignores_the_line(tmp_path):
    store = UserStore(root=tmp_path)
    capture = _Capture()

    _run(store, [_t("loud")], capture=capture, driver=_Driver(capture))

    line = _stop_line(store)
    assert line["reason"] == "queue_empty"
    assert "__run__" not in journal_handled_track_ids(replay_journal_path(store))
    assert summarize_replay_journal(replay_journal_path(store))["run_stopped"] == 1
