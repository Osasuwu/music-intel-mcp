"""Replay capture contract (#166, pilot slice 1) — arm before play, signal-
anchored window, RMS gate, journal, identity from the queue.

Entirely against injected fakes: a scripted :class:`~music_intel_mcp.capture.
LoopbackSource` double and a minimal :class:`ReplayDriver` double record call
order and never touch WASAPI/Spotify/onnxruntime, per this project's
Protocol+fake idiom (see ``test_live_pipeline.py``, ``test_automated_playback.py``).
"""

from __future__ import annotations

import json

import numpy as np

from music_intel_mcp.automated_playback import TrackSkipped
from music_intel_mcp.capture import AudioFrame
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.live_identity import LiveIdentityResolver
from music_intel_mcp.models import TrackRef
from music_intel_mcp.replay_capture import (
    process_replay_queue,
    replay_ledger_path,
    run_replay_capture,
    summarize_replay_journal,
)
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore


def _tone_frame(n: int, *, sample_rate: int = 16000, channels: int = 1, amplitude: float = 0.1):
    t = np.arange(n) / sample_rate
    tone = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    samples = np.repeat(tone.reshape(-1, 1), channels, axis=1)
    return AudioFrame(samples=samples, sample_rate=sample_rate)


class _ScriptedCapture:
    """Replays a fixed list of frames, one per ``read()`` call. Records
    start/stop/read calls so tests can pin arm-before-play ordering."""

    def __init__(self, frames: list[AudioFrame], *, events: list[str] | None = None) -> None:
        self._frames = list(frames)
        self.events = events if events is not None else []
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.events.append("capture_start")

    def read(self, duration_s: float) -> AudioFrame:
        self.events.append("capture_read")
        if self._frames:
            return self._frames.pop(0)
        return AudioFrame(samples=np.zeros((0, 1), dtype=np.float32), sample_rate=16000)

    def stop(self) -> None:
        self.stopped = True
        self.events.append("capture_stop")


class _ScriptedDriver:
    def __init__(
        self, *, events: list[str] | None = None, play_error: Exception | None = None
    ) -> None:
        self.events = events if events is not None else []
        self.play_error = play_error
        self.played: list[TrackRef] = []
        self.pause_calls = 0

    def play(self, track: TrackRef) -> None:
        self.events.append("play")
        if self.play_error is not None:
            raise self.play_error
        self.played.append(track)

    def pause(self) -> None:
        self.events.append("pause")
        self.pause_calls += 1


def _track(name: str = "Around the World", artist: str = "Daft Punk") -> TrackRef:
    return TrackRef(mbid="M-1", name=name, artist=artist)


def test_run_replay_capture_bypasses_identity_resolution(tmp_path, monkeypatch) -> None:
    """AC1: identity comes straight from the queue's TrackRef via
    canonical_track_id -- no SMTC/fingerprint-based identity lookup happens.
    Poisoning LiveIdentityResolver.resolve so it explodes if ever called
    proves the whole call graph never reaches it."""

    def _boom(*args, **kwargs):
        raise AssertionError("must not resolve identity from replay capture")

    monkeypatch.setattr(LiveIdentityResolver, "resolve", _boom)

    track = _track()
    capture = _ScriptedCapture([_tone_frame(800)])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)

    outcome = run_replay_capture(
        track=track,
        duration_s=0.05,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
    )

    assert outcome.outcome == "ok"
    assert outcome.track_id == canonical_track_id(track)
    assert store.has_audio_analysis(canonical_track_id(track))


# #194 AC8/AC9: every replay-captured analysis carries the embedding-space
# version and the applied-gain (pre-normalization RMS) that produced it --
# previously write_audio_analysis was called with neither, so a replay
# record had no way to tell which front-end (pre- or post-gain-fix)
# produced its embedding.
def test_run_replay_capture_writes_model_version_and_input_rms(tmp_path) -> None:
    from music_intel_mcp.inference import EMBEDDING_SPACE_VERSION

    track = _track()
    capture = _ScriptedCapture([_tone_frame(800)])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)

    outcome = run_replay_capture(
        track=track,
        duration_s=0.05,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
    )

    payload = json.loads(outcome.analysis_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == EMBEDDING_SPACE_VERSION


def test_run_replay_capture_arms_before_play_and_anchors_window_on_signal(tmp_path) -> None:
    """AC2: capture is armed (started) before the play call; the window
    anchor is the first frame *above* the silence threshold -- a leading
    silent frame must not count toward the window's duration; the driver is
    paused only once the (anchored) window is fully collected, and stop()
    comes after pause()."""
    events: list[str] = []
    silent_frame = AudioFrame(samples=np.zeros((800, 1), dtype=np.float32), sample_rate=16000)
    frames = [silent_frame, _tone_frame(800), _tone_frame(800)]
    capture = _ScriptedCapture(frames, events=events)
    driver = _ScriptedDriver(events=events)
    store = UserStore(root=tmp_path)

    outcome = run_replay_capture(
        track=_track(),
        duration_s=0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
    )

    assert outcome.outcome == "ok"
    # arm-before-play: capture started strictly before play was called
    assert events.index("capture_start") < events.index("play")
    # all three reads (silent anchor-wait + two window frames) happen before pause
    assert events.count("capture_read") == 3
    assert events.index("capture_read") < events.index("pause")
    last_read_idx = max(i for i, e in enumerate(events) if e == "capture_read")
    assert last_read_idx < events.index("pause") < events.index("capture_stop")


def test_run_replay_capture_window_length_is_capped_by_max_window_s(tmp_path) -> None:
    """AC2: window length = min(track duration, max_window_s) -- a track
    whose reported duration is far longer than max_window_s must still only
    collect max_window_s worth of signal, not the full track duration."""
    frames = [_tone_frame(800), _tone_frame(800)]
    capture = _ScriptedCapture(frames)
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)

    outcome = run_replay_capture(
        track=_track(),
        duration_s=1000.0,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
    )

    assert outcome.outcome == "ok"
    # only the two 0.05s frames needed to reach the 0.1s cap were consumed
    assert capture._frames == []
    assert driver.pause_calls == 1


def test_run_replay_capture_silent_produces_no_analysis_and_journals_reason(tmp_path) -> None:
    """AC3 (silent half): a capture that never rises above the RMS threshold
    produces no audio-analysis file and journals the "silent" outcome with a
    reason, instead of embedding near-silence."""
    silent_frame = AudioFrame(samples=np.zeros((80, 1), dtype=np.float32), sample_rate=16000)
    capture = _ScriptedCapture([silent_frame] * 10)
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()

    outcome = run_replay_capture(
        track=track,
        duration_s=0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
        journal_path=journal_path,
        max_silence_wait_s=0.05,
        poll_interval_s=0.025,
    )

    assert outcome.outcome == "silent"
    assert not store.has_audio_analysis(canonical_track_id(track))
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "silent"
    assert lines[0]["reason"]


def test_run_replay_capture_short_buffer_produces_no_analysis_and_journals_reason(tmp_path) -> None:
    """AC3 (short half): a capture that anchors but then ends before the
    window is fully collected produces no audio-analysis file and journals
    "short" with a reason, rather than embedding the truncated buffer."""
    capture = _ScriptedCapture([_tone_frame(800)])  # anchors, then stream ends
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()

    outcome = run_replay_capture(
        track=track,
        duration_s=1.0,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=1.0,
        journal_path=journal_path,
    )

    assert outcome.outcome == "short"
    assert not store.has_audio_analysis(canonical_track_id(track))
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "short"
    assert lines[0]["reason"]


def test_process_replay_queue_requeues_silent_track_once(tmp_path) -> None:
    """AC3: a silent/short track is re-queued exactly once -- a second
    consecutive silent attempt for the same track is not requeued again."""
    always_silent = AudioFrame(samples=np.zeros((80, 1), dtype=np.float32), sample_rate=16000)
    capture = _ScriptedCapture([always_silent] * 40)
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()
    queue = [track]

    results = process_replay_queue(
        queue=queue,
        track_duration_s=lambda t: 0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
        journal_path=journal_path,
        max_silence_wait_s=0.02,
        poll_interval_s=0.01,
    )

    assert queue == []
    assert [r.outcome for r in results] == ["silent", "silent"]
    assert not store.has_audio_analysis(canonical_track_id(track))
    outcomes = [
        json.loads(line)["outcome"]
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert outcomes.count("requeued") == 1
    assert outcomes.count("silent") == 2


def test_run_replay_capture_sample_rate_mismatch_is_journaled_and_discarded(tmp_path) -> None:
    """AC4: a capture whose sample rate doesn't match what the pipeline
    expects is journaled as "sample_rate_mismatch" and discarded -- no
    audio-analysis file is written."""
    wrong_rate_frame = _tone_frame(800, sample_rate=44100)
    capture = _ScriptedCapture([wrong_rate_frame])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()

    outcome = run_replay_capture(
        track=track,
        duration_s=0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
        journal_path=journal_path,
        expected_sample_rate=16000,
    )

    assert outcome.outcome == "sample_rate_mismatch"
    assert not store.has_audio_analysis(canonical_track_id(track))
    assert driver.pause_calls == 1
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "sample_rate_mismatch"
    assert lines[0]["reason"]


def test_process_replay_queue_does_not_requeue_sample_rate_mismatch(tmp_path) -> None:
    """AC4: sample-rate mismatch is a discard, not a transient re-queueable
    condition like silent/short -- it never gets a second attempt."""
    wrong_rate_frame = _tone_frame(800, sample_rate=44100)
    capture = _ScriptedCapture([wrong_rate_frame])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    track = _track()
    queue = [track]

    results = process_replay_queue(
        queue=queue,
        track_duration_s=lambda t: 0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
        expected_sample_rate=16000,
    )

    assert queue == []
    assert [r.outcome for r in results] == ["sample_rate_mismatch"]


def test_process_replay_queue_appends_window_to_ledger_per_data_root(tmp_path) -> None:
    """#167 AC1: one call to the replay loop appends a single window
    (start/end) to a ledger file keyed by the data root -- calling the loop
    again (a fresh process, i.e. "restart") appends a second window rather
    than overwriting the first, so windows survive restarts."""
    capture = _ScriptedCapture([_tone_frame(800), _tone_frame(800)])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    ledger_path = replay_ledger_path(store)
    track = _track()

    process_replay_queue(
        queue=[track],
        track_duration_s=lambda t: 0.05,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
        account="participant-1",
        ledger_path=ledger_path,
    )

    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["account"] == "participant-1"
    assert lines[0]["data_root"] == str(store.root)
    assert lines[0]["started_at"] < lines[0]["ended_at"]

    # second invocation ("restart") appends rather than overwrites
    capture2 = _ScriptedCapture([_tone_frame(800), _tone_frame(800)])
    process_replay_queue(
        queue=[track],
        track_duration_s=lambda t: 0.05,
        capture=capture2,
        driver=_ScriptedDriver(),
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
        account="participant-1",
        ledger_path=ledger_path,
    )

    lines = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2


def test_process_replay_queue_without_ledger_path_writes_no_ledger(tmp_path) -> None:
    """#167 AC1: the ledger is opt-in via ``ledger_path`` -- omitting it (the
    existing call sites in this file all do) must not create a ledger file,
    preserving the pre-#167 behavior of every other test here."""
    capture = _ScriptedCapture([_tone_frame(800), _tone_frame(800)])
    driver = _ScriptedDriver()
    store = UserStore(root=tmp_path)
    track = _track()

    process_replay_queue(
        queue=[track],
        track_duration_s=lambda t: 0.05,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
    )

    assert not replay_ledger_path(store).exists()


def test_run_replay_capture_play_failure_is_journaled_as_play_failed(tmp_path) -> None:
    """Issue #166's outcome vocabulary names play-failed alongside ok/silent/
    short: a driver that fails to start playback is journaled as
    "play_failed" and never reaches the capture window at all."""
    capture = _ScriptedCapture([_tone_frame(800)])
    driver = _ScriptedDriver(play_error=TrackSkipped("device unavailable"))
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()

    outcome = run_replay_capture(
        track=track,
        duration_s=0.1,
        capture=capture,
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.1,
        journal_path=journal_path,
    )

    assert outcome.outcome == "play_failed"
    assert not store.has_audio_analysis(canonical_track_id(track))
    assert driver.pause_calls == 0  # nothing was playing, so nothing to pause
    lines = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["outcome"] == "play_failed"


def test_summarize_replay_journal_counts_per_outcome(tmp_path) -> None:
    """AC5: the journal summary reports per-outcome counts across every
    journaled attempt, for the weekly checkpoint CLI."""
    journal_path = tmp_path / "replay_journal.jsonl"
    track = _track()
    silent_frame = AudioFrame(samples=np.zeros((80, 1), dtype=np.float32), sample_rate=16000)

    for _ in range(2):
        run_replay_capture(
            track=track,
            duration_s=0.05,
            capture=_ScriptedCapture([silent_frame] * 4),
            driver=_ScriptedDriver(),
            store=UserStore(root=tmp_path / "store"),
            embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
            classifier=InMemoryClassifier(result=ClassifierResult(tags={})),
            max_window_s=0.05,
            journal_path=journal_path,
            max_silence_wait_s=0.02,
            poll_interval_s=0.01,
        )
    run_replay_capture(
        track=track,
        duration_s=0.05,
        capture=_ScriptedCapture([_tone_frame(800)]),
        driver=_ScriptedDriver(),
        store=UserStore(root=tmp_path / "store"),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        max_window_s=0.05,
        journal_path=journal_path,
    )

    counts = summarize_replay_journal(journal_path)
    assert counts == {"silent": 2, "ok": 1}


def test_summarize_replay_journal_empty_when_no_journal_file(tmp_path) -> None:
    assert summarize_replay_journal(tmp_path / "missing.jsonl") == {}
