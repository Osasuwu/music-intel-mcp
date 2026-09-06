"""Replay capture contract (#166, pilot slice 1).

Automated playback (#128/#159) and live capture (#124/#139) are today two
loops that never meet: playback advances on wall clock alone, capture starts
whenever it starts, and a short/silent buffer would be embedded and written
first-write-wins, never replaced. This module is the single contract joining
them for the replay pilot: capture is armed on the target process **before**
the play call, the window anchors on the first non-silent frame and lasts
``min(track_duration, max_window_s)``, the driver pauses once the window is
collected, and a capture whose RMS is below threshold or whose buffer came up
short is discarded and journaled rather than silently embedded.

Identity is taken directly from the queue's ``TrackRef`` via
``canonical_track_id`` — this entry point never calls
``LiveIdentityResolver.resolve()`` or computes a chromaprint fingerprint for
identity purposes (decision ``2e17aafa``, refined by the #166 AC1 comment
thread: fpcalc-for-pool-storage is a separate, deferred concern, #140/#161).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from .automated_playback import TrackSkipped
from .capture import LoopbackSource, RingBufferSink
from .inference import AudioEmbeddingModel, ClassifierModel, run_inference
from .models import TrackRef
from .shared_store import canonical_track_id
from .store import UserStore

# Conservative, uncalibrated constants (issue #166 is `afk:2-plan`: these are
# judgement calls the plan pins now, not values derived from a real pool --
# same stance as #140 decision `bdb3c02a`). FakeLoopbackCapture's 0.1-amplitude
# test tone has RMS~=0.0707, comfortably above this threshold.
RMS_SILENCE_THRESHOLD = 0.01
DEFAULT_REPLAY_WINDOW_S = 120.0
SILENCE_POLL_INTERVAL_S = 0.5
MAX_SILENCE_WAIT_S = 10.0

ReplayOutcome = Literal["ok", "silent", "short", "play_failed", "sample_rate_mismatch"]


@runtime_checkable
class ReplayDriver(Protocol):
    """The seam the replay loop drives playback through. Deliberately
    ``TrackRef``-shaped (not ``SpotifyPlaybackClient``'s string-track-id
    ``play``/no-arg ``pause``) so it is testable without a device/token; a
    thin adapter bridges the two in real CLI wiring."""

    def play(self, track: TrackRef) -> None: ...

    def pause(self) -> None: ...


class ReplayJournalEntry(BaseModel):
    """One JSONL line per replay attempt (AC5)."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    outcome: str
    started_at: str
    ended_at: str
    reason: str | None = None


class ReplayLedgerEntry(BaseModel):
    """One JSONL line per replay-loop invocation (#167 AC1) — a coarser,
    per-session window distinct from :class:`ReplayJournalEntry`'s per-track
    granularity. The ESH importer (#167 AC2) reads these windows to tag
    export rows falling inside one as agent-originated."""

    model_config = ConfigDict(extra="forbid")

    account: str
    data_root: str
    started_at: str
    ended_at: str


@dataclass(frozen=True)
class ReplayCaptureOutcome:
    track_id: str
    outcome: ReplayOutcome
    analysis_path: Path | None
    reason: str | None
    started_at: datetime
    ended_at: datetime


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


def append_replay_journal_entry(path: Path, entry: ReplayJournalEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def summarize_replay_journal(path: Path) -> dict[str, int]:
    """Per-outcome counts across every journaled attempt (AC5's CLI summary)."""
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    return counts


def replay_journal_path(store: UserStore) -> Path:
    return store.root / "replay_journal.jsonl"


def replay_ledger_path(store: UserStore) -> Path:
    return store.root / "replay_ledger.jsonl"


def append_replay_ledger_entry(path: Path, entry: ReplayLedgerEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")


def _journal(
    journal_path: Path | None,
    *,
    track_id: str,
    outcome: str,
    started_at: datetime,
    ended_at: datetime,
    reason: str | None,
) -> None:
    if journal_path is None:
        return
    append_replay_journal_entry(
        journal_path,
        ReplayJournalEntry(
            track_id=track_id,
            outcome=outcome,
            started_at=started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            reason=reason,
        ),
    )


def run_replay_capture(
    *,
    track: TrackRef,
    duration_s: float,
    capture: LoopbackSource,
    driver: ReplayDriver,
    store: UserStore,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    journal_path: Path | None = None,
    rms_threshold: float = RMS_SILENCE_THRESHOLD,
    max_window_s: float = DEFAULT_REPLAY_WINDOW_S,
    poll_interval_s: float = SILENCE_POLL_INTERVAL_S,
    max_silence_wait_s: float = MAX_SILENCE_WAIT_S,
    expected_sample_rate: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ReplayCaptureOutcome:
    """Run one replay capture attempt for ``track`` (already resolved by the
    caller's queue selection — no identity resolution happens here)."""
    track_id = canonical_track_id(track)
    started_at = now()
    window_len = min(duration_s, max_window_s)

    def _finish(outcome: ReplayOutcome, *, reason: str | None, analysis_path: Path | None = None):
        ended_at = now()
        _journal(
            journal_path,
            track_id=track_id,
            outcome=outcome,
            started_at=started_at,
            ended_at=ended_at,
            reason=reason,
        )
        return ReplayCaptureOutcome(
            track_id=track_id,
            outcome=outcome,
            analysis_path=analysis_path,
            reason=reason,
            started_at=started_at,
            ended_at=ended_at,
        )

    capture.start()
    try:
        try:
            driver.play(track)
        except TrackSkipped as exc:
            return _finish("play_failed", reason=str(exc) or "play failed")

        sink: RingBufferSink | None = None
        waited = 0.0
        while waited < max_silence_wait_s:
            frame = capture.read(poll_interval_s)
            if expected_sample_rate is not None and frame.sample_rate != expected_sample_rate:
                driver.pause()
                return _finish(
                    "sample_rate_mismatch",
                    reason=f"got {frame.sample_rate}Hz, expected {expected_sample_rate}Hz",
                )
            if _rms(frame.samples) >= rms_threshold:
                sink = RingBufferSink(
                    max_seconds=window_len,
                    sample_rate=frame.sample_rate,
                    channels=frame.samples.shape[1],
                )
                sink.write(frame)
                break
            waited += poll_interval_s

        if sink is None:
            driver.pause()
            return _finish("silent", reason="no signal above threshold before max wait")

        while sink.duration_s < window_len:
            frame = capture.read(poll_interval_s)
            if frame.samples.shape[0] == 0:
                break
            sink.write(frame)

        driver.pause()
    finally:
        capture.stop()

    pcm = sink.read_all()
    if sink.duration_s + 1e-9 < window_len:
        return _finish(
            "short", reason=f"captured {sink.duration_s:.2f}s < window {window_len:.2f}s"
        )
    if _rms(pcm) < rms_threshold:
        return _finish("silent", reason=f"rms below threshold {rms_threshold}")

    inference = run_inference(
        pcm, sample_rate=sink.sample_rate, embedding_model=embedding_model, classifier=classifier
    )
    analysis_path = store.write_audio_analysis(
        track_id=track_id, embedding=inference.embedding, tags=inference.tags
    )
    return _finish("ok", reason=None, analysis_path=analysis_path)


def process_replay_queue(
    *,
    queue: list[TrackRef],
    track_duration_s: Callable[[TrackRef], float],
    capture: LoopbackSource,
    driver: ReplayDriver,
    store: UserStore,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    journal_path: Path | None = None,
    requeue_limit: int = 1,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    account: str | None = None,
    ledger_path: Path | None = None,
    **run_kwargs,
) -> list[ReplayCaptureOutcome]:
    """Drive ``queue`` (mutable, drained in place -- mirrors the CLI's
    existing ``_cmd_automated_playback`` requeue-via-append idiom) through
    :func:`run_replay_capture`, re-queuing a silent/short track once (AC3).

    When ``ledger_path`` is given, the whole invocation's wall-clock span is
    appended as one window to the replay-window ledger (#167 AC1) -- coarser
    than the per-track journal, and appended (never overwritten) so windows
    survive restarts across separate process invocations."""
    results: list[ReplayCaptureOutcome] = []
    requeue_counts: dict[str, int] = {}
    loop_started_at = now()

    while queue:
        track = queue.pop(0)
        outcome = run_replay_capture(
            track=track,
            duration_s=track_duration_s(track),
            capture=capture,
            driver=driver,
            store=store,
            embedding_model=embedding_model,
            classifier=classifier,
            journal_path=journal_path,
            now=now,
            **run_kwargs,
        )
        results.append(outcome)

        if outcome.outcome in ("silent", "short"):
            n = requeue_counts.get(outcome.track_id, 0)
            if n < requeue_limit:
                requeue_counts[outcome.track_id] = n + 1
                queue.append(track)
                ts = now()
                _journal(
                    journal_path,
                    track_id=outcome.track_id,
                    outcome="requeued",
                    started_at=ts,
                    ended_at=ts,
                    reason=outcome.outcome,
                )

    if ledger_path is not None:
        loop_ended_at = now()
        append_replay_ledger_entry(
            ledger_path,
            ReplayLedgerEntry(
                account=account or "",
                data_root=str(store.root),
                started_at=loop_started_at.isoformat(),
                ended_at=loop_ended_at.isoformat(),
            ),
        )

    return results
