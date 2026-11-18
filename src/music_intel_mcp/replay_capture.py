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

Record contract (#276, grill 2026-11-03):

- **Yield.** A pause from run control (#279: a pause handled inside
  :class:`~music_intel_mcp.replay_control.RunControl`, polled before every
  capture read) ends the attempt ``yielded`` and discards its buffer.
  ``process_replay_queue`` puts a yielded track at the back of the queue
  without spending its requeue. (#276 shipped this as a ``should_yield``
  callable; #279 folded it into run control -- one control path.)
- **Ledger.** With ``ledger_path`` set, every play attempt appends one
  :class:`ReplayLedgerEntry` -- every outcome, ``play_failed`` and ``yielded``
  included -- and an empty ``account`` is refused before capture starts. The
  ledger audits the dedicated replay account's windows; it is not applied to a
  participant's own history export.
- **Capture route.** ``capture_route`` (:data:`CaptureRoute`) is required. It
  goes on the per-user analysis record (the root record) and on the ``ok``
  journal entry; a pool record gains no field (pool records stay
  provenance-free, ``446d7d0a``, #197), so under a pool the journal entry is
  the per-user record of the route.
- **Window probe.** Every accepted capture appends exactly one
  :class:`~music_intel_mcp.window_probe.WindowPair` to
  ``<data_root>/window_probe.jsonl`` -- written here, not by a caller-supplied
  rider (decision (a) on #276, 16.11, replacing the #169 rider pattern). The
  long leg reuses this capture's inference embedding; the short leg costs one
  extra embedding pass over a front truncation of the same buffer.

Run control (#279, decision ``81859701``): one ``control_source`` polled in
the queue loop and before every capture read, stop rules over consecutive
attempt outcomes, every ``Exception`` from the driver or the capture as a
journaled ``error`` outcome, keep-awake held only while a capture is in
progress, and one ``run_stopped`` journal line with the stop reason. See
:mod:`~music_intel_mcp.replay_control` and :func:`process_replay_queue`.

``on_capture_analyzed`` stays as an optional observer, called once per
*accepted* capture with the analyzed PCM and its inference result; a
discarded silent/short/yielded capture never reaches it.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, get_args, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .automated_playback import TrackSkipped
from .capture import LoopbackSource, RingBufferSink
from .inference import (
    EMBEDDING_SPACE_VERSION,
    AudioEmbeddingModel,
    ClassifierModel,
    run_inference,
)
from .models import TrackRef
from .replay_control import (
    DEFAULT_PAUSE_POLL_S,
    FAILURE_STREAK_LIMIT,
    SILENT_STREAK_LIMIT,
    ControlSignal,
    KeepAwake,
    NullKeepAwake,
    RunControl,
)
from .shared_store import canonical_track_id
from .store import UserStore
from .window_probe import (
    DEFAULT_SHORT_WINDOW_S,
    append_window_pair,
    embed_window_pair,
    window_probe_path,
)

logger = logging.getLogger(__name__)

# Conservative, uncalibrated constants (issue #166 is `afk:2-plan`: these are
# judgement calls the plan pins now, not values derived from a real pool --
# same stance as #140 decision `bdb3c02a`). FakeLoopbackCapture's 0.1-amplitude
# test tone has RMS~=0.0707, comfortably above this threshold.
RMS_SILENCE_THRESHOLD = 0.01
DEFAULT_REPLAY_WINDOW_S = 120.0
SILENCE_POLL_INTERVAL_S = 0.5
MAX_SILENCE_WAIT_S = 10.0

ReplayOutcome = Literal[
    "ok",
    "silent",
    "short",
    "play_failed",
    "sample_rate_mismatch",
    "yielded",
    "error",
    "stopped",
]

CaptureRoute = Literal["per_app_output", "muted_endpoint", "cable"]
"""Where the replay browser's audio was routed (grill 2026-11-03 D17b): its
own per-process output on an unused endpoint, the muted default endpoint, or
a virtual cable."""

CAPTURE_ROUTES: tuple[str, ...] = get_args(CaptureRoute)


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
    capture_route: str | None = None  # #276: set on ``ok`` entries only


class ReplayLedgerEntry(BaseModel):
    """One JSONL line per replay *play attempt* (#276, grill 2026-11-03 D3b;
    was one per loop invocation under #167 AC1). It audits the dedicated
    replay account's windows; an entry without an account is refused. Not
    applied to participants' history exports any more (D3: with a dedicated
    account every export row is organic listening)."""

    model_config = ConfigDict(extra="forbid")

    account: str = Field(min_length=1)
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


RESUME_SKIP_ATTEMPTS = 2  # silent/short attempts after which a track is given up on


def journal_handled_track_ids(path: Path) -> set[str]:
    """Track ids a resumed run must not play again (#277): any with an ``ok``
    entry, and any with ``RESUME_SKIP_ATTEMPTS`` ``silent``/``short`` entries
    (the first attempt plus its one requeue). ``requeued``, ``yielded`` and
    ``play_failed`` entries do not count. A missing journal handles nothing."""
    if not path.exists():
        return set()
    done: set[str] = set()
    failures: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        track_id, outcome = entry["track_id"], entry["outcome"]
        if outcome == "ok":
            done.add(track_id)
        elif outcome in ("silent", "short"):
            failures[track_id] = failures.get(track_id, 0) + 1
    done.update(t for t, n in failures.items() if n >= RESUME_SKIP_ATTEMPTS)
    return done


def is_track_analysed(store: UserStore, track: TrackRef) -> bool:
    """#277: a track is analysed when an audio-analysis record exists under
    its ``spotify:`` key or its ``mbid:`` key (or its canonical key)."""
    keys = [canonical_track_id(track)]
    if track.spotify_id:
        keys.append(f"spotify:{track.spotify_id}")
    if track.mbid:
        keys.append(f"mbid:{track.mbid}")
    return any(store.has_audio_analysis(key) for key in keys)


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
    capture_route: str | None = None,
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
            capture_route=capture_route,
        ),
    )


_YIELD_REASON = "organic session playing"
_STOP_REASON = "control stop"
RUN_TRACK_ID = "__run__"
"""``track_id`` of the one journal line a run writes when it ends (#279)."""

RunStopReason = Literal["silent_streak", "failure_streak", "control_stop", "queue_empty"]

_FAILURE_OUTCOMES = frozenset({"play_failed", "sample_rate_mismatch", "error", "short"})
_REQUEUED_OUTCOMES = frozenset({"silent", "short", "error"})


def _require_account(ledger_path: Path | None, account: str | None) -> None:
    """#276 AC2: a ledger entry without an account is refused -- up front,
    before anything plays, not when the first entry is written."""
    if ledger_path is not None and not account:
        raise ValueError("a replay ledger needs the replay account (account is empty)")


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
    on_capture_analyzed: Callable[..., None] | None = None,
    control: RunControl | None = None,
    keep_awake: KeepAwake | None = None,
    account: str | None = None,
    ledger_path: Path | None = None,
    probe_short_window_s: float = DEFAULT_SHORT_WINDOW_S,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    capture_route: CaptureRoute,
) -> ReplayCaptureOutcome:
    """Run one replay capture attempt for ``track`` (already resolved by the
    caller's queue selection — no identity resolution happens here).

    ``control`` (#279) is polled before every capture read: ``stop`` ends the
    attempt ``stopped``; a pause handled inside it (its ``pause_epoch``
    moved) ends the attempt ``yielded``. Either way the buffer is discarded.
    ``keep_awake`` is held from the moment the capture is armed until the
    attempt's ``finally``. Any ``Exception`` from the driver or the capture
    ends the attempt ``error`` (reason ``"<ExcType>: <msg>"``); the
    ``finally`` attempts ``driver.pause()`` once play was attempted, and a
    failure there is logged, not raised. The same holds for the steps after
    the capture (inference, the analysis write, the window-pair probe line,
    ``on_capture_analyzed``): an ``Exception`` there is that track's ``error``.
    ``KeyboardInterrupt`` propagates after the same cleanup.

    Raises ``ValueError`` before arming the capture when ``capture_route`` is
    not a :data:`CaptureRoute` or when ``ledger_path`` is set without an
    ``account``."""
    if capture_route not in CAPTURE_ROUTES:
        raise ValueError(f"capture_route must be one of {CAPTURE_ROUTES}, got {capture_route!r}")
    _require_account(ledger_path, account)
    keep_awake = keep_awake if keep_awake is not None else NullKeepAwake()
    track_id = canonical_track_id(track)
    started_at = now()
    window_len = min(duration_s, max_window_s)
    epoch0 = control.pause_epoch if control is not None else 0

    def _finish(outcome: ReplayOutcome, *, reason: str | None, analysis_path: Path | None = None):
        ended_at = now()
        _journal(
            journal_path,
            track_id=track_id,
            outcome=outcome,
            started_at=started_at,
            ended_at=ended_at,
            reason=reason,
            capture_route=capture_route if outcome == "ok" else None,
        )
        if ledger_path is not None:
            append_replay_ledger_entry(
                ledger_path,
                ReplayLedgerEntry(
                    account=account or "",
                    data_root=str(store.root),
                    started_at=started_at.isoformat(),
                    ended_at=ended_at.isoformat(),
                ),
            )
        return ReplayCaptureOutcome(
            track_id=track_id,
            outcome=outcome,
            analysis_path=analysis_path,
            reason=reason,
            started_at=started_at,
            ended_at=ended_at,
        )

    playing = False  # play was attempted and did not report TrackSkipped
    paused = False  # the driver was paused on this attempt's own path

    def _pause() -> None:
        nonlocal paused
        driver.pause()
        paused = True

    def _interrupted() -> ReplayOutcome | None:
        """Poll ``control`` before a read: ``stopped``, ``yielded`` or None."""
        nonlocal paused
        if control is None:
            return None
        if control.poll() == "stop":
            _pause()
            return "stopped"
        if control.pause_epoch != epoch0:
            paused = True  # RunControl paused the driver when the pause began
            return "yielded"
        return None

    sink: RingBufferSink | None = None
    try:
        capture.start()
        keep_awake.acquire()
        playing = True
        try:
            driver.play(track)
        except TrackSkipped as exc:
            playing = False
            return _finish("play_failed", reason=str(exc) or "play failed")

        waited = 0.0
        while waited < max_silence_wait_s:
            if (stop := _interrupted()) is not None:
                return _finish(stop, reason=_STOP_REASON if stop == "stopped" else _YIELD_REASON)
            frame = capture.read(poll_interval_s)
            if expected_sample_rate is not None and frame.sample_rate != expected_sample_rate:
                _pause()
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
            _pause()
            return _finish("silent", reason="no signal above threshold before max wait")

        while sink.duration_s < window_len:
            if (stop := _interrupted()) is not None:
                return _finish(stop, reason=_STOP_REASON if stop == "stopped" else _YIELD_REASON)
            frame = capture.read(poll_interval_s)
            if frame.samples.shape[0] == 0:
                break
            sink.write(frame)

        _pause()
    except Exception as exc:  # #279 AC4: never escapes the run; BaseException does
        return _finish("error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if playing and not paused:
            try:
                driver.pause()
            except Exception:
                logger.exception("driver.pause() failed in the capture's finally")
        try:
            capture.stop()
        except Exception:
            logger.exception("capture.stop() failed in the capture's finally")
        keep_awake.release()

    pcm = sink.read_all()
    if sink.duration_s + 1e-9 < window_len:
        return _finish(
            "short", reason=f"captured {sink.duration_s:.2f}s < window {window_len:.2f}s"
        )
    if _rms(pcm) < rms_threshold:
        return _finish("silent", reason=f"rms below threshold {rms_threshold}")

    try:
        inference = run_inference(
            pcm,
            sample_rate=sink.sample_rate,
            embedding_model=embedding_model,
            classifier=classifier,
        )
        analysis_path = store.write_audio_analysis(
            track_id=track_id,
            embedding=inference.embedding,
            tags=inference.tags,
            model_version=EMBEDDING_SPACE_VERSION,
            input_rms=inference.input_rms,
            capture_route=capture_route,
        )
        # #276 AC5: exactly one window-probe line per accepted capture, written
        # here rather than by a rider (decision (a), 16.11). Same buffer, and the
        # long leg is the embedding just computed.
        append_window_pair(
            window_probe_path(store),
            embed_window_pair(
                track_id=track_id,
                pcm=pcm,
                sample_rate=sink.sample_rate,
                embedding_model=embedding_model,
                short_window_s=probe_short_window_s,
                long_embedding=inference.embedding,
            ),
        )
        if on_capture_analyzed is not None:
            # Passive observer of an accepted capture, given the analyzed PCM,
            # never the raw frames.
            on_capture_analyzed(
                track_id=track_id,
                pcm=pcm,
                sample_rate=sink.sample_rate,
                embedding=inference.embedding,
                tags=inference.tags,
            )
    except Exception as exc:  # #279 AC4: a post-capture failure is this track's error
        return _finish("error", reason=f"{type(exc).__name__}: {exc}")
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
    resume: bool = False,
    capture_route: CaptureRoute,
    control_source: Callable[[], ControlSignal] | None = None,
    keep_awake: KeepAwake | None = None,
    pause_poll_s: float = DEFAULT_PAUSE_POLL_S,
    sleep: Callable[[float], None] = time.sleep,
    **run_kwargs,
) -> list[ReplayCaptureOutcome]:
    """Drive ``queue`` (mutable, drained in place -- mirrors the CLI's
    existing ``_cmd_automated_playback`` requeue-via-append idiom) through
    :func:`run_replay_capture`, re-queuing a ``silent``/``short``/``error``
    track up to ``requeue_limit`` times (AC3; ``error`` spends a requeue,
    Пётр 16.11). A ``yielded`` track goes to the back of the queue and
    journals ``requeued`` (reason ``yielded``) without spending its requeue
    (#276); a ``stopped`` one is not requeued.

    Run control (#279): ``control_source`` is wrapped in one
    :class:`~music_intel_mcp.replay_control.RunControl`, polled before each
    track and before every capture read; ``stop`` ends the run. The run also
    stops after :data:`SILENT_STREAK_LIMIT` consecutive ``silent`` attempts
    or :data:`FAILURE_STREAK_LIMIT` consecutive ``play_failed`` /
    ``sample_rate_mismatch`` / ``error`` / ``short`` ones; ``ok`` resets both
    streaks, ``yielded`` and ``stopped`` leave them. ``keep_awake`` is
    released at run end whatever happens. With ``journal_path`` set, the run
    ends with one journal line, ``track_id`` :data:`RUN_TRACK_ID`, outcome
    ``run_stopped`` and a :data:`RunStopReason` -- not written when an
    exception (``KeyboardInterrupt``) propagates.

    When ``ledger_path`` is given, each play attempt appends one window to the
    replay ledger (#276; appended, never overwritten, so windows survive
    restarts); ``account`` is then required, checked before the loop starts.

    ``resume=True`` (needs ``journal_path``) first drops every queued track the
    journal already settles (:func:`journal_handled_track_ids`, #277). Resuming
    or not, each track is checked with :func:`is_track_analysed` immediately
    before it is played and skipped, unplayed and unjournaled, when an analysis
    already exists."""
    _require_account(ledger_path, account)
    keep_awake = keep_awake if keep_awake is not None else NullKeepAwake()
    control = (
        RunControl(
            control_source,
            driver=driver,
            keep_awake=keep_awake,
            pause_poll_s=pause_poll_s,
            sleep=sleep,
        )
        if control_source is not None
        else None
    )
    run_started_at = now()
    results: list[ReplayCaptureOutcome] = []
    if resume and journal_path is not None:
        handled = journal_handled_track_ids(journal_path)
        queue[:] = [t for t in queue if canonical_track_id(t) not in handled]
    requeue_counts: dict[str, int] = {}
    silent_streak = 0
    failure_streak = 0
    stop_reason: RunStopReason = "queue_empty"

    def _requeued(track: TrackRef, track_id: str, reason: str) -> None:
        queue.append(track)
        ts = now()
        _journal(
            journal_path,
            track_id=track_id,
            outcome="requeued",
            started_at=ts,
            ended_at=ts,
            reason=reason,
        )

    try:
        while queue:
            if control is not None and control.poll() == "stop":
                stop_reason = "control_stop"
                break
            track = queue.pop(0)
            if is_track_analysed(store, track):
                continue
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
                account=account,
                ledger_path=ledger_path,
                capture_route=capture_route,
                control=control,
                keep_awake=keep_awake,
                **run_kwargs,
            )
            results.append(outcome)

            if outcome.outcome == "stopped":
                stop_reason = "control_stop"
                break
            if outcome.outcome == "ok":
                silent_streak = failure_streak = 0
            elif outcome.outcome == "silent":
                silent_streak += 1
            elif outcome.outcome in _FAILURE_OUTCOMES:
                failure_streak += 1

            if outcome.outcome == "yielded":
                _requeued(track, outcome.track_id, "yielded")
            elif outcome.outcome in _REQUEUED_OUTCOMES:
                n = requeue_counts.get(outcome.track_id, 0)
                if n < requeue_limit:
                    requeue_counts[outcome.track_id] = n + 1
                    _requeued(track, outcome.track_id, outcome.outcome)

            if silent_streak >= SILENT_STREAK_LIMIT:
                stop_reason = "silent_streak"
                break
            if failure_streak >= FAILURE_STREAK_LIMIT:
                stop_reason = "failure_streak"
                break

        _journal(
            journal_path,
            track_id=RUN_TRACK_ID,
            outcome="run_stopped",
            started_at=run_started_at,
            ended_at=now(),
            reason=stop_reason,
        )
    finally:
        keep_awake.release()

    return results
