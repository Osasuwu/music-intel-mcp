"""YouTube stream-decode capture (#170 AC1) — post-pilot follow-up.

The pilot's live capture path (``capture.py``) listens to loopback audio from
a playing desktop app. A YouTube Music participant instead needs the audio
decoded directly from a ``youtube_id`` — no playback device involved. This
module is the decode-side counterpart to ``capture.LoopbackSource``: a
``StreamDecodeSource`` Protocol seam with a real yt-dlp-backed implementation
and an in-memory fake, mirroring that module's
``LoopbackSource``/``FakeLoopbackCapture``/``WasapiProcessLoopbackCapture``
triad so ``decode_and_run_inference`` below is fully testable without
network access or yt-dlp installed.

``yt-dlp`` is an optional dependency (AC1): imported lazily inside
``YtDlpStreamDecodeSource.decode`` (mirrors ``live_identity.py``'s
``AcoustIdApiSource.match`` lazy ``httpx`` import) so the module — and the
package as a whole — stays importable without it. Its absence surfaces as
``YtDlpNotInstalledError`` only when ``decode`` is actually called, not at
import time.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .capture import AudioFrame
from .inference import AudioEmbeddingModel, ClassifierModel, InferenceResult, run_inference
from .live_pipeline import LiveCaptureOutcome
from .models import TrackRef
from .replay_capture import ReplayJournalEntry, append_replay_journal_entry
from .shared_store import canonical_track_id
from .store import UserStore


class YtDlpNotInstalledError(RuntimeError):
    """Raised by :meth:`YtDlpStreamDecodeSource.decode` when ``yt-dlp`` is not
    installed — the branch degrades (a catchable error at call time) rather
    than breaking the package (#170 AC1)."""


class VideoUnavailableError(RuntimeError):
    """Raised by a :class:`StreamDecodeSource` when the extractor fails
    because the video itself is gone -- removed, private, or region-locked
    (#170 AC2). ``YtDlpStreamDecodeSource`` raises this from a caught
    ``yt_dlp.utils.DownloadError``; deliberately not distinguished further
    (removed vs. private vs. region-locked vs. a transient extractor error)
    since separating those would need parsing yt-dlp's error-message text --
    unwarranted complexity for a locally-run pilot tool where "the extractor
    could not get this video" is already the actionable signal."""


@runtime_checkable
class StreamDecodeSource(Protocol):
    """decode seam every stream-decode backend (real or fake) implements —
    mirrors ``capture.py``'s ``LoopbackSource`` Protocol."""

    def decode(self, youtube_id: str) -> AudioFrame: ...


class FakeStreamDecodeSource:
    """Test double for :class:`StreamDecodeSource` — synthesizes a stereo
    sine tone instead of touching yt-dlp/network, mirroring ``capture.py``'s
    ``FakeLoopbackCapture``."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 2,
        duration_s: float = 1.0,
        frequency_hz: float = 440.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration_s = duration_s
        self.frequency_hz = frequency_hz
        self.decoded_youtube_ids: list[str] = []

    def decode(self, youtube_id: str) -> AudioFrame:
        self.decoded_youtube_ids.append(youtube_id)
        n = max(0, int(self.duration_s * self.sample_rate))
        t = np.arange(n) / self.sample_rate
        tone = (0.1 * np.sin(2 * np.pi * self.frequency_hz * t)).astype(np.float32)
        samples = np.repeat(tone.reshape(-1, 1), self.channels, axis=1)
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)


class YtDlpStreamDecodeSource:
    """Real :class:`StreamDecodeSource` backend — resolves ``youtube_id`` to a
    direct audio-stream URL via yt-dlp, then decodes it to stereo PCM locally
    via an ``ffmpeg`` subprocess (decision ``29743073``: decode runs locally
    beside the desktop app; yt-dlp is never vendored, only its extraction
    step used). ``yt_dlp`` is imported lazily inside :meth:`decode` (mirrors
    ``live_identity.py``'s ``AcoustIdApiSource.match`` lazy ``httpx`` import)
    so the module stays importable without the dependency installed; not
    unit-tested against a real video — exercised only in the live smoke
    session, same caveat as ``capture.py``'s
    ``WasapiProcessLoopbackCapture``."""

    def __init__(self, *, sample_rate: int = 44100, channels: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def decode(self, youtube_id: str) -> AudioFrame:
        try:
            import yt_dlp
        except ImportError as exc:
            raise YtDlpNotInstalledError(
                "yt-dlp is not installed -- stream-decode capture is "
                "unavailable for this branch. Install it with "
                "`pip install yt-dlp` to enable YouTube stream decoding."
            ) from exc

        ydl_opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise VideoUnavailableError(f"{youtube_id}: {exc}") from exc
        stream_url = info["url"]

        try:
            samples = _decode_stream_to_pcm(
                stream_url, sample_rate=self.sample_rate, channels=self.channels
            )
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
            # code review on PR #196: an ffmpeg decode failure (bad stream,
            # ffmpeg missing) was previously uncaught here and would crash the
            # whole batch instead of journaling this one track "unavailable"
            # like the extractor-failure branch above already does -- route
            # it through the same safe-failure contract (VideoUnavailableError
            # is caught by run_stream_decode_capture and excluded from
            # journaled_unavailable_track_ids re-queue). ValueError added in
            # a follow-up review pass: a googlevideo stream URL that expires
            # or drops mid-transfer can leave ffmpeg exiting 0 with a
            # truncated stdout buffer -- _decode_stream_to_pcm's
            # raw.reshape(-1, channels) raises ValueError in that case, the
            # same batch-crash failure mode as the subprocess errors above.
            raise VideoUnavailableError(f"{youtube_id}: ffmpeg decode failed: {exc}") from exc
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)


def _decode_stream_to_pcm(stream_url: str, *, sample_rate: int, channels: int) -> np.ndarray:
    """``ffmpeg`` subprocess: decode the extracted stream URL to raw
    interleaved float32 PCM, reshaped to :class:`AudioFrame`'s
    ``(n_samples, channels)`` contract. Part of the real, not-unit-tested
    ``YtDlpStreamDecodeSource`` backend (see its docstring)."""
    cmd = [
        "ffmpeg",
        "-i",
        stream_url,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    return raw.reshape(-1, channels)


def decode_and_run_inference(
    source: StreamDecodeSource,
    youtube_id: str,
    *,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
) -> InferenceResult:
    """#170 AC1: the seam between stream-decode and ``run_inference`` — the
    decoded stereo ``AudioFrame``'s samples are passed straight through,
    unreduced. ``run_inference``'s own frontend (``inference._mel_patches``)
    already downmixes to mono internally when it needs to; this orchestration
    must not pre-mix the channels away before that call."""
    frame = source.decode(youtube_id)
    return run_inference(
        frame.samples,
        sample_rate=frame.sample_rate,
        embedding_model=embedding_model,
        classifier=classifier,
    )


@dataclass(frozen=True)
class StreamDecodeCaptureResult:
    """Outcome of one :func:`run_stream_decode_capture` attempt. Distinct from
    ``live_pipeline.LiveCaptureResult`` -- that dataclass requires a resolved
    ``LiveResolvedIdentity`` from a live "now playing" OS signal, which does
    not apply to this queue-driven decode path (#170 AC2)."""

    track_id: str
    inference: InferenceResult | None
    analysis_path: Path | None
    outcome: LiveCaptureOutcome


def _journal_stream_decode(
    journal_path: Path | None,
    *,
    track_id: str,
    outcome: LiveCaptureOutcome,
    reason: str | None,
    now: datetime,
) -> None:
    if journal_path is None:
        return
    timestamp = now.isoformat()
    append_replay_journal_entry(
        journal_path,
        ReplayJournalEntry(
            track_id=track_id,
            outcome=outcome,
            started_at=timestamp,
            ended_at=timestamp,
            reason=reason,
        ),
    )


def run_stream_decode_capture(
    *,
    track_id: str,
    youtube_id: str,
    source: StreamDecodeSource,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    store: UserStore,
    journal_path: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> StreamDecodeCaptureResult:
    """#170 AC2: one journal line per decode attempt.

    Dedup-checks ``store.has_audio_analysis`` first (mirrors #163's pool/root
    exclusion) so an already-analyzed track is skipped without decoding at
    all. A :class:`VideoUnavailableError` from the source -- a removed,
    private, or region-locked video -- is caught and journaled as outcome
    ``"unavailable"`` rather than propagating; the caller sees a result, not
    an exception, so a batch driver can move on to the next queue item.
    ``"unavailable"`` is counted separately from every other outcome by the
    existing generic ``summarize_replay_journal`` (no journal-format change
    needed) and is excluded from re-queueing via
    :func:`journaled_unavailable_track_ids` composed into the
    ``has_audio_analysis`` predicate passed to ``select_replay_queue`` --
    not a new requeue mechanism (decision ``20c86c53``: #166's silent/short
    requeue does not apply here)."""
    if store.has_audio_analysis(track_id):
        _journal_stream_decode(
            journal_path, track_id=track_id, outcome="skipped", reason=None, now=now()
        )
        return StreamDecodeCaptureResult(
            track_id=track_id, inference=None, analysis_path=None, outcome="skipped"
        )

    try:
        inference = decode_and_run_inference(
            source, youtube_id, embedding_model=embedding_model, classifier=classifier
        )
    except VideoUnavailableError as exc:
        _journal_stream_decode(
            journal_path, track_id=track_id, outcome="unavailable", reason=str(exc), now=now()
        )
        return StreamDecodeCaptureResult(
            track_id=track_id, inference=None, analysis_path=None, outcome="unavailable"
        )

    store.write_audio_analysis(
        track_id=track_id, embedding=inference.embedding, tags=inference.tags
    )
    _journal_stream_decode(journal_path, track_id=track_id, outcome="ok", reason=None, now=now())
    return StreamDecodeCaptureResult(
        track_id=track_id,
        inference=inference,
        analysis_path=store.audio_analysis_path(track_id),
        outcome="ok",
    )


def stream_decode_journal_path(store: UserStore) -> Path:
    """Separate from ``replay_capture.replay_journal_path`` (decode-side, not
    loopback-side) so ``summarize_replay_journal``'s ``"requeued"`` handling
    -- meaningless here, since this path never requeues (see
    :func:`process_stream_decode_queue`) -- never mixes with the loopback
    replay journal's own attempts."""
    return store.root / "stream_decode_journal.jsonl"


def process_stream_decode_queue(
    *,
    queue: list[TrackRef],
    source: StreamDecodeSource,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    store: UserStore,
    journal_path: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[StreamDecodeCaptureResult]:
    """#170 AC9: drive ``queue`` end-to-end through
    :func:`run_stream_decode_capture`, one call per track. No requeue logic
    here -- unlike ``replay_capture.process_replay_queue``'s silent/short
    requeue for the loopback path (decision ``20c86c53``: does not apply to
    stream decode). AC2's "not re-queued" semantics for an ``"unavailable"``
    outcome are enforced by excluding that track_id from a *future* queue's
    candidate selection (:func:`journaled_unavailable_track_ids` composed
    into ``has_audio_analysis``), not by an in-loop retry. A track with no
    ``youtube_id`` was never a decode candidate and is skipped without a
    journal entry."""
    results: list[StreamDecodeCaptureResult] = []
    for track in queue:
        if track.youtube_id is None:
            continue
        results.append(
            run_stream_decode_capture(
                track_id=canonical_track_id(track),
                youtube_id=track.youtube_id,
                source=source,
                embedding_model=embedding_model,
                classifier=classifier,
                store=store,
                journal_path=journal_path,
                now=now,
            )
        )
    return results


def journaled_unavailable_track_ids(journal_path: Path) -> set[str]:
    """The set of ``track_id``s journaled with outcome ``"unavailable"``
    (#170 AC2's "not re-queued" clause). A pure read of the JSONL journal --
    intended to be composed into the ``has_audio_analysis`` predicate passed
    to ``select_replay_queue``/``replay_queue_coverage``
    (``replay_queue.py``'s existing injection seam), so an unavailable video
    is not offered again on a later run without any change to that module's
    public API."""
    if not journal_path.exists():
        return set()
    ids: set[str] = set()
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if entry["outcome"] == "unavailable":
            ids.add(entry["track_id"])
    return ids
