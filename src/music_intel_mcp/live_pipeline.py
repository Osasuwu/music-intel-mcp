"""Live-capture pipeline (#124, reordered by #139 AC1) — wires the seams:

loopback capture (AC1/AC2) -> chromaprint (AC1/AC7) -> live identity waterfall
(AC2/AC3/AC4/AC6) -> inference (AC3) -> local store + provenance sidecar (AC5).

Capture runs *first*, immediately on track-change; identity resolution
(including the chromaprint fingerprint) happens only after capture completes
— the OS media session's title/artist is enough to start recording, but the
audio itself is needed before AcoustID can be tried. This is the opposite
order of the original #124 spike, which resolved identity via the batch
waterfall before ever starting capture.

Every dependency is injected as a Protocol from its own module
(``nowplaying.NowPlayingSource``, ``capture.LoopbackSource``,
``inference.AudioEmbeddingModel``/``ClassifierModel``,
``live_identity.LiveIdentityResolver``), so the whole orchestration is
testable against fakes without touching WASAPI, SMTC, fpcalc, or
onnxruntime. The real backends are wired together and exercised only in the
live smoke session with the user.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .capture import LoopbackSource, RingBufferSink
from .chromaprint_fpcalc import compute_fingerprint
from .inference import AudioEmbeddingModel, ClassifierModel, InferenceResult, run_inference
from .live_identity import LiveIdentityResolver, LiveResolvedIdentity, ProvenanceSidecar
from .nowplaying import NowPlayingSource
from .store import UserStore

FingerprintFn = Callable[[np.ndarray, int], tuple[str, float]]


@dataclass
class LiveCaptureResult:
    identity: LiveResolvedIdentity
    inference: InferenceResult
    analysis_path: Path


def run_live_capture_spike(
    *,
    duration_s: float,
    now_playing_source: NowPlayingSource,
    live_identity_resolver: LiveIdentityResolver,
    capture: LoopbackSource,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    store: UserStore,
    fingerprint_fn: FingerprintFn = compute_fingerprint,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> LiveCaptureResult | None:
    """Run one capture pass. ``None`` when nothing is currently playing (AC4 —
    there is no track to capture or identify against).

    Fingerprinting failure (``fpcalc`` missing, subprocess error, ...) is not
    fatal — the AcoustID rung is simply skipped and the waterfall falls
    through to the string chain, per this project's fragmentation-over-
    false-merge bias (AC3)."""
    now_playing = now_playing_source.current()
    if now_playing is None:
        return None

    capture.start()
    try:
        frame = capture.read(duration_s)
    finally:
        capture.stop()

    sink = RingBufferSink(
        max_seconds=duration_s, sample_rate=frame.sample_rate, channels=frame.samples.shape[1]
    )
    sink.write(frame)
    pcm = sink.read_all()

    fingerprint: str | None = None
    fp_duration_s = sink.duration_s
    try:
        fingerprint, fp_duration_s = fingerprint_fn(pcm, sink.sample_rate)
    except Exception:
        fingerprint = None

    identity = live_identity_resolver.resolve(
        title=now_playing.title,
        artist=now_playing.artist,
        fingerprint=fingerprint,
        duration_s=fp_duration_s,
    )

    inference = run_inference(
        pcm, sample_rate=sink.sample_rate, embedding_model=embedding_model, classifier=classifier
    )

    track_id = identity.mbid or identity.spotify_id or identity.isrc or identity.name_key
    provenance = ProvenanceSidecar(
        raw_title=now_playing.title,
        raw_artist=now_playing.artist,
        app_id=now_playing.app_id,
        captured_at=now().isoformat(),
        chromaprint_fingerprint=fingerprint,
    )
    analysis_path = store.write_audio_analysis(
        track_id=track_id,
        embedding=inference.embedding,
        tags=inference.tags,
        provenance=provenance,
    )

    return LiveCaptureResult(identity=identity, inference=inference, analysis_path=analysis_path)
