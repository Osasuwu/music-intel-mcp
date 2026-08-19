"""Live-capture spike orchestration (#124) — wires the four seams together:

now-playing (AC4) -> identity waterfall (AC4) -> loopback capture (AC1/AC2)
-> inference (AC3) -> local store (AC5).

Every dependency is injected as a Protocol from its own module
(``nowplaying.NowPlayingSource``, ``capture.LoopbackSource``,
``inference.AudioEmbeddingModel``/``ClassifierModel``), so the whole
orchestration is testable against fakes without touching WASAPI, SMTC, or
onnxruntime. The real backends are wired together and exercised only in the
live smoke session with the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .capture import LoopbackSource, RingBufferSink
from .identity import IdentityResolver, ResolvedIdentity
from .inference import AudioEmbeddingModel, ClassifierModel, InferenceResult, run_inference
from .nowplaying import NowPlayingSource, resolve_now_playing
from .store import UserStore


@dataclass
class LiveCaptureResult:
    identity: ResolvedIdentity
    inference: InferenceResult
    analysis_path: Path


def run_live_capture_spike(
    *,
    duration_s: float,
    now_playing_source: NowPlayingSource,
    identity_resolver: IdentityResolver,
    capture: LoopbackSource,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    store: UserStore,
) -> LiveCaptureResult | None:
    """Run one capture-spike pass. ``None`` when nothing is currently playing
    (AC4 — there is no track to identify or capture against)."""
    resolved = resolve_now_playing(now_playing_source, identity_resolver)
    if resolved is None:
        return None
    _now_playing, identity = resolved

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

    inference = run_inference(
        pcm, sample_rate=sink.sample_rate, embedding_model=embedding_model, classifier=classifier
    )

    track_id = identity.mbid or identity.input_key
    analysis_path = store.write_audio_analysis(
        track_id=track_id, embedding=inference.embedding, tags=inference.tags
    )

    return LiveCaptureResult(identity=identity, inference=inference, analysis_path=analysis_path)
