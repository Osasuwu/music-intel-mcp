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
from typing import Literal

import numpy as np

from .capture import LoopbackSource, RingBufferSink
from .chromaprint_fpcalc import compute_fingerprint, compute_raw_fingerprint
from .inference import AudioEmbeddingModel, ClassifierModel, InferenceResult, run_inference
from .live_identity import LiveIdentityResolver, LiveResolvedIdentity, ProvenanceSidecar
from .models import TrackRef
from .nowplaying import NowPlayingSource
from .replay_capture import (
    RMS_SILENCE_THRESHOLD,
    ReplayJournalEntry,
    _rms,
    append_replay_journal_entry,
)
from .shared_store import canonical_track_id
from .store import UserStore

FingerprintFn = Callable[[np.ndarray, int], tuple[str, float]]
RawFingerprintFn = Callable[[np.ndarray, int], tuple[list[int], float]]

LiveCaptureOutcome = Literal["ok", "skipped", "silent", "short"]


def live_capture_journal_path(store: UserStore) -> Path:
    return store.root / "live_capture_journal.jsonl"


def _journal_live_discard(
    journal_path: Path | None,
    *,
    track_id: str,
    outcome: LiveCaptureOutcome,
    reason: str,
    now: Callable[[], datetime],
) -> None:
    if journal_path is None:
        return
    ts = now().isoformat()
    append_replay_journal_entry(
        journal_path,
        ReplayJournalEntry(
            track_id=track_id, outcome=outcome, started_at=ts, ended_at=ts, reason=reason
        ),
    )


@dataclass
class LiveCaptureResult:
    identity: LiveResolvedIdentity
    inference: InferenceResult | None
    analysis_path: Path | None
    skipped: bool = False
    outcome: LiveCaptureOutcome = "ok"


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
    raw_fingerprint_fn: RawFingerprintFn = compute_raw_fingerprint,
    journal_path: Path | None = None,
    rms_threshold: float = RMS_SILENCE_THRESHOLD,
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

    # #158 AC1: one function produces the key everywhere — build the same
    # TrackRef shape the backfill/history paths use and derive the key via
    # canonical_track_id, so a live-captured key and a batch-computed key for
    # the same identity are always identical.
    #
    # Exception: a name-only resolution (``identity.name_key`` set) must keep
    # using the live waterfall's *normalized* name key, not canonical_track_id's
    # plain-casefold name rung over the raw OS media-session title —
    # canonical_track_id() doesn't strip feat./"(Official Video)"/remaster
    # noise the way normalize_track_name() does (CONTEXT.md "Normalization
    # (AC4)"), so re-deriving through TrackRef would silently re-fragment
    # cosmetically-different titles of the same recording.
    if identity.name_key is not None:
        track_id = f"name:{identity.name_key}"
    else:
        track_ref = TrackRef(
            spotify_id=identity.spotify_id,
            isrc=identity.isrc,
            mbid=identity.mbid,
            name=identity.name,
            artist=identity.artist,
        )
        track_id = canonical_track_id(track_ref)

    # #179: same RMS/short gate #166 defines for replay, applied to the organic
    # path — a capture below the RMS threshold or shorter than the requested
    # window must not embed and must not write an audio-analysis file (silent,
    # ad, or spoken-intro captures mean-pool to near-identical vectors and are
    # never replaced under first-write-wins). Checked here so the gate fires
    # before the expensive inference/store-write steps below.
    if sink.duration_s + 1e-9 < duration_s:
        reason = f"captured {sink.duration_s:.2f}s < window {duration_s:.2f}s"
        _journal_live_discard(
            journal_path, track_id=track_id, outcome="short", reason=reason, now=now
        )
        return LiveCaptureResult(
            identity=identity, inference=None, analysis_path=None, outcome="short"
        )
    if _rms(pcm) < rms_threshold:
        reason = f"rms below threshold {rms_threshold}"
        _journal_live_discard(
            journal_path, track_id=track_id, outcome="silent", reason=reason, now=now
        )
        return LiveCaptureResult(
            identity=identity, inference=None, analysis_path=None, outcome="silent"
        )

    # #126 AC1/AC4: dedup purely off the identity waterfall + local store — an
    # already-analyzed track is skipped, no re-inference (the expensive step).
    if store.has_audio_analysis(track_id):
        return LiveCaptureResult(
            identity=identity,
            inference=None,
            analysis_path=store.audio_analysis_path(track_id),
            skipped=True,
            outcome="skipped",
        )

    inference = run_inference(
        pcm, sample_rate=sink.sample_rate, embedding_model=embedding_model, classifier=classifier
    )

    # #140 AC1: a second, separate fpcalc call against the same captured PCM
    # ("one temp wav, two calls") -- evidence for offline near-duplicate
    # comparison only, never an identity lookup, so failure here is exactly
    # as non-fatal as the compressed-fingerprint call above.
    try:
        raw_fingerprint, raw_fp_duration_s = raw_fingerprint_fn(pcm, sink.sample_rate)
        store.write_fingerprint(
            track_id=track_id, fingerprint=raw_fingerprint, duration_s=raw_fp_duration_s
        )
    except Exception:
        pass

    provenance = ProvenanceSidecar(
        raw_title=now_playing.title,
        raw_artist=now_playing.artist,
        app_id=now_playing.app_id,
        captured_at=now().isoformat(),
        chromaprint_fingerprint=fingerprint,
    )
    # #126 AC2: first-write-wins — write_audio_analysis atomically claims
    # track_id; a concurrent second writer for the same track discards its
    # write rather than overwriting (embeddings aren't meaningfully averageable).
    analysis_path = store.write_audio_analysis(
        track_id=track_id,
        embedding=inference.embedding,
        tags=inference.tags,
        provenance=provenance,
    )

    return LiveCaptureResult(identity=identity, inference=inference, analysis_path=analysis_path)
