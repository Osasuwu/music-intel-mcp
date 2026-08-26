"""End-to-end orchestration test for the live-capture pipeline (#124, reordered
by #139 AC1).

Wires now-playing -> capture -> chromaprint -> identity waterfall -> inference
-> local store, entirely against injected fakes (this codebase's Protocol+fake
idiom). The real WASAPI/SMTC/fpcalc/onnxruntime backends are exercised in the
live smoke session with the user, not here.

AC1: capture must run *before* identity resolution (chromaprint is computed
from the PCM this process just captured) — a shared ``events`` list records
call order across the fakes to pin that down, not just the end result.
"""

from __future__ import annotations

import numpy as np

from music_intel_mcp.capture import FakeLoopbackCapture
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.live_identity import (
    AcoustIdMatch,
    InMemoryAcoustIdSource,
    LiveIdentityResolver,
)
from music_intel_mcp.live_pipeline import run_live_capture_spike
from music_intel_mcp.nowplaying import InMemoryNowPlayingSource, NowPlayingInfo
from music_intel_mcp.store import UserStore


def _fake_fingerprint_fn(events: list[str]):
    def fn(pcm, sample_rate):
        events.append("fingerprint")
        return "fp-fake", 0.25

    return fn


def test_run_live_capture_spike_captures_before_resolving_identity(tmp_path) -> None:
    """AC1: capture starts immediately and identity resolution happens after,
    using a fingerprint computed from the captured PCM."""
    events: list[str] = []

    class _TrackingCapture(FakeLoopbackCapture):
        def start(self):
            events.append("capture_start")
            super().start()

        def read(self, duration_s):
            events.append("capture_read")
            return super().read(duration_s)

    class _TrackingAcoustId(InMemoryAcoustIdSource):
        def match(self, fingerprint, duration_s):
            events.append("identity_resolve")
            return super().match(fingerprint, duration_s)

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Around the World", artist="Daft Punk", app_id="Spotify.exe")
    )
    acoustid = _TrackingAcoustId({"fp-fake": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    live_resolver = LiveIdentityResolver(acoustid_source=acoustid)
    capture = _TrackingCapture(sample_rate=16000, channels=1)
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32))
    classifier = InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9}))
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=capture,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        fingerprint_fn=_fake_fingerprint_fn(events),
    )

    assert events == ["capture_start", "capture_read", "fingerprint", "identity_resolve"]
    assert result is not None
    assert result.identity.mbid == "M-1"
    assert result.identity.level == "acoustid"
    assert result.inference.tags["genre---electronic"] == 0.9
    assert result.analysis_path.exists()
    assert result.analysis_path.is_relative_to(tmp_path)

    import json

    payload = json.loads(result.analysis_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["raw_title"] == "Around the World"
    assert payload["provenance"]["chromaprint_fingerprint"] == "fp-fake"


def test_run_live_capture_spike_falls_through_when_fingerprinting_fails(tmp_path) -> None:
    """fpcalc missing/erroring must not kill the pipeline — the AcoustID rung
    is simply skipped and the waterfall falls through to the string chain."""

    def _failing_fingerprint_fn(pcm, sample_rate):
        raise RuntimeError("fpcalc not found")

    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(title="Some Song", artist="Some Artist", app_id="Spotify.exe")
    )
    live_resolver = LiveIdentityResolver()  # no sources -> bottoms out at name key
    result = run_live_capture_spike(
        duration_s=0.05,
        now_playing_source=now_playing,
        live_identity_resolver=live_resolver,
        capture=FakeLoopbackCapture(sample_rate=16000, channels=1),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path),
        fingerprint_fn=_failing_fingerprint_fn,
    )

    assert result is not None
    assert result.identity.level == "name"


def test_run_live_capture_spike_none_when_nothing_playing(tmp_path) -> None:
    result = run_live_capture_spike(
        duration_s=0.1,
        now_playing_source=InMemoryNowPlayingSource(None),
        live_identity_resolver=LiveIdentityResolver(),
        capture=FakeLoopbackCapture(),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.0])),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path),
    )
    assert result is None
