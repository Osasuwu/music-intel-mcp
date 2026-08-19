"""End-to-end orchestration test for the live-capture spike (#124).

Wires now-playing -> identity waterfall -> capture -> inference -> local
store, entirely against injected fakes (this codebase's Protocol+fake idiom).
The real WASAPI/SMTC/onnxruntime backends are exercised in the live smoke
session with the user, not here (issue #124 AC1).
"""

from __future__ import annotations

import numpy as np

from music_intel_mcp.capture import FakeLoopbackCapture
from music_intel_mcp.identity import (
    IdentityResolver,
    InMemoryIsrcMbidIndex,
    InMemorySpotifyIsrcSource,
)
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.live_pipeline import run_live_capture_spike
from music_intel_mcp.nowplaying import InMemoryNowPlayingSource, NowPlayingInfo
from music_intel_mcp.store import UserStore


def test_run_live_capture_spike_writes_local_analysis(tmp_path) -> None:
    now_playing = InMemoryNowPlayingSource(
        NowPlayingInfo(
            title="Around the World",
            artist="Daft Punk",
            app_id="Spotify.exe",
            process_id=4242,
            spotify_id="1pKYYY0dkg23sQQXi0Q5zN",
        )
    )
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"FRDM19700001": "mbid-around-the-world"}),
        spotify_source=InMemorySpotifyIsrcSource({"1pKYYY0dkg23sQQXi0Q5zN": "FRDM19700001"}),
    )
    capture = FakeLoopbackCapture(sample_rate=16000, channels=1)
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32))
    classifier = InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9}))
    store = UserStore(root=tmp_path)

    result = run_live_capture_spike(
        duration_s=0.25,
        now_playing_source=now_playing,
        identity_resolver=resolver,
        capture=capture,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
    )

    assert result is not None
    assert result.identity.mbid == "mbid-around-the-world"
    assert result.inference.tags["genre---electronic"] == 0.9
    assert capture.started and capture.stopped
    assert result.analysis_path.exists()
    assert result.analysis_path.is_relative_to(tmp_path)


def test_run_live_capture_spike_none_when_nothing_playing(tmp_path) -> None:
    result = run_live_capture_spike(
        duration_s=0.1,
        now_playing_source=InMemoryNowPlayingSource(None),
        identity_resolver=IdentityResolver(InMemoryIsrcMbidIndex()),
        capture=FakeLoopbackCapture(),
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.0])),
        classifier=InMemoryClassifier(result=ClassifierResult()),
        store=UserStore(root=tmp_path),
    )
    assert result is None
