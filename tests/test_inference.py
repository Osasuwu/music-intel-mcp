"""Tests for the embedding/classifier inference seam (#124 AC3)."""

from __future__ import annotations

import numpy as np
import pytest

from music_intel_mcp.inference import (
    ClassifierResult,
    InMemoryClassifier,
    InMemoryEmbeddingModel,
    run_inference,
)


def _pcm(seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    n = int(seconds * sample_rate)
    return np.zeros((n, 1), dtype=np.float32)


# AC3: full pipeline (capture -> librosa -> onnxruntime) produces an embedding
# vector + classifier tags. The librosa/onnxruntime models are exercised for
# real only in the live smoke session; here the orchestration is proven
# end-to-end against injected fakes, matching this codebase's Protocol+fake
# idiom (audio.py AudioFeatureSource, identity.py IsrcMbidIndex).
def test_run_inference_produces_embedding_and_tags() -> None:
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    classifier = InMemoryClassifier(
        result=ClassifierResult(tags={"genre---electronic": 0.9, "mood---energetic": 0.7})
    )

    result = run_inference(
        _pcm(), sample_rate=16000, embedding_model=embedding_model, classifier=classifier
    )

    assert result.embedding.shape == (3,)
    assert result.tags["genre---electronic"] == pytest.approx(0.9)
    assert embedding_model.calls == 1
    assert classifier.calls == 1
