"""Tests for the embedding/classifier inference seam (#124 AC3)."""

from __future__ import annotations

import numpy as np
import pytest

from music_intel_mcp.inference import (
    ClassifierResult,
    InMemoryClassifier,
    InMemoryEmbeddingModel,
    ModelFileNotFoundError,
    RssCeilingExceededError,
    _resolve_model_path,
    check_rss_ceiling,
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


# #160 AC1: an explicit model_path that does not exist must fail loudly and
# name both the missing path and how to configure it — previously this branch
# of _resolve_model_path had no existence check at all (silent fallback risk).
def test_resolve_model_path_raises_when_explicit_path_missing(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.onnx"

    with pytest.raises(ModelFileNotFoundError) as exc_info:
        _resolve_model_path(missing, "SOME_MODEL_PATH_ENV")

    message = str(exc_info.value)
    assert str(missing) in message
    assert "SOME_MODEL_PATH_ENV" in message


def test_resolve_model_path_raises_when_env_path_missing(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "does-not-exist.onnx"
    monkeypatch.setenv("SOME_MODEL_PATH_ENV", str(missing))

    with pytest.raises(ModelFileNotFoundError) as exc_info:
        _resolve_model_path(None, "SOME_MODEL_PATH_ENV")

    message = str(exc_info.value)
    assert str(missing) in message
    assert "SOME_MODEL_PATH_ENV" in message


def test_resolve_model_path_raises_when_unconfigured_and_no_default(monkeypatch) -> None:
    monkeypatch.delenv("SOME_MODEL_PATH_ENV", raising=False)

    with pytest.raises(ModelFileNotFoundError) as exc_info:
        _resolve_model_path(None, "SOME_MODEL_PATH_ENV")

    assert "SOME_MODEL_PATH_ENV" in str(exc_info.value)


# #160 AC2: a configurable RSS ceiling checked after each inference — exceeding
# it must raise a specific, catchable error so the capture loop can stop and
# journal the reason. Testable via an injectable rss_reader (this codebase's
# Protocol+fake DI idiom), not by monkeypatching psutil internals.
def test_check_rss_ceiling_raises_when_reader_exceeds_ceiling() -> None:
    with pytest.raises(RssCeilingExceededError) as exc_info:
        check_rss_ceiling(ceiling_mb=100.0, rss_reader=lambda: 250.0)

    message = str(exc_info.value)
    assert "250" in message
    assert "100" in message


def test_check_rss_ceiling_does_not_raise_when_under_ceiling() -> None:
    check_rss_ceiling(ceiling_mb=100.0, rss_reader=lambda: 50.0)
