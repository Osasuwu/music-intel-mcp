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
    _mel_patches,
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


def _tone(seconds: float = 1.0, sample_rate: int = 16000, gain: float = 1.0) -> np.ndarray:
    # float64 throughout the trig, casting to float32 only at the very end (as
    # a real captured recording would already be quantized) -- computing sin()
    # on a float32 phase argument bakes gain-dependent rounding into the test
    # fixture itself, before _mel_patches ever sees the signal.
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    return (gain * 0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


# #194 AC: _mel_patches output must be gain-invariant -- the same recording at
# different playback volumes must produce the same model input. This fails on
# main (25a9852) because log10(1 + _MEL_LOG_SCALE * mel) has no amplitude
# normalization: the `1 +` offset is non-homogeneous under scaling.
def test_mel_patches_is_gain_invariant() -> None:
    base, base_rms = _mel_patches(_tone(gain=1.0), 16000)
    quiet, quiet_rms = _mel_patches(_tone(gain=0.1), 16000)
    loud, loud_rms = _mel_patches(_tone(gain=4.0), 16000)

    np.testing.assert_allclose(quiet, base, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(loud, base, rtol=1e-4, atol=1e-5)
    # the pre-normalization RMS scalar itself must still reflect the real gain
    assert quiet_rms == pytest.approx(base_rms * 0.1, rel=1e-3)
    assert loud_rms == pytest.approx(base_rms * 4.0, rel=1e-3)


# #194 AC: WASAPI loopback quantizes to int16 (pcm = int16 / 32768.0) *before*
# normalization runs, so playback volume also scales a fixed quantization/
# dither floor. This residual is invisible to the pure-float rescale test
# above by construction (that test never quantizes), so it needs its own
# bounded-divergence assertion.
def test_mel_patches_int16_quantized_gain_divergence_within_bound() -> None:
    def quantized(gain: float) -> np.ndarray:
        pcm = _tone(gain=gain)
        int16 = np.clip(pcm * 32768.0, -32768, 32767).astype(np.int16)
        return (int16.astype(np.float32) / 32768.0).astype(np.float32)

    base, _ = _mel_patches(quantized(1.0), 16000)
    quiet, _ = _mel_patches(quantized(0.1), 16000)
    loud, _ = _mel_patches(quantized(4.0), 16000)

    assert np.isfinite(base).all()
    # Mean, not max: at -20dB playback (gain=0.1) the quantization floor is a
    # much larger fraction of the signal, so normalizing back up amplifies
    # that fixed floor ~10x and a handful of near-silent log-mel bins spike
    # well above any tight per-element bound -- that's the expected residual
    # this test exists to characterize, not a bug. The mean divergence across
    # the whole patch stays small and is the meaningful bound here.
    assert np.mean(np.abs(quiet - base)) < 0.1
    assert np.mean(np.abs(loud - base)) < 0.1


# #194 AC: degenerate rms(mono) < 1e-6 must pass through unchanged (no gain
# cap, no divide-by-near-zero blowup) -- still finite either way.
def test_mel_patches_all_zero_frame_is_finite() -> None:
    silence = np.zeros(16000, dtype=np.float32)
    out, rms = _mel_patches(silence, 16000)
    assert np.isfinite(out).all()
    assert rms == pytest.approx(0.0)


def test_mel_patches_clipped_square_wave_is_finite() -> None:
    t = np.arange(16000, dtype=np.float32)
    square = np.sign(np.sin(2 * np.pi * 440.0 * t / 16000.0)).astype(np.float32)
    out, _ = _mel_patches(square, 16000)
    assert np.isfinite(out).all()
