"""Unit tests for the onnxruntime/librosa engine benchmark (#123).

Synthetic data only, no network, no real ONNX model download/inference —
those are exercised once manually as the E2E smoke run (see PR body), not by
this suite. Covers the pure-Python pieces: percentile math, deterministic
clip-duration sampling, the sustainability verdict, and the mel-spectrogram
patch shape (librosa itself is a local computation, no network involved, but
lives in the optional ``onnx-bench`` extra so its test is skipped when that
extra isn't installed).
"""

from __future__ import annotations

import pytest
from benchmark_onnx_engine import (
    N_MELS,
    PATCH_FRAMES,
    PEAK_RSS_CEILING_MB,
    REALTIME_RATIO_CEILING,
    ClipResult,
    build_report,
    percentile,
    sample_clip_durations,
    synthetic_clip,
)

# --------------------------------------------------------------------------- #
# percentile
# --------------------------------------------------------------------------- #


def test_percentile_median_odd_count():
    assert percentile([1.0, 3.0, 2.0], 50) == 2.0


def test_percentile_p95_matches_known_interpolation():
    values = [float(i) for i in range(1, 101)]  # 1..100
    # k = (100-1) * 0.95 = 94.05 -> interpolate between index 94 (95.0) and 95 (96.0)
    assert percentile(values, 95) == pytest.approx(95.05)


def test_percentile_single_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


# --------------------------------------------------------------------------- #
# sample_clip_durations
# --------------------------------------------------------------------------- #


def test_sample_clip_durations_deterministic_per_seed():
    a = sample_clip_durations(50, seed=7)
    b = sample_clip_durations(50, seed=7)
    assert a == b


def test_sample_clip_durations_varies_by_seed():
    a = sample_clip_durations(50, seed=1)
    b = sample_clip_durations(50, seed=2)
    assert a != b


def test_sample_clip_durations_count_and_positivity():
    durations = sample_clip_durations(50, seed=0)
    assert len(durations) == 50
    assert all(d > 0 for d in durations)


# --------------------------------------------------------------------------- #
# synthetic_clip
# --------------------------------------------------------------------------- #


def test_synthetic_clip_shape_and_bounds():
    np = pytest.importorskip("numpy")
    wave, sample_rate = synthetic_clip(2.0, seed=0, sample_rate=44_100)
    assert sample_rate == 44_100
    assert wave.shape == (int(2.0 * 44_100),)
    assert wave.dtype == np.float32
    assert float(np.max(np.abs(wave))) <= 1.0 + 1e-6


def test_synthetic_clip_deterministic_per_seed():
    wave_a, _ = synthetic_clip(1.0, seed=3)
    wave_b, _ = synthetic_clip(1.0, seed=3)
    assert (wave_a == wave_b).all()


def test_synthetic_clip_not_silent():
    np = pytest.importorskip("numpy")
    wave, _ = synthetic_clip(1.0, seed=0)
    assert np.abs(wave).mean() > 0.01


# --------------------------------------------------------------------------- #
# waveform_to_patches (librosa lives in the optional onnx-bench extra)
# --------------------------------------------------------------------------- #


def test_waveform_to_patches_shape():
    pytest.importorskip("librosa")
    from benchmark_onnx_engine import SAMPLE_RATE, waveform_to_patches

    wave, sample_rate = synthetic_clip(3.0, seed=0, sample_rate=SAMPLE_RATE)
    patches = waveform_to_patches(wave, sample_rate)
    assert patches.ndim == 3
    assert patches.shape[1] == PATCH_FRAMES
    assert patches.shape[2] == N_MELS
    assert patches.shape[0] >= 1


def test_waveform_to_patches_resamples_when_needed():
    pytest.importorskip("librosa")
    from benchmark_onnx_engine import waveform_to_patches

    wave, sample_rate = synthetic_clip(1.0, seed=0, sample_rate=44_100)
    patches = waveform_to_patches(wave, sample_rate)
    assert patches.shape[1:] == (PATCH_FRAMES, N_MELS)


# --------------------------------------------------------------------------- #
# build_report / sustainability verdict
# --------------------------------------------------------------------------- #


def _clips(latencies_and_durations: list[tuple[float, float]]) -> list[ClipResult]:
    return [ClipResult(duration_s=d, latency_s=lat) for lat, d in latencies_and_durations]


def test_build_report_sustainable_when_fast_and_light():
    clips = _clips([(0.05, 120.0), (0.06, 90.0), (0.07, 180.0)])
    report = build_report(clips, peak_rss_mb=500.0, models={})
    assert report.sustainable is True
    assert report.n_clips == 3
    assert report.peak_rss_mb == 500.0


def test_build_report_not_sustainable_when_latency_exceeds_ceiling():
    # shortest clip is 90s; p95 latency needs to exceed ratio ceiling * 90s
    slow_latency = REALTIME_RATIO_CEILING * 90.0 + 1.0
    clips = _clips([(slow_latency, 90.0), (slow_latency, 90.0), (slow_latency, 90.0)])
    report = build_report(clips, peak_rss_mb=500.0, models={})
    assert report.sustainable is False


def test_build_report_not_sustainable_when_memory_exceeds_ceiling():
    clips = _clips([(0.05, 120.0), (0.06, 90.0)])
    report = build_report(clips, peak_rss_mb=PEAK_RSS_CEILING_MB + 1.0, models={})
    assert report.sustainable is False


def test_build_report_computes_p50_p95_mean():
    clips = _clips([(0.10, 100.0), (0.20, 100.0), (0.30, 100.0), (0.40, 100.0)])
    report = build_report(clips, peak_rss_mb=100.0, models={})
    assert report.p50_latency_s == pytest.approx(percentile([0.10, 0.20, 0.30, 0.40], 50))
    assert report.p95_latency_s == pytest.approx(percentile([0.10, 0.20, 0.30, 0.40], 95))
    assert report.mean_latency_s == pytest.approx(0.25)


def test_report_to_dict_round_trips_fields():
    clips = _clips([(0.05, 120.0), (0.06, 90.0)])
    report = build_report(clips, peak_rss_mb=321.0, models={"embedding": {"url": "x"}})
    data = report.to_dict()
    assert data["n_clips"] == 2
    assert data["peak_rss_mb"] == 321.0
    assert data["models"] == {"embedding": {"url": "x"}}
    assert len(data["clips"]) == 2
    assert data["clips"][0] == {"duration_s": 120.0, "latency_s": 0.05}
