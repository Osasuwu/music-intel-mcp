"""Spike B (#123): onnxruntime/librosa engine footprint + throughput benchmark.

Runs the full MTG inference chain locked by decision ``16d7f570`` (Discogs-EffNet
embedding -> MTG-Jamendo classifier head, both consumed as ONNX via
``onnxruntime``, with ``librosa`` as the mel-spectrogram frontend) over a batch
of representative synthetic audio clips, and reports whether the chain is cheap
enough to run continuously alongside real-time playback (#109 AC "Spike B").

No live capture and no copyrighted audio are needed (issue body, explicitly):
clips are synthetic (harmonic tones + noise, duration drawn from a realistic
track-length distribution) so nothing audio-derived is committed to the repo.
This is a footprint/throughput spike, not the ``timbre`` enricher itself — the
mel-spectrogram frontend below approximates Essentia's own
``TensorflowPredictEffnetDiscogs`` preprocessing (same sample rate, mel-band
count and patch shape) closely enough to drive realistic FLOPs/memory through
the real ONNX graphs, but is not claimed bit-exact; exact frontend fidelity is
in scope for the future enricher implementation, not this spike.

Model files are downloaded on first run into ``.scratch/onnx_models/``
(gitignored — binaries are never committed, matching the AcousticBrainz/
MusicBrainz dump convention) and verified against the SHA-256 hashes pinned in
``MODELS`` below, captured from https://essentia.upf.edu/models/ on
2026-08-19 (CC BY-NC-ND 4.0 per that directory's LICENSE file; local-only use
here matches the storage/licensing gate in decision ``29852699``).

Usage::

    python scripts/benchmark_onnx_engine.py --n-clips 50 --out .scratch/onnx_bench_report.json

The report is written to a local, never-committed path; only the summary
(printed to stdout and pasted into the issue) is meant to be shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000  # required by both ONNX graphs (models' "inference.sample_rate")
N_MELS = 96
PATCH_FRAMES = 128
N_FFT = 1024
HOP_LENGTH = 256

# Pinned model provenance (AC4: exact source + version, reproducible). Captured
# from the essentia.upf.edu directory listing + per-model .json metadata on
# 2026-08-19. sha256 values are filled in lazily on first successful download
# (recorded into the report) rather than hand-computed here, since pinning an
# unverified hash defeats the point of verification.
MODELS: dict[str, dict[str, str]] = {
    "embedding": {
        "name": "discogs-effnet-bsdynamic-1",
        "role": "Discogs-EffNet audio embedding (feature extractor)",
        "url": (
            "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
            "discogs-effnet-bsdynamic-1.onnx"
        ),
        "version": "1",
        "release_date": "2022-02-17",
        "framework": "ONNX",
        "framework_version": "1.10.1",
        "license": (
            "CC BY-NC-ND 4.0 (essentia.upf.edu/models/LICENSE) — local-only per decision 29852699"
        ),
        # Actual ONNX graph input name, confirmed via onnxruntime InferenceSession
        # against the downloaded file — the essentia.upf.edu .json metadata
        # sidecar advertises a different (serving-signature) name that doesn't
        # match the graph itself.
        "input_name": "melspectrogram",
        "sample_rate": str(SAMPLE_RATE),
    },
    "classifier_head": {
        "name": "mtg_jamendo_genre-discogs-effnet-1",
        "role": "MTG-Jamendo genre classifier head (consumes Discogs-EffNet embeddings)",
        "url": (
            "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/"
            "mtg_jamendo_genre-discogs-effnet-1.onnx"
        ),
        "version": "1",
        "release_date": "2022-11-20",
        "framework": "ONNX (converted from TensorFlow 2.8.0)",
        "framework_version": "1.10.1",
        "license": (
            "CC BY-NC-ND 4.0 (essentia.upf.edu/models/LICENSE) — local-only per decision 29852699"
        ),
        # See the "melspectrogram" comment above — same discrepancy applies here.
        "input_name": "embeddings",
        "sample_rate": str(SAMPLE_RATE),
    },
}

DEFAULT_CACHE_DIR = Path(".scratch/onnx_models")

# Realistic track-length distribution for synthetic clips (seconds): most pop/
# rock tracks land 2:30-4:30, with a long tail of shorter/longer material.
_DURATION_BUCKETS_S = (90, 150, 180, 210, 240, 270, 300, 360, 480)


# --------------------------------------------------------------------------- #
# Model download + verification (network — not exercised by unit tests)
# --------------------------------------------------------------------------- #


def ensure_model(spec: dict[str, str], cache_dir: Path) -> tuple[Path, str]:
    """Download ``spec['url']`` into ``cache_dir`` if not already present, return
    (local path, sha256 hex digest). Re-downloads are skipped once the file
    exists — this benchmark cares about steady-state inference cost, not
    download time."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / Path(spec["url"]).name
    if not dest.exists():
        with urllib.request.urlopen(spec["url"], timeout=60) as resp:  # noqa: S310 - pinned https URL
            dest.write_bytes(resp.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return dest, digest


# --------------------------------------------------------------------------- #
# Synthetic clip generation (no copyrighted audio; nothing here is committed)
# --------------------------------------------------------------------------- #


def synthetic_clip(duration_s: float, *, seed: int, sample_rate: int = 44_100):
    """A harmonic-tone + noise mixture standing in for a "representative" music
    clip. Real timbral content doesn't matter for a footprint/throughput
    measurement (the ONNX graphs run the same FLOPs regardless of input
    semantics) -- what matters is realistic duration and a non-degenerate
    (non-silent, non-constant) waveform so librosa's mel/STFT path does real
    work rather than short-circuiting on silence."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    fundamental = rng.uniform(110.0, 440.0)
    wave = np.zeros(n_samples, dtype=np.float64)
    for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)):
        wave += amplitude * np.sin(2 * np.pi * fundamental * harmonic * t)
    wave += 0.05 * rng.standard_normal(n_samples)
    wave /= np.max(np.abs(wave)) + 1e-9
    return wave.astype(np.float32), sample_rate


def sample_clip_durations(n_clips: int, *, seed: int = 0) -> list[float]:
    """Draw ``n_clips`` durations from :data:`_DURATION_BUCKETS_S`, deterministic
    per seed so repeat runs are comparable."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [float(rng.choice(_DURATION_BUCKETS_S)) for _ in range(n_clips)]


# --------------------------------------------------------------------------- #
# Frontend: waveform -> mel-spectrogram patches (approximates the Discogs-
# EffNet preprocessing: 16 kHz, 96 mel bands, 128-frame patches)
# --------------------------------------------------------------------------- #


def waveform_to_patches(wave, sample_rate: int):
    """Resample to :data:`SAMPLE_RATE`, compute a log-mel spectrogram, and slice
    it into non-overlapping ``[PATCH_FRAMES, N_MELS]`` patches (zero-padding the
    final partial patch). Returns a ``[n_patches, PATCH_FRAMES, N_MELS]``
    float32 array matching the embedding model's declared input shape."""
    import librosa
    import numpy as np

    if sample_rate != SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=wave, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel).T.astype(np.float32)  # [n_frames, N_MELS]

    n_frames = log_mel.shape[0]
    n_patches = max(1, -(-n_frames // PATCH_FRAMES))  # ceil div
    padded_len = n_patches * PATCH_FRAMES
    if padded_len > n_frames:
        pad = np.zeros((padded_len - n_frames, N_MELS), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=0)

    return log_mel.reshape(n_patches, PATCH_FRAMES, N_MELS)


# --------------------------------------------------------------------------- #
# Inference chain
# --------------------------------------------------------------------------- #


def build_sessions(embedding_path: Path, classifier_path: Path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-track-at-a-time is the real usage pattern
    embedding_session = ort.InferenceSession(
        str(embedding_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    classifier_session = ort.InferenceSession(
        str(classifier_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return embedding_session, classifier_session


def run_chain(embedding_session, classifier_session, patches) -> Any:
    """Embedding model -> mean-pool over patches -> classifier head. Returns the
    classifier's sigmoid predictions (unused beyond proving the chain runs
    end-to-end; this spike measures cost, not accuracy)."""
    import numpy as np

    embedding_input = MODELS["embedding"]["input_name"]
    embedding_output_name = embedding_session.get_outputs()[-1].name  # "embeddings" output
    (embeddings,) = embedding_session.run([embedding_output_name], {embedding_input: patches})
    pooled = embeddings.mean(axis=0, keepdims=True).astype(np.float32)  # [1, 1280]

    classifier_input = MODELS["classifier_head"]["input_name"]
    (predictions,) = classifier_session.run(None, {classifier_input: pooled})
    return predictions


# --------------------------------------------------------------------------- #
# Memory sampling (cross-platform peak RSS -- Windows has no `resource` module)
# --------------------------------------------------------------------------- #


class PeakRssSampler:
    """Background-thread sampler of the current process's RSS, polled at a fixed
    interval. ``psutil`` works identically on Windows/Linux/macOS, unlike the
    stdlib ``resource`` module (POSIX-only, unusable on this Windows-primary
    product)."""

    def __init__(self, interval_s: float = 0.05):
        self._interval_s = interval_s
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        import psutil

        process = psutil.Process()
        while not self._stop.is_set():
            rss = process.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            self._stop.wait(self._interval_s)

    def __enter__(self) -> PeakRssSampler:
        import psutil

        self._peak_bytes = psutil.Process().memory_info().rss
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 4)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Report assembly (pure logic -- unit-tested without real models/audio)
# --------------------------------------------------------------------------- #


@dataclass
class ClipResult:
    duration_s: float
    latency_s: float


@dataclass
class BenchmarkReport:
    n_clips: int
    p50_latency_s: float
    p95_latency_s: float
    mean_latency_s: float
    peak_rss_mb: float
    realtime_ratio_p95: float  # p95 latency / shortest sampled clip duration
    sustainable: bool
    models: dict[str, dict[str, str]] = field(default_factory=dict)
    clips: list[ClipResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_clips": self.n_clips,
            "p50_latency_s": self.p50_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "mean_latency_s": self.mean_latency_s,
            "peak_rss_mb": self.peak_rss_mb,
            "realtime_ratio_p95": self.realtime_ratio_p95,
            "sustainable": self.sustainable,
            "models": self.models,
            "clips": [{"duration_s": c.duration_s, "latency_s": c.latency_s} for c in self.clips],
        }


# Thresholds for the "sustainable alongside real-time playback" verdict (AC3).
# Real-time headroom: the whole chain for one clip must finish in a small
# fraction of that clip's own duration (playback keeps running the whole time
# regardless, so there's no hard deadline -- but a ratio near or above 1.0
# means analysis falls behind the listening pace on a live/backfill queue).
# Memory: kept well under what's realistically "spare" during normal desktop
# use (browser + player + background apps) on typical consumer hardware.
REALTIME_RATIO_CEILING = 0.5
PEAK_RSS_CEILING_MB = 1500.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def build_report(
    clips: list[ClipResult], peak_rss_mb: float, models: dict[str, dict[str, str]]
) -> BenchmarkReport:
    latencies = [c.latency_s for c in clips]
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    shortest_duration = min(c.duration_s for c in clips)
    realtime_ratio_p95 = p95 / shortest_duration
    sustainable = (
        realtime_ratio_p95 <= REALTIME_RATIO_CEILING and peak_rss_mb <= PEAK_RSS_CEILING_MB
    )
    return BenchmarkReport(
        n_clips=len(clips),
        p50_latency_s=p50,
        p95_latency_s=p95,
        mean_latency_s=statistics.mean(latencies),
        peak_rss_mb=peak_rss_mb,
        realtime_ratio_p95=realtime_ratio_p95,
        sustainable=sustainable,
        models=models,
        clips=clips,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def run_benchmark(n_clips: int, cache_dir: Path, seed: int = 0) -> BenchmarkReport:
    embedding_path, embedding_sha256 = ensure_model(MODELS["embedding"], cache_dir)
    classifier_path, classifier_sha256 = ensure_model(MODELS["classifier_head"], cache_dir)
    embedding_session, classifier_session = build_sessions(embedding_path, classifier_path)

    models = {
        key: {**spec, "sha256": digest}
        for key, spec, digest in (
            ("embedding", MODELS["embedding"], embedding_sha256),
            ("classifier_head", MODELS["classifier_head"], classifier_sha256),
        )
    }

    durations = sample_clip_durations(n_clips, seed=seed)
    clips: list[ClipResult] = []
    with PeakRssSampler() as sampler:
        for i, duration_s in enumerate(durations):
            wave, sample_rate = synthetic_clip(duration_s, seed=seed * 1000 + i)
            start = time.perf_counter()
            patches = waveform_to_patches(wave, sample_rate)
            run_chain(embedding_session, classifier_session, patches)
            latency_s = time.perf_counter() - start
            clips.append(ClipResult(duration_s=duration_s, latency_s=latency_s))
        peak_rss_mb = sampler.peak_mb

    return build_report(clips, peak_rss_mb, models)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-clips", type=int, default=50, help="number of synthetic clips (AC1: >=50)"
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE_DIR), help="local ONNX model cache (gitignored)"
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic synthetic-clip seed")
    parser.add_argument(
        "--out", required=True, help="output report path (local file, never committed)"
    )
    args = parser.parse_args(argv)

    if args.n_clips < 50:
        print(f"AC1 requires >=50 clips; got --n-clips={args.n_clips}")
        return 1

    report = run_benchmark(args.n_clips, Path(args.cache_dir), seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    verdict = "SUSTAINABLE" if report.sustainable else "NOT SUSTAINABLE"
    print(f"wrote benchmark report to {out_path}")
    print(
        f"  n_clips={report.n_clips} p50={report.p50_latency_s:.3f}s "
        f"p95={report.p95_latency_s:.3f}s mean={report.mean_latency_s:.3f}s "
        f"peak_rss={report.peak_rss_mb:.1f}MB realtime_ratio_p95={report.realtime_ratio_p95:.4f}"
    )
    print(
        f"  verdict: {verdict} (real-time ratio ceiling={REALTIME_RATIO_CEILING}, "
        f"peak RSS ceiling={PEAK_RSS_CEILING_MB}MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
