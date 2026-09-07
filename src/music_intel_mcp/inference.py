"""librosa + onnxruntime inference for the live-capture spike (#124 AC3).

Engine decision (16d7f570): onnxruntime + librosa, not Essentia (no Windows
wheels). Two ONNX models chain together: Discogs-EffNet produces the track
embedding, MTG-Jamendo classifier heads consume that embedding to produce
genre/mood/instrument tags. Model paths are env-pointed
(``DISCOGS_EFFNET_MODEL_PATH`` / ``MTG_JAMENDO_MODEL_PATH``), mirroring the
``AcousticBrainzDump`` env-pointed-external-resource idiom in ``audio.py`` —
except unlike that dump's honest-empty-on-missing, a missing model here raises
immediately: inference cannot proceed without it, there is no partial result.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

_DISCOGS_EFFNET_MODEL_PATH_ENV = "DISCOGS_EFFNET_MODEL_PATH"
_MTG_JAMENDO_MODEL_PATH_ENV = "MTG_JAMENDO_MODEL_PATH"

# .scratch/ is the repo's existing convention for gitignored, env-pointed
# external artifacts (see CLAUDE.md — AcousticBrainz/MusicBrainz dumps).
# When the env vars above aren't set, fall back to the model files here
# (as downloaded from essentia.upf.edu) so `capture-spike` works without
# per-session env exports.
_SCRATCH_MODELS_DIR = Path(__file__).resolve().parents[2] / ".scratch" / "models"
_DISCOGS_EFFNET_DEFAULT_FILENAME = "discogs-effnet-bsdynamic-1.onnx"
_MTG_JAMENDO_DEFAULT_FILENAME = "mtg_jamendo_top50tags-discogs-effnet-1.onnx"

DISCOGS_EFFNET_SAMPLE_RATE = 16000


class ModelFileNotFoundError(RuntimeError):
    """Raised when a required ONNX model file cannot be found on disk. Inference
    must refuse to start rather than silently falling back (#160 AC1)."""


class RssCeilingExceededError(RuntimeError):
    """Raised when process RSS exceeds the configured ceiling after an
    inference pass (#160 AC2) — the caller (the live capture loop) must stop
    rather than continue."""


# Shared with scripts/benchmark_onnx_engine.py (#160 AC3) — one ceiling, not
# two independently-maintained copies.
PEAK_RSS_CEILING_MB = 1500.0


def _default_rss_reader() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def check_rss_ceiling(
    *,
    ceiling_mb: float = PEAK_RSS_CEILING_MB,
    rss_reader: Callable[[], float] | None = None,
) -> None:
    """Raise :class:`RssCeilingExceededError` if current RSS exceeds
    ``ceiling_mb``. ``rss_reader`` is injectable so callers/tests never need
    to monkeypatch ``psutil`` internals (#160 AC2)."""
    reader = rss_reader if rss_reader is not None else _default_rss_reader
    rss_mb = reader()
    if rss_mb > ceiling_mb:
        raise RssCeilingExceededError(
            f"RSS {rss_mb:.1f} MB exceeded ceiling {ceiling_mb:.1f} MB after "
            "inference — stopping the capture loop."
        )


@dataclass
class ClassifierResult:
    """MTG-Jamendo classifier head output: ``"<category>---<label>"`` -> score."""

    tags: dict[str, float] = field(default_factory=dict)


# #194 AC6: bumped whenever the front-end parametrization changes the model
# input space (not just the ONNX filename) -- gain normalization added here
# means every embedding computed before this version is not comparable to one
# computed after it. Consumers (near_dup.py, the store) key on this string to
# decide when a record's embedding must be treated as stale.
EMBEDDING_SPACE_VERSION = "discogs-effnet-bsdynamic-1+rms-norm-v1"


@dataclass
class InferenceResult:
    embedding: np.ndarray
    tags: dict[str, float]
    input_rms: float | None = None


@runtime_checkable
class AudioEmbeddingModel(Protocol):
    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray: ...


@runtime_checkable
class ClassifierModel(Protocol):
    def classify(self, embedding: np.ndarray) -> ClassifierResult: ...


class InMemoryEmbeddingModel:
    """Fixed-vector :class:`AudioEmbeddingModel` for tests."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector
        self.calls = 0
        self.last_input_rms: float | None = None

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        return self._vector


class InMemoryClassifier:
    """Fixed-result :class:`ClassifierModel` for tests."""

    def __init__(self, result: ClassifierResult) -> None:
        self._result = result
        self.calls = 0

    def classify(self, embedding: np.ndarray) -> ClassifierResult:
        self.calls += 1
        return self._result


def _load_labels(model_path: Path) -> list[str] | None:
    """Essentia model downloads ship a sibling ``<name>.json`` metadata file
    with a ``classes`` label list (matching the ``.onnx`` filename stem)."""
    import json

    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta.get("classes")
    return list(classes) if classes else None


def _resolve_model_path(
    explicit: str | Path | None, env_var: str, *, default_filename: str | None = None
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise ModelFileNotFoundError(
                f"Model file not found: {path} — pass an existing path explicitly, "
                f"or unset it and configure {env_var} instead."
            )
        return path
    env = os.environ.get(env_var)
    if env:
        path = Path(env)
        if not path.exists():
            raise ModelFileNotFoundError(
                f"{env_var} points at a missing file: {path} — fix the env var to "
                "point at an existing ONNX model file."
            )
        return path
    if default_filename is not None:
        fallback = _SCRATCH_MODELS_DIR / default_filename
        if fallback.exists():
            return fallback
    raise ModelFileNotFoundError(
        f"{env_var} is not set, no explicit path was given, and no default "
        f"model file was found under {_SCRATCH_MODELS_DIR} — inference cannot "
        f"run without the ONNX model file. Configure {env_var} to point at it."
    )


# Mel frontend + patching, matching essentia's TensorflowInputMusiCNN /
# TensorflowPredictEffnetDiscogs preprocessing (sampleRate=16000, frameSize=512,
# hopSize=256, numberBands=96, log10(1 + 10000*x) compression), fed to the
# model as non-overlapping 128-frame patches, mean-pooled into one track-level
# embedding. Best-effort match to essentia's own DSP (not bit-exact — the
# model repo does not publish the reference preprocessing as runnable code);
# adequate for this spike, not for score-level fidelity claims.
_MEL_FRAME_SIZE = 512
_MEL_HOP_SIZE = 256
_MEL_N_BANDS = 96
_MEL_LOG_SCALE = 10000.0
_PATCH_FRAMES = 128

# #194: gain-invariance target and degenerate-signal floor. RMS (not peak, not
# LUFS) to a fixed -20 dBFS -- see CONTEXT.md "Gain-invariant mel front-end"
# (decision f0f66484) for why the alternatives were rejected.
_RMS_NORMALIZATION_TARGET = 0.1
_RMS_DEGENERATE_FLOOR = 1e-6


def _normalize_rms(mono: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale ``mono`` to :data:`_RMS_NORMALIZATION_TARGET` RMS. Returns the
    normalized signal and the *pre-normalization* RMS (the scalar recorded
    alongside the analysis, per #194's applied-gain AC). Below
    ``_RMS_DEGENERATE_FLOOR`` the signal is returned unchanged -- no gain cap,
    no clipping special-case (decision 271c9acf): a cap would reintroduce the
    level-dependence this normalizer exists to remove."""
    rms = float(np.sqrt(np.mean(np.square(mono))))
    if rms < _RMS_DEGENERATE_FLOOR:
        return mono, rms
    return mono * (_RMS_NORMALIZATION_TARGET / rms), rms


def _mel_patches(pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    import librosa

    # #194 AC3: the mel/log pipeline runs in float64 end-to-end, casting down
    # to float32 only on the final patch array (the ONNX model's input dtype).
    # A float32 signal carries ~1e-7 relative rounding noise whose *absolute*
    # size scales with the signal's magnitude -- since the three gain levels
    # this feeds into differ in magnitude before normalization divides it back
    # out, that noise doesn't cancel and shows up as a gain-dependent residual
    # after the log/mel nonlinearity. float64 pushes it below the noise floor
    # of the tolerance this front-end is contracted to (rtol=1e-4, atol=1e-5).
    mono = pcm.mean(axis=1) if pcm.ndim == 2 else pcm
    mono = mono.astype(np.float64)
    if sample_rate != DISCOGS_EFFNET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=DISCOGS_EFFNET_SAMPLE_RATE)
    mono, input_rms = _normalize_rms(mono)

    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=DISCOGS_EFFNET_SAMPLE_RATE,
        n_fft=_MEL_FRAME_SIZE,
        hop_length=_MEL_HOP_SIZE,
        n_mels=_MEL_N_BANDS,
        power=1.0,
    )
    log_mel = np.log10(1.0 + _MEL_LOG_SCALE * mel).T  # (frames, bands), float64

    n_frames = log_mel.shape[0]
    if n_frames < _PATCH_FRAMES:
        log_mel = np.pad(log_mel, ((0, _PATCH_FRAMES - n_frames), (0, 0)))
        n_frames = _PATCH_FRAMES
    n_patches = n_frames // _PATCH_FRAMES
    trimmed = log_mel[: n_patches * _PATCH_FRAMES]
    patches = trimmed.reshape(n_patches, _PATCH_FRAMES, _MEL_N_BANDS).astype(np.float32)
    return patches, input_rms


class DiscogsEffnetOnnxModel:
    """Real :class:`AudioEmbeddingModel` — librosa mel-spectrogram frontend
    feeding the Discogs-EffNet ONNX embedding model. Exercised for real only
    in the live smoke session (issue #124 AC1/AC3); not unit-tested since it
    needs the actual (large, licensed) model file on disk."""

    def __init__(self, *, model_path: str | Path | None = None) -> None:
        self._model_path = _resolve_model_path(
            model_path,
            _DISCOGS_EFFNET_MODEL_PATH_ENV,
            default_filename=_DISCOGS_EFFNET_DEFAULT_FILENAME,
        )
        self._session = None
        self.last_input_rms: float | None = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._model_path))
        return self._session

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        patches, input_rms = _mel_patches(pcm, sample_rate)
        # #194 AC9: the pre-normalization RMS is threaded to run_inference via
        # this attribute rather than changing embed()'s Protocol-declared
        # return type -- InMemoryEmbeddingModel mirrors this attribute so both
        # fakes and the real model satisfy AudioEmbeddingModel unchanged.
        self.last_input_rms = input_rms
        outputs = session.run(["embeddings"], {input_name: patches})
        # One embedding per 128-frame patch -> mean-pool into a track-level vector.
        return np.asarray(outputs[0]).mean(axis=0)


class MtgJamendoClassifier:
    """Real :class:`ClassifierModel` — MTG-Jamendo classifier head over a
    Discogs-EffNet embedding. Same real/live-session-only caveat as
    :class:`DiscogsEffnetOnnxModel`."""

    def __init__(
        self, *, model_path: str | Path | None = None, labels: list[str] | None = None
    ) -> None:
        self._model_path = _resolve_model_path(
            model_path, _MTG_JAMENDO_MODEL_PATH_ENV, default_filename=_MTG_JAMENDO_DEFAULT_FILENAME
        )
        self._labels = labels or _load_labels(self._model_path)
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._model_path))
        return self._session

    def classify(self, embedding: np.ndarray) -> ClassifierResult:
        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(
            ["activations"], {input_name: embedding[np.newaxis, ...].astype(np.float32)}
        )
        scores = np.asarray(outputs[0]).reshape(-1)
        labels = self._labels or [f"label_{i}" for i in range(len(scores))]
        return ClassifierResult(tags=dict(zip(labels, scores.tolist(), strict=False)))


def run_inference(
    pcm: np.ndarray,
    *,
    sample_rate: int,
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
) -> InferenceResult:
    """Wire capture -> embedding -> classifier tags (#124 AC3)."""
    embedding = embedding_model.embed(pcm, sample_rate)
    result = classifier.classify(embedding)
    input_rms = getattr(embedding_model, "last_input_rms", None)
    return InferenceResult(embedding=embedding, tags=result.tags, input_rms=input_rms)
