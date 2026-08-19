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


@dataclass
class ClassifierResult:
    """MTG-Jamendo classifier head output: ``"<category>---<label>"`` -> score."""

    tags: dict[str, float] = field(default_factory=dict)


@dataclass
class InferenceResult:
    embedding: np.ndarray
    tags: dict[str, float]


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
        return Path(explicit)
    env = os.environ.get(env_var)
    if env:
        path = Path(env)
        if not path.exists():
            raise RuntimeError(f"{env_var} points at a missing file: {path}")
        return path
    if default_filename is not None:
        fallback = _SCRATCH_MODELS_DIR / default_filename
        if fallback.exists():
            return fallback
    raise RuntimeError(
        f"{env_var} is not set, no explicit path was given, and no default "
        f"model file was found under {_SCRATCH_MODELS_DIR} — inference cannot "
        "run without the ONNX model file."
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


def _mel_patches(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    mono = pcm.mean(axis=1) if pcm.ndim == 2 else pcm
    mono = mono.astype(np.float32)
    if sample_rate != DISCOGS_EFFNET_SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=DISCOGS_EFFNET_SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=mono,
        sr=DISCOGS_EFFNET_SAMPLE_RATE,
        n_fft=_MEL_FRAME_SIZE,
        hop_length=_MEL_HOP_SIZE,
        n_mels=_MEL_N_BANDS,
        power=1.0,
    )
    log_mel = np.log10(1.0 + _MEL_LOG_SCALE * mel).astype(np.float32).T  # (frames, bands)

    n_frames = log_mel.shape[0]
    if n_frames < _PATCH_FRAMES:
        log_mel = np.pad(log_mel, ((0, _PATCH_FRAMES - n_frames), (0, 0)))
        n_frames = _PATCH_FRAMES
    n_patches = n_frames // _PATCH_FRAMES
    trimmed = log_mel[: n_patches * _PATCH_FRAMES]
    return trimmed.reshape(n_patches, _PATCH_FRAMES, _MEL_N_BANDS)


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

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._model_path))
        return self._session

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        patches = _mel_patches(pcm, sample_rate)
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
    return InferenceResult(embedding=embedding, tags=result.tags)
