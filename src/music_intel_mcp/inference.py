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


def _resolve_model_path(explicit: str | Path | None, env_var: str) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(env_var)
    if not env:
        raise RuntimeError(
            f"{env_var} is not set and no explicit path was given — inference "
            "cannot run without the ONNX model file."
        )
    path = Path(env)
    if not path.exists():
        raise RuntimeError(f"{env_var} points at a missing file: {path}")
    return path


class DiscogsEffnetOnnxModel:
    """Real :class:`AudioEmbeddingModel` — librosa mel-spectrogram frontend
    feeding the Discogs-EffNet ONNX embedding model. Exercised for real only
    in the live smoke session (issue #124 AC1/AC3); not unit-tested since it
    needs the actual (large, licensed) model file on disk."""

    def __init__(self, *, model_path: str | Path | None = None) -> None:
        self._model_path = _resolve_model_path(model_path, _DISCOGS_EFFNET_MODEL_PATH_ENV)
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._model_path))
        return self._session

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        import librosa

        mono = pcm.mean(axis=1) if pcm.ndim == 2 else pcm
        if sample_rate != DISCOGS_EFFNET_SAMPLE_RATE:
            mono = librosa.resample(
                mono.astype(np.float32), orig_sr=sample_rate, target_sr=DISCOGS_EFFNET_SAMPLE_RATE
            )
        mel = librosa.feature.melspectrogram(
            y=mono.astype(np.float32), sr=DISCOGS_EFFNET_SAMPLE_RATE, n_mels=96
        )
        log_mel = librosa.power_to_db(mel).astype(np.float32)

        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: log_mel[np.newaxis, ...]})
        return np.asarray(outputs[0]).reshape(-1)


class MtgJamendoClassifier:
    """Real :class:`ClassifierModel` — MTG-Jamendo classifier head over a
    Discogs-EffNet embedding. Same real/live-session-only caveat as
    :class:`DiscogsEffnetOnnxModel`."""

    def __init__(
        self, *, model_path: str | Path | None = None, labels: list[str] | None = None
    ) -> None:
        self._model_path = _resolve_model_path(model_path, _MTG_JAMENDO_MODEL_PATH_ENV)
        self._labels = labels
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self._model_path))
        return self._session

    def classify(self, embedding: np.ndarray) -> ClassifierResult:
        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: embedding[np.newaxis, ...].astype(np.float32)})
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
