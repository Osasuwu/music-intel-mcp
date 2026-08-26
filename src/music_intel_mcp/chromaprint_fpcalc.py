"""Chromaprint fingerprinting via the ``fpcalc`` CLI (#139 AC1/AC7).

The live-capture path needs a chromaprint fingerprint computed from the raw
PCM this process just captured (never written to disk as a permanent audio
file — :class:`~music_intel_mcp.capture.RingBufferSink` is in-memory only).
``fpcalc`` (the reference Chromaprint command-line tool, same one the
AcoustID ecosystem is built around) reads any file ``libavcodec``/``ffmpeg``
can decode and prints a JSON ``{"duration": ..., "fingerprint": ...}``
payload — so this module writes the captured PCM to a *temporary* WAV file,
shells out, and deletes it, rather than depending on a Python chromaprint
binding (there isn't a well-documented one to trust — see AC7's setup-script
companion, which fetches the ``fpcalc`` binary itself).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

_FPCALC_TIMEOUT_S = 30.0


class FpcalcNotFoundError(RuntimeError):
    """Raised when the ``fpcalc`` binary isn't on PATH (AC7's setup script installs it)."""


def _write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM WAV from a float32 ``(n_samples, channels)`` array in [-1, 1]."""
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    clipped = np.clip(pcm, -1.0, 1.0)
    samples_i16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_i16.tobytes())


def compute_fingerprint(
    pcm: np.ndarray, sample_rate: int, *, fpcalc_path: str = "fpcalc"
) -> tuple[str, float]:
    """Chromaprint fingerprint + duration (seconds) for captured PCM (AC1).

    Raises :class:`FpcalcNotFoundError` if ``fpcalc`` isn't installed, and
    ``RuntimeError`` if it exits non-zero or emits unparseable output.
    """
    if shutil.which(fpcalc_path) is None:
        raise FpcalcNotFoundError(
            f"'{fpcalc_path}' not found on PATH — run the collector setup script "
            "to fetch it (#139 AC7)."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "capture.wav"
        _write_wav(wav_path, pcm, sample_rate)
        result = subprocess.run(
            [fpcalc_path, "-json", str(wav_path)],
            capture_output=True,
            text=True,
            timeout=_FPCALC_TIMEOUT_S,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(f"fpcalc exited {result.returncode}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        return payload["fingerprint"], float(payload["duration"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"fpcalc produced unparseable output: {result.stdout!r}") from exc
