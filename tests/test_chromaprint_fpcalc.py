"""fpcalc-backed chromaprint fingerprinting (#139 AC1/AC7).

``fpcalc`` itself is never actually invoked here (AC9 — no reliance on a real
external binary in CI); ``subprocess.run`` is monkeypatched with a fake that
mimics its ``-json`` output shape.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from music_intel_mcp.chromaprint_fpcalc import FpcalcNotFoundError, compute_fingerprint


def _sine_pcm(seconds: float = 1.0, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    return tone.astype(np.float32).reshape(-1, 1)


def test_compute_fingerprint_parses_fpcalc_json(monkeypatch):
    monkeypatch.setattr(
        "music_intel_mcp.chromaprint_fpcalc.shutil.which", lambda _name: "/usr/bin/fpcalc"
    )

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "fpcalc"
        assert cmd[1] == "-json"
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"duration": 12, "fingerprint": "AQAA...FAKE"}', stderr=""
        )

    monkeypatch.setattr("music_intel_mcp.chromaprint_fpcalc.subprocess.run", fake_run)

    fingerprint, duration = compute_fingerprint(_sine_pcm(), 44100)

    assert fingerprint == "AQAA...FAKE"
    assert duration == 12.0


def test_compute_fingerprint_raises_when_fpcalc_missing(monkeypatch):
    monkeypatch.setattr("music_intel_mcp.chromaprint_fpcalc.shutil.which", lambda _name: None)

    with pytest.raises(FpcalcNotFoundError):
        compute_fingerprint(_sine_pcm(), 44100)


def test_compute_fingerprint_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        "music_intel_mcp.chromaprint_fpcalc.shutil.which", lambda _name: "/usr/bin/fpcalc"
    )
    monkeypatch.setattr(
        "music_intel_mcp.chromaprint_fpcalc.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        compute_fingerprint(_sine_pcm(), 44100)


def test_compute_fingerprint_raises_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(
        "music_intel_mcp.chromaprint_fpcalc.shutil.which", lambda _name: "/usr/bin/fpcalc"
    )
    monkeypatch.setattr(
        "music_intel_mcp.chromaprint_fpcalc.subprocess.run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr=""),
    )

    with pytest.raises(RuntimeError, match="unparseable"):
        compute_fingerprint(_sine_pcm(), 44100)
