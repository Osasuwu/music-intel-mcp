"""Tests for the subprocess/IPC-based native-helper WASAPI loopback client (#124).

Unlike the old ctypes/comtypes implementation this replaces, the subprocess
boundary is mockable: these tests fake ``subprocess.Popen`` entirely, so the
header-parsing / error-handling / PCM-accumulation logic gets real automated
coverage without touching real hardware or a real WASAPI activation.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from music_intel_mcp import _wasapi_loopback as wl

_HEADER_FORMAT = "<IIII"


def _header(status: int, sample_rate: int = 44100, channels: int = 2, bits: int = 16) -> bytes:
    return struct.pack(_HEADER_FORMAT, status, sample_rate, channels, bits)


class _FakeHelperPath:
    """Stands in for the module's _HELPER_PATH so tests don't need a real build."""

    def exists(self) -> bool:
        return True

    def __str__(self) -> str:
        return "fake_wasapi_loopback_helper.exe"


class _FakeProc:
    def __init__(self, stdout_bytes: bytes, *, returncode: int | None = None) -> None:
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(b"")
        self._returncode = returncode
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        return self._returncode

    def terminate(self):
        self.terminated = True
        self._returncode = 0
        self.returncode = 0

    def kill(self):
        self.killed = True
        self._returncode = -9
        self.returncode = -9


@pytest.fixture(autouse=True)
def _fake_helper_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wl, "_HELPER_PATH", _FakeHelperPath())


def test_activate_process_loopback_raises_when_helper_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingPath(_FakeHelperPath):
        def exists(self) -> bool:
            return False

    monkeypatch.setattr(wl, "_HELPER_PATH", _MissingPath())

    with pytest.raises(wl.WasapiLoopbackError, match="not found"):
        wl.activate_process_loopback(target_pid=1234)


def test_activate_process_loopback_success_returns_proc_and_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm = (b"\x01\x00\x02\x00") * 100  # 100 stereo int16 frames
    fake_proc = _FakeProc(_header(0) + pcm)
    monkeypatch.setattr(wl.subprocess, "Popen", lambda *a, **k: fake_proc)

    proc, reader = wl.activate_process_loopback(target_pid=1234, sample_rate=44100, channels=2)

    assert proc is fake_proc
    # Give the background pump thread a moment to drain the fake stream.
    import time

    deadline = time.monotonic() + 2.0
    while reader.available() < len(pcm) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert reader.available() == len(pcm)


def test_activate_process_loopback_raises_on_timeout_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProc(_header(0xFFFFFFFF), returncode=1)
    monkeypatch.setattr(wl.subprocess, "Popen", lambda *a, **k: fake_proc)

    with pytest.raises(wl.WasapiLoopbackError, match="timed out"):
        wl.activate_process_loopback(target_pid=1234)


def test_activate_process_loopback_raises_on_hresult_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProc(_header(0x80070005), returncode=1)  # E_ACCESSDENIED
    monkeypatch.setattr(wl.subprocess, "Popen", lambda *a, **k: fake_proc)

    with pytest.raises(wl.WasapiLoopbackError, match="0x80070005"):
        wl.activate_process_loopback(target_pid=1234)


def test_activate_process_loopback_raises_on_format_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProc(_header(0, sample_rate=48000, channels=2, bits=16))
    monkeypatch.setattr(wl.subprocess, "Popen", lambda *a, **k: fake_proc)

    with pytest.raises(wl.WasapiLoopbackError, match="format mismatch"):
        wl.activate_process_loopback(target_pid=1234, sample_rate=44100, channels=2)


def test_activate_process_loopback_raises_on_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_proc = _FakeProc(b"\x00\x00", returncode=1)  # short/garbage header, already exited
    monkeypatch.setattr(wl.subprocess, "Popen", lambda *a, **k: fake_proc)

    with pytest.raises(wl.WasapiLoopbackError, match="exited early"):
        wl.activate_process_loopback(target_pid=1234)


def test_read_available_converts_pcm_to_float32_array() -> None:
    # Two stereo frames: (1, -1), (16384, -16384) as int16.
    pcm = struct.pack("<4h", 1, -1, 16384, -16384)
    reader = wl._StreamReader(io.BytesIO(pcm))
    import time

    time.sleep(0.05)  # let the pump thread drain the tiny fixed buffer

    out = wl.read_available(reader, duration_s=2.0, sample_rate=1, channels=2)

    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [1 / 32768.0, -1 / 32768.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [16384 / 32768.0, -16384 / 32768.0], atol=1e-6)


def test_read_available_drops_trailing_partial_frame() -> None:
    # 5 int16 values = 2 full stereo frames + 1 orphan sample.
    pcm = struct.pack("<5h", 1, 2, 3, 4, 5)
    reader = wl._StreamReader(io.BytesIO(pcm))
    import time

    time.sleep(0.05)

    out = wl.read_available(reader, duration_s=0.5, sample_rate=6, channels=2)
    assert out.shape == (2, 2)


def test_stop_terminates_running_process() -> None:
    fake_proc = _FakeProc(b"")
    wl.stop(fake_proc)
    assert fake_proc.terminated is True


def test_stop_is_noop_for_already_exited_process() -> None:
    fake_proc = _FakeProc(b"", returncode=0)
    wl.stop(fake_proc)
    assert fake_proc.terminated is False
