"""Per-process WASAPI loopback capture via a subprocess-based native helper.

Isolated in its own module (imported lazily by :mod:`capture`, never at package
import time) because it is Windows-only and hardware-facing.
:class:`~music_intel_mcp.capture.WasapiProcessLoopbackCapture` is validated
against a real playback session (issue #124 AC1).

Implementation note: the WASAPI process-loopback activation
(``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK``) delivers its
``ActivateCompleted`` callback via genuine out-of-process RPC from audiosrv.
comtypes never receives that callback for this specific API (root-caused via
a hand-rolled native repro that proved the same activation call works
correctly outside comtypes/Python). This module instead spawns a small
compiled C++ helper (``native/wasapi_loopback_helper``) that performs the
activation and capture natively, and streams raw PCM back over a pipe — see
that directory's ``main.cpp`` for the wire protocol and the activation code.

Wire protocol (helper stdout, binary):
  1. One 16-byte header: 4x little-endian uint32 =
     (status, sample_rate, channels, bits_per_sample). status == 0 is
     success; nonzero is an HRESULT (or 0xFFFFFFFF for "activation timed
     out") and no PCM follows.
  2. On success: a continuous raw interleaved PCM byte stream until the
     helper's max-seconds deadline elapses or the process is terminated.

Audio is never written to disk by this module or the helper (issue #124
AC2) — the helper writes only to its stdout pipe, and this module only ever
buffers the bytes it reads from that pipe in memory.
"""

from __future__ import annotations

import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

_HEADER_SIZE = 16
_HEADER_FORMAT = "<IIII"

# Helper self-terminates after this many seconds regardless of how much is
# actually consumed — a safety cap, not the intended capture duration.
_HELPER_MAX_SECONDS = 3600.0

_HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "native"
    / "wasapi_loopback_helper"
    / "wasapi_loopback_helper.exe"
)


class WasapiLoopbackError(OSError):
    """Raised when the native helper fails to activate or start capture."""


class _StreamReader:
    """Pumps a subprocess stdout pipe into an in-memory, lock-protected buffer."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._eof = False
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while True:
            chunk = self._stream.read(65536)
            if not chunk:
                break
            with self._lock:
                self._buffer.extend(chunk)
        with self._lock:
            self._eof = True

    def take(self, max_bytes: int | None = None) -> bytes:
        """Remove and return up to ``max_bytes`` from the front of the buffer."""
        with self._lock:
            if max_bytes is None or max_bytes >= len(self._buffer):
                data = bytes(self._buffer)
                self._buffer.clear()
            else:
                data = bytes(self._buffer[:max_bytes])
                del self._buffer[:max_bytes]
            return data

    def available(self) -> int:
        with self._lock:
            return len(self._buffer)


def activate_process_loopback(
    *, target_pid: int, sample_rate: int = 44100, channels: int = 2
) -> tuple[subprocess.Popen, _StreamReader]:
    """Spawn the native helper and wait for it to report activation status.

    Returns ``(proc, reader)`` where ``proc`` is the running helper process
    (``stop()`` terminates it) and ``reader`` accumulates PCM bytes read from
    its stdout (consumed by :func:`read_available`).

    The helper's own audio format is fixed (44100 Hz, 16-bit, stereo) — the
    ``sample_rate``/``channels`` arguments here describe what the caller
    expects and are validated against what the helper actually reports.
    """
    if not _HELPER_PATH.exists():
        raise WasapiLoopbackError(
            f"native helper not found at {_HELPER_PATH} — build it with "
            "scripts/build_wasapi_helper.ps1"
        )

    proc = subprocess.Popen(
        [str(_HELPER_PATH), str(target_pid), str(_HELPER_MAX_SECONDS)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )

    header = _read_exact(proc.stdout, _HEADER_SIZE, timeout_s=30.0, proc=proc)
    status, hdr_sample_rate, hdr_channels, hdr_bits = struct.unpack(_HEADER_FORMAT, header)

    if status != 0:
        proc.wait(timeout=5)
        if status == 0xFFFFFFFF:
            raise WasapiLoopbackError("WASAPI activation timed out in native helper")
        raise WasapiLoopbackError(
            f"WASAPI activation/capture-start failed in native helper: HRESULT=0x{status:08X}"
        )

    if hdr_sample_rate != sample_rate or hdr_channels != channels or hdr_bits != 16:
        proc.terminate()
        raise WasapiLoopbackError(
            f"native helper format mismatch: expected {sample_rate}Hz/"
            f"{channels}ch/16bit, got {hdr_sample_rate}Hz/{hdr_channels}ch/"
            f"{hdr_bits}bit"
        )

    reader = _StreamReader(proc.stdout)
    return proc, reader


def _read_exact(stream, n: int, *, timeout_s: float, proc: subprocess.Popen) -> bytes:
    """Read exactly ``n`` bytes from ``stream``, or raise on timeout/EOF."""
    data = bytearray()
    deadline = time.monotonic() + timeout_s
    while len(data) < n:
        if time.monotonic() > deadline:
            proc.kill()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            raise TimeoutError(f"timed out waiting for native helper header ({stderr.strip()})")
        chunk = stream.read(n - len(data))
        if not chunk:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise WasapiLoopbackError(
                    f"native helper exited early (code={proc.returncode}): {stderr.strip()}"
                )
            continue
        data.extend(chunk)
    return bytes(data)


def read_available(
    reader: _StreamReader, *, duration_s: float, sample_rate: int, channels: int
) -> np.ndarray:
    """Poll ``reader`` for up to ``duration_s`` worth of PCM, then return it.

    Returns a ``float32`` array shaped ``(n_samples, channels)`` in [-1, 1].
    May return fewer samples than ``duration_s`` implies if the helper has
    not produced that much audio yet.
    """
    bytes_per_frame = channels * 2  # 16-bit PCM
    target_bytes = int(duration_s * sample_rate) * bytes_per_frame

    deadline = time.monotonic() + duration_s + 1.0  # small grace period
    while reader.available() < target_bytes and time.monotonic() < deadline:
        time.sleep(0.01)

    raw = reader.take(target_bytes)
    # Drop any trailing partial frame.
    usable_len = len(raw) - (len(raw) % bytes_per_frame)
    raw = raw[:usable_len]

    samples_i16 = np.frombuffer(raw, dtype="<i2")
    n_frames = len(samples_i16) // channels if channels else 0
    samples_i16 = samples_i16[: n_frames * channels].reshape(n_frames, channels)
    return (samples_i16.astype(np.float32)) / 32768.0


def stop(proc: subprocess.Popen) -> None:
    """Terminate the native helper. Hard kill — nothing is ever flushed to disk."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
