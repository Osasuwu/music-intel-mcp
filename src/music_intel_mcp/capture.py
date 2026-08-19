"""Live audio capture for the WASAPI per-process loopback spike (#124).

Sourcing decision (1c0f37f0): passive, per-process loopback — prefer
``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`` over device-level loopback so
capture isolates the target streaming app and never picks up the post-mix
system output. Listen-analyze-discard: captured PCM lives only in
:class:`RingBufferSink`, a bounded in-process buffer, and is never written to
disk (AC2). The real Windows backend (:class:`WasapiProcessLoopbackCapture`)
and the in-memory sink share the :class:`LoopbackSource` seam so the
orchestration pipeline (``live_pipeline.py``) is fully testable against
:class:`FakeLoopbackCapture` without hardware; only the real backend is
exercised in the live smoke session with a real target process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class AudioFrame:
    """A block of PCM samples, shape ``(n_samples, channels)``, ``float32``."""

    samples: np.ndarray
    sample_rate: int


@runtime_checkable
class LoopbackSource(Protocol):
    """start/read/stop seam every capture backend (real or fake) implements."""

    def start(self) -> None: ...

    def read(self, duration_s: float) -> AudioFrame: ...

    def stop(self) -> None: ...


class RingBufferSink:
    """Bounded in-memory PCM buffer — the only place captured audio lives.

    Enforces AC2 structurally: no method here ever opens a file. Oldest frames
    are dropped once ``max_seconds`` of audio has accumulated, so a long-running
    capture session can't grow the buffer unbounded.
    """

    def __init__(self, *, max_seconds: float, sample_rate: int, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._max_samples = int(max_seconds * sample_rate)
        self._frames: deque[np.ndarray] = deque()
        self._total_samples = 0

    def write(self, frame: AudioFrame) -> None:
        self._frames.append(frame.samples)
        self._total_samples += len(frame.samples)
        while self._total_samples > self._max_samples and len(self._frames) > 1:
            dropped = self._frames.popleft()
            self._total_samples -= len(dropped)

    def read_all(self) -> np.ndarray:
        """Concatenate every buffered frame. Empty buffer -> honest-empty array."""
        if not self._frames:
            return np.zeros((0, self.channels), dtype=np.float32)
        return np.concatenate(list(self._frames), axis=0)

    @property
    def duration_s(self) -> float:
        return self._total_samples / self.sample_rate if self.sample_rate else 0.0


class FakeLoopbackCapture:
    """Test double for :class:`LoopbackSource` — synthesizes a sine tone instead
    of touching WASAPI, so the orchestration pipeline is testable without
    hardware or a live playback session."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        frequency_hz: float = 440.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frequency_hz = frequency_hz
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self, duration_s: float) -> AudioFrame:
        n = max(0, int(duration_s * self.sample_rate))
        t = np.arange(n) / self.sample_rate
        tone = (0.1 * np.sin(2 * np.pi * self.frequency_hz * t)).astype(np.float32)
        samples = np.repeat(tone.reshape(-1, 1), self.channels, axis=1)
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)

    def stop(self) -> None:
        self.stopped = True


class WasapiProcessLoopbackCapture:
    """Real per-process WASAPI loopback backend (Windows 10 2004+, build 19041+).

    Activates ``VAD\\Process_Loopback`` via ``ActivateAudioInterfaceAsync`` with
    ``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`` scoped to ``target_pid`` (and
    its child processes, per ``PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE``)
    so only that process's render stream is captured — never the post-mix system
    output. COM/ctypes interop against this API is intentionally kept isolated to
    this one class; it is validated against a real playback session (issue #124
    AC1) rather than unit-tested, since it has no meaningful in-process fake.
    """

    def __init__(self, *, target_pid: int, sample_rate: int = 44100, channels: int = 2) -> None:
        self.target_pid = target_pid
        self.sample_rate = sample_rate
        self.channels = channels
        self.started = False
        self.stopped = False
        self._audio_client = None
        self._capture_client = None

    def start(self) -> None:
        from . import _wasapi_loopback

        self._audio_client, self._capture_client = _wasapi_loopback.activate_process_loopback(
            target_pid=self.target_pid,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self.started = True

    def read(self, duration_s: float) -> AudioFrame:
        if not self.started or self._capture_client is None:
            raise RuntimeError("WasapiProcessLoopbackCapture.start() must run before read()")
        from . import _wasapi_loopback

        samples = _wasapi_loopback.read_available(
            self._capture_client,
            duration_s=duration_s,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)

    def stop(self) -> None:
        if self._audio_client is not None:
            from . import _wasapi_loopback

            _wasapi_loopback.stop(self._audio_client)
        self.stopped = True
