"""Tests for the live-capture spike (#124): in-memory ring buffer + capture seam."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from music_intel_mcp.capture import AudioFrame, FakeLoopbackCapture, RingBufferSink


def _frame(n_samples: int, sample_rate: int = 16000, channels: int = 1) -> AudioFrame:
    samples = np.ones((n_samples, channels), dtype=np.float32)
    return AudioFrame(samples=samples, sample_rate=sample_rate)


# AC2: audio never persisted to disk -- in-memory ring buffer only.
def test_ring_buffer_never_opens_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden_open(*args, **kwargs):
        raise AssertionError("RingBufferSink must never touch the filesystem")

    monkeypatch.setattr(builtins, "open", _forbidden_open)

    sink = RingBufferSink(max_seconds=1.0, sample_rate=16000, channels=1)
    sink.write(_frame(8000))
    sink.write(_frame(8000))
    out = sink.read_all()

    assert out.shape == (16000, 1)


def test_ring_buffer_drops_oldest_frames_past_capacity() -> None:
    sink = RingBufferSink(max_seconds=1.0, sample_rate=100, channels=1)
    sink.write(_frame(60, sample_rate=100))
    sink.write(_frame(60, sample_rate=100))

    assert sink.duration_s <= 1.0
    assert sink.read_all().shape[0] <= 100


def test_ring_buffer_read_all_empty_is_honest_empty() -> None:
    sink = RingBufferSink(max_seconds=1.0, sample_rate=16000, channels=2)
    out = sink.read_all()
    assert out.shape == (0, 2)


# AC1/AC3 support: capture seam is start/read/stop, testable via a fake without hardware.
def test_fake_loopback_capture_yields_requested_duration() -> None:
    capture = FakeLoopbackCapture(sample_rate=16000, channels=1)
    capture.start()
    frame = capture.read(duration_s=0.5)
    capture.stop()

    assert capture.started is True
    assert capture.stopped is True
    assert frame.sample_rate == 16000
    assert frame.samples.shape == (8000, 1)
