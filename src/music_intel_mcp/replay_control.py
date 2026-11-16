"""Run control for the replay queue (#279, decision ``81859701``).

One place decides whether a replay run continues, pauses or stops:
:class:`RunControl` wraps a raw ``control_source`` (S4/#280 supplies the
audibility and SMTC sources; tests use a scripted fake) and is polled by both
the queue loop and the capture loops in :mod:`~music_intel_mcp.replay_capture`.
A ``pause`` is handled inside :meth:`RunControl.poll` -- the driver is paused,
keep-awake released, the source re-polled until it says ``continue`` or
``stop`` -- so the loops only ever see ``continue`` or ``stop``. Each pause
bumps :attr:`RunControl.pause_epoch`; a capture attempt that sees the epoch
change ends ``yielded`` and discards its buffer.

Keep-awake (:class:`KeepAwake`) is held only while a capture is in progress:
acquired once the capture is armed, released in the capture's ``finally``,
in a pause, and at run end.
"""

from __future__ import annotations

import ctypes
import logging
import time
from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ControlSignal = Literal["continue", "pause", "stop"]

SILENT_STREAK_LIMIT = 5
"""Consecutive ``silent`` attempts that stop the run (#279 AC2)."""

FAILURE_STREAK_LIMIT = 3
"""Consecutive non-silent failures that stop the run (#279 AC3):
``play_failed``, ``sample_rate_mismatch``, ``error`` and ``short`` (Пётр,
16.11: a short capture is not a silent failure)."""

DEFAULT_PAUSE_POLL_S = 5.0

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@runtime_checkable
class KeepAwake(Protocol):
    """Keeps the machine from sleeping while held. Both calls idempotent."""

    def acquire(self) -> None: ...

    def release(self) -> None: ...


class NullKeepAwake:
    """No-op :class:`KeepAwake` -- the default when no seam is supplied."""

    def acquire(self) -> None: ...

    def release(self) -> None: ...


class WindowsKeepAwake:
    """``SetThreadExecutionState`` behind the :class:`KeepAwake` seam. The
    state is per thread, so acquire and release must run on the capture
    thread (the replay loop is single-threaded). ``set_state`` is injectable
    so the call sequence is testable off Windows."""

    def __init__(self, set_state: Callable[[int], object] | None = None) -> None:
        self._set_state = set_state
        self._held = False

    def _call(self, flags: int) -> None:
        if self._set_state is None:
            self._set_state = ctypes.windll.kernel32.SetThreadExecutionState  # type: ignore[attr-defined]
        self._set_state(flags)

    def acquire(self) -> None:
        if not self._held:
            self._call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            self._held = True

    def release(self) -> None:
        if self._held:
            self._call(ES_CONTINUOUS)
            self._held = False


class _Pausable(Protocol):
    def pause(self) -> None: ...


class RunControl:
    """Polls ``control_source`` and absorbs ``pause`` (see module docstring)."""

    def __init__(
        self,
        control_source: Callable[[], ControlSignal],
        *,
        driver: _Pausable,
        keep_awake: KeepAwake,
        pause_poll_s: float = DEFAULT_PAUSE_POLL_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._source = control_source
        self._driver = driver
        self._keep_awake = keep_awake
        self._pause_poll_s = pause_poll_s
        self._sleep = sleep
        self.pause_epoch = 0

    def poll(self) -> Literal["continue", "stop"]:
        signal = self._source()
        if signal != "pause":
            return "stop" if signal == "stop" else "continue"
        self.pause_epoch += 1
        try:
            self._driver.pause()
        except Exception:  # a failed pause must not end the run from here
            logger.exception("driver.pause() failed while pausing for control")
        self._keep_awake.release()
        while True:
            self._sleep(self._pause_poll_s)
            signal = self._source()
            if signal == "stop":
                return "stop"
            if signal == "continue":
                return "continue"
