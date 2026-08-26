"""Continuous local capture loop (#136) — turns the one-shot #124 spike into
a supervised collector that keeps running until stopped.

Wraps :func:`~music_intel_mcp.live_pipeline.run_live_capture_spike` per
detected track change: polls a :class:`~music_intel_mcp.nowplaying.NowPlayingSource`
on an interval, skips re-capturing a track that's still playing, and isolates
any per-track failure (capture/inference/store) behind ``on_error`` so one bad
track never kills an unattended session — the loop logs and keeps
polling rather than raising. ``stop_event`` (a plain ``threading.Event``) is
the cooperative shutdown seam a tray "Quit" action or a test drives.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .capture import LoopbackSource
from .inference import AudioEmbeddingModel, ClassifierModel
from .live_identity import LiveIdentityResolver
from .live_pipeline import LiveCaptureResult, run_live_capture_spike
from .nowplaying import InMemoryNowPlayingSource, NowPlayingInfo, NowPlayingSource
from .store import UserStore

_TrackKey = tuple[str, str, str]


def _track_key(info: NowPlayingInfo) -> _TrackKey:
    """Same-track-still-playing proxy: title+artist+source app, checked before
    identity resolution ever runs (which happens inside the wrapped spike)."""
    return (info.title, info.artist, info.app_id)


def run_continuous_capture(
    *,
    now_playing_source: NowPlayingSource,
    live_identity_resolver: LiveIdentityResolver,
    capture_factory: Callable[[NowPlayingInfo], LoopbackSource],
    embedding_model: AudioEmbeddingModel,
    classifier: ClassifierModel,
    store: UserStore,
    capture_duration_s: float = 120.0,
    poll_interval_s: float = 5.0,
    stop_event: threading.Event | None = None,
    on_result: Callable[[NowPlayingInfo, LiveCaptureResult | None], None] | None = None,
    on_error: Callable[[NowPlayingInfo, Exception], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll forever (until ``stop_event`` is set) capturing each new track once.

    ``capture_factory`` builds a fresh :class:`LoopbackSource` per detected
    track (the real backend needs the track's resolved ``process_id`` at
    construction time, which can change between tracks even for the same app).
    """
    stop_event = stop_event or threading.Event()
    last_key: _TrackKey | None = None

    while not stop_event.is_set():
        now_playing = now_playing_source.current()
        if now_playing is None:
            last_key = None
        else:
            key = _track_key(now_playing)
            if key == last_key or now_playing.process_id is None:
                pass
            else:
                last_key = key
                try:
                    capture = capture_factory(now_playing)
                    result = run_live_capture_spike(
                        duration_s=capture_duration_s,
                        now_playing_source=InMemoryNowPlayingSource(now_playing),
                        live_identity_resolver=live_identity_resolver,
                        capture=capture,
                        embedding_model=embedding_model,
                        classifier=classifier,
                        store=store,
                    )
                except Exception as exc:  # must survive to keep polling (#136) — an
                    # unattended session shouldn't die on one bad track.
                    if on_error is not None:
                        on_error(now_playing, exc)
                else:
                    if on_result is not None:
                        on_result(now_playing, result)

        sleep(poll_interval_s)
