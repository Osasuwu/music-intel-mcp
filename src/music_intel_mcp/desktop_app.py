"""System-tray desktop shell for the continuous capture loop (#136).

Wraps :func:`~music_intel_mcp.continuous_capture.run_continuous_capture` in a
background thread behind a ``pystray`` icon so a non-technical user can
start collection with a double-click and stop it from the tray menu, instead
of a terminal invocation (``music-intel capture-loop``). This is the narrow
live-capture-supervision UI the "Engine-first scope" invariant carves out —
not a general product UI, which stays out of V0/V1 per CONTEXT.md.

Opt-in ``desktop`` extra (``pystray`` + ``Pillow``); this module is never
imported by the core pipeline, only by its own console-script entry point
(``music-intel-desktop``), so the extras stay off the default install.
"""

from __future__ import annotations

import os
import threading

from .continuous_capture import run_continuous_capture
from .identity import MusicBrainzIsrcIndex
from .live_identity import AcoustIdApiSource, LiveIdentityResolver, LiveNegativeCache
from .store import UserStore

_CAPTURE_DURATION_S = 120.0  # #139 AC1: ~120s capture window
_POLL_INTERVAL_S = 5.0


def _build_live_resolver(store: UserStore) -> LiveIdentityResolver:
    """Mirrors ``cli.py``'s ``_build_live_resolver``: every network-backed
    rung is gated on its own credential, so a missing key skips that rung
    instead of crashing the tray loop."""
    from .spotify_api import SpotifyApiIsrcSource, SpotifySearchApiSource

    acoustid_source = AcoustIdApiSource() if os.environ.get("ACOUSTID_API_KEY") else None
    spotify_search = None
    spotify_isrc = None
    if os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"):
        isrc_source = SpotifyApiIsrcSource()
        spotify_search = SpotifySearchApiSource(isrc_source=isrc_source)
        spotify_isrc = isrc_source

    return LiveIdentityResolver(
        acoustid_source=acoustid_source,
        spotify_search=spotify_search,
        spotify_isrc=spotify_isrc,
        isrc_index=MusicBrainzIsrcIndex(),
        negative_cache=LiveNegativeCache(root=store.root),
    )


class _Status:
    """Thread-safe last-status string the tray menu polls to render its title."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._text = "starting..."

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text

    def get(self) -> str:
        with self._lock:
            return self._text


def _run_loop(stop_event: threading.Event, status: _Status) -> None:
    from .capture import WasapiProcessLoopbackCapture
    from .inference import DiscogsEffnetOnnxModel, MtgJamendoClassifier
    from .nowplaying import DEFAULT_DROP_LOG_FILENAME, SmtcNowPlayingSource

    store = UserStore()
    live_resolver = _build_live_resolver(store)
    embedding_model = DiscogsEffnetOnnxModel()
    classifier = MtgJamendoClassifier()

    def capture_factory(now_playing):
        return WasapiProcessLoopbackCapture(target_pid=now_playing.process_id)

    def on_result(now_playing, result) -> None:
        if result is None:
            status.set(f"skipped: {now_playing.artist} - {now_playing.title}")
        else:
            status.set(f"captured: {now_playing.artist} - {now_playing.title}")

    def on_error(now_playing, exc: Exception) -> None:
        status.set(f"error on '{now_playing.title}': {exc}")

    status.set("listening for now-playing...")
    try:
        run_continuous_capture(
            now_playing_source=SmtcNowPlayingSource(
                identity_resolver=live_resolver,
                drop_log_path=store.root / DEFAULT_DROP_LOG_FILENAME,
            ),
            live_identity_resolver=live_resolver,
            capture_factory=capture_factory,
            embedding_model=embedding_model,
            classifier=classifier,
            store=store,
            capture_duration_s=_CAPTURE_DURATION_S,
            poll_interval_s=_POLL_INTERVAL_S,
            stop_event=stop_event,
            on_result=on_result,
            on_error=on_error,
        )
    except Exception as exc:  # keep the tray alive to show the fatal error
        status.set(f"stopped: fatal error ({exc})")


def _build_icon_image(color: str):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), fill=color)
    return image


def main() -> int:
    """Console-script entry point (``music-intel-desktop``)."""
    import pystray

    stop_event = threading.Event()
    status = _Status()

    worker = threading.Thread(target=_run_loop, args=(stop_event, status), daemon=True)
    worker.start()

    def on_quit(icon, _item) -> None:
        stop_event.set()
        icon.stop()

    icon = pystray.Icon(
        "music-intel",
        icon=_build_icon_image("#2b67c6"),
        title="music-intel — audio collector",
        menu=pystray.Menu(
            pystray.MenuItem(lambda _item: status.get(), None, enabled=False),
            pystray.MenuItem("Quit", on_quit),
        ),
    )
    icon.run()
    worker.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
