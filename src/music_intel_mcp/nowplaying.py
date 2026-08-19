"""Now-playing detection for the live-capture spike (#124 AC4).

Uses Windows System Media Transport Controls (SMTC) via the ``winsdk`` package
rather than Spotify Web API user-authorized OAuth: this codebase's Spotify
integration (``spotify_api.py``) only implements the client-credentials flow,
and SMTC gives both the track metadata *and* the source app/process in one
OS-level call with no new auth flow to build. ``resolve_now_playing`` feeds
whatever identity SMTC exposes into the existing spotify_id -> ISRC -> MBID
waterfall (:class:`~music_intel_mcp.identity.IdentityResolver`) so the played
track is matched the same way every other track in this codebase is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .identity import IdentityResolver, ResolvedIdentity
from .models import TrackRef


@dataclass
class NowPlayingInfo:
    """What SMTC (or a test double) reports about the currently playing track."""

    title: str
    artist: str
    app_id: str
    process_id: int | None = None
    spotify_id: str | None = None


@runtime_checkable
class NowPlayingSource(Protocol):
    def current(self) -> NowPlayingInfo | None: ...


class InMemoryNowPlayingSource:
    """Fixed-value :class:`NowPlayingSource` for tests."""

    def __init__(self, info: NowPlayingInfo | None) -> None:
        self._info = info

    def current(self) -> NowPlayingInfo | None:
        return self._info


class SmtcNowPlayingSource:
    """Production :class:`NowPlayingSource` backed by the Windows SMTC session
    manager (``winsdk.windows.media.control``). Imports ``winsdk`` lazily so
    the module stays importable (and this class merely unusable) off Windows
    or without the package installed."""

    def current(self) -> NowPlayingInfo | None:
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        )

        manager = _await(SessionManager.request_async())
        session = manager.get_current_session()
        if session is None:
            return None
        props = _await(session.try_get_media_properties_async())
        app_id = session.source_app_user_model_id
        process_id = _process_id_for_app(app_id)
        return NowPlayingInfo(
            title=props.title or "",
            artist=props.artist or "",
            app_id=app_id,
            process_id=process_id,
        )


def _await(async_op):
    """Synchronously drain a WinRT ``IAsyncOperation`` (winsdk has no bundled
    sync helper); used only by the production SMTC path, never in tests."""
    import asyncio

    return asyncio.run(async_op)


def _process_id_for_app(app_id: str) -> int | None:
    """Best-effort PID lookup for an SMTC ``source_app_user_model_id``, used to
    scope WASAPI per-process loopback capture.

    The AUMID suffix (e.g. ``Spotify`` from ``SpotifyAB.SpotifyMusic_...!Spotify``)
    is the app's display name, not its exe filename — psutil reports
    ``Spotify.exe``, so comparison strips the extension. UWP-packaged apps (like
    the Store Spotify build) also run as several sibling ``Spotify.exe``
    processes rather than one; loopback capture uses
    ``PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE`` (_wasapi_loopback.py),
    which only covers the target PID's own descendants, so the ancestor-most
    match (the one whose parent isn't itself a match) is picked to maximize
    tree coverage.
    """
    import psutil

    name = app_id.rsplit("!", 1)[-1]  # AUMID may be "Package!App" or a bare exe name
    stem = Path(name).stem.lower()

    matches = [
        proc.info["pid"]
        for proc in psutil.process_iter(["pid", "name"])
        if Path(proc.info.get("name") or "").stem.lower() == stem
    ]
    if not matches:
        return None

    match_set = set(matches)
    for pid in matches:
        try:
            ppid = psutil.Process(pid).ppid()
        except psutil.Error:
            continue
        if ppid not in match_set:
            return pid
    return matches[0]


def resolve_now_playing(
    source: NowPlayingSource, resolver: IdentityResolver
) -> tuple[NowPlayingInfo, ResolvedIdentity] | None:
    """Read the current track from ``source`` and resolve it through
    ``resolver``'s identity waterfall. ``None`` when nothing is playing."""
    now_playing = source.current()
    if now_playing is None:
        return None
    track = TrackRef(
        spotify_id=now_playing.spotify_id,
        name=now_playing.title,
        artist=now_playing.artist,
    )
    identity = resolver.resolve(track)
    return now_playing, identity
