"""Now-playing detection for the live-capture spike (#124 AC4).

Uses Windows System Media Transport Controls (SMTC) via the ``winsdk`` package
rather than Spotify Web API user-authorized OAuth: this codebase's Spotify
integration (``spotify_api.py``) only implements the client-credentials flow,
and SMTC gives both the track metadata *and* the source app/process in one
OS-level call with no new auth flow to build.

Two identity resolvers are involved, deliberately kept apart (#145): the
pre-capture browser-admission gate (``SmtcNowPlayingSource`` /
``_is_resolvable``) walks the richer, AcoustID-first
:class:`~music_intel_mcp.live_identity.LiveIdentityResolver` waterfall to
decide whether a candidate browser tab is even worth capturing, while
``resolve_now_playing`` — a separate, out-of-scope-for-#145 function — feeds
whatever identity SMTC exposes into the older spotify_id -> ISRC -> MBID
waterfall (:class:`~music_intel_mcp.identity.IdentityResolver`) so the played
track is matched the same way every other track in this codebase is. They
share a same-named ``resolver`` parameter by coincidence, not by design
relationship.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from .identity import IdentityResolver, ResolvedIdentity
from .live_identity import LiveIdentityResolver
from .models import TrackRef
from .store import resolve_data_root

# #145 AC3: rejected browser SMTC candidates are a transparent rejection, not
# a silent drop — appended here (JSONL, mirrors store.py's append-only
# convention) so a wrong admission call is debuggable after the fact.
DEFAULT_DROP_LOG_FILENAME = "smtc_drops.jsonl"

# App-id allowlist (#138): only these app "stems" (AUMID suffix / exe name,
# lowercased, extension-stripped — same normalization as _process_id_for_app)
# are ever captured. A browser's AUMID identifies the browser process, not the
# tab/site, so a browser candidate additionally needs the resolvability gate
# below (see _select_now_playing) — see CONTEXT.md "Live capture pipeline".
DEFAULT_SPOTIFY_APP_STEM = "spotify"
DEFAULT_BROWSER_APP_STEMS = frozenset({"chrome", "msedge", "firefox", "brave", "opera", "vivaldi"})

AppClass = Literal["spotify", "browser", "other"]


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
    or without the package installed.

    Enumerates *all* sessions (#138) rather than trusting
    ``get_current_session()``'s single-session guess — a music session behind
    a non-media foreground app (e.g. a browser tab holding focus) is otherwise
    invisible. Selection across the enumerated, allowlisted, Playing
    candidates is delegated to :func:`_select_now_playing`, which needs
    ``identity_resolver`` to gate ambiguous browser-AUMID candidates.
    """

    def __init__(
        self,
        *,
        identity_resolver: LiveIdentityResolver,
        drop_log_path: str | Path | None = None,
    ) -> None:
        self._identity_resolver = identity_resolver
        self._admission_memo: dict[tuple[str, str, str], bool] = {}
        self._drop_log_path = (
            Path(drop_log_path)
            if drop_log_path is not None
            else resolve_data_root(None) / DEFAULT_DROP_LOG_FILENAME
        )

    def _log_drop(self, info: NowPlayingInfo) -> None:
        self._drop_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"title": info.title, "artist": info.artist, "app_id": info.app_id}
        with self._drop_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    def current(self) -> NowPlayingInfo | None:
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as SessionManager,
        )
        from winsdk.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus,
        )

        manager = _await(SessionManager.request_async())
        candidates = []
        for session in manager.get_sessions():
            props = _await(session.try_get_media_properties_async())
            playback_info = session.get_playback_info()
            app_id = session.source_app_user_model_id
            info = NowPlayingInfo(
                title=props.title or "",
                artist=props.artist or "",
                app_id=app_id,
                process_id=_process_id_for_app(app_id),
            )
            is_playing = (
                playback_info is not None
                and playback_info.playback_status == PlaybackStatus.PLAYING
            )
            candidates.append(_SmtcCandidate(info=info, is_playing=is_playing))

        return _select_now_playing(
            candidates,
            resolver=self._identity_resolver,
            admission_memo=self._admission_memo,
            on_drop=self._log_drop,
        )


def _await(async_op):
    """Synchronously drain a WinRT ``IAsyncOperation`` (winsdk has no bundled
    sync helper); used only by the production SMTC path, never in tests."""
    import asyncio

    return asyncio.run(async_op)


@dataclass
class _SmtcCandidate:
    """One enumerated SMTC session, pre-selection (#138)."""

    info: NowPlayingInfo
    is_playing: bool


def _app_stem(app_id: str) -> str:
    """Normalize an SMTC ``source_app_user_model_id`` to a comparable "stem":
    strip the AUMID's ``Package!App`` suffix (or bare exe name) down to its
    lowercased, extension-free display name — the same normalization
    :func:`_process_id_for_app` uses to match psutil process names.

    Real multi-profile Chromium AUMIDs (e.g. ``MSEdge.UserData.Profile2``)
    have no ``!`` separator and multiple dot segments, so ``Path.stem``
    (which only strips the last segment) leaves ``msedge.userdata`` instead
    of the bare process stem. Take the first dot segment instead — this
    still collapses a plain ``chrome.exe`` to ``chrome``."""
    name = app_id.rsplit("!", 1)[-1]
    return Path(name).name.split(".", 1)[0].lower()


def _classify_app(
    app_id: str,
    *,
    spotify_stem: str = DEFAULT_SPOTIFY_APP_STEM,
    browser_stems: frozenset[str] = DEFAULT_BROWSER_APP_STEMS,
) -> AppClass:
    stem = _app_stem(app_id)
    if stem == spotify_stem:
        return "spotify"
    if stem in browser_stems:
        return "browser"
    return "other"


def _admission_memo_key(candidate: _SmtcCandidate) -> tuple[str, str, str]:
    """Mirrors ``continuous_capture.py``'s ``_track_key`` shape."""
    return (candidate.info.title, candidate.info.artist, candidate.info.app_id)


def _is_resolvable(
    candidate: _SmtcCandidate,
    resolver: LiveIdentityResolver,
    admission_memo: dict[tuple[str, str, str], bool],
) -> bool:
    """A browser SMTC session's ``app_id`` can't distinguish a music tab from
    any other tab (e.g. a podcast, a video call) playing audio in the same
    browser — so browser candidates are only admitted if their track actually
    resolves through the identity waterfall to something more specific than a
    bare name match. This gate runs pre-capture (no PCM yet), so it only ever
    reaches the waterfall's text rungs (spotify_search, mb_name) — the
    ``fingerprint=None`` AcoustID rung is a no-op here (#145). See
    CONTEXT.md "Live capture pipeline" for why an allowlist alone can't do
    this.

    ``admission_memo`` caches the verdict per ``(title, artist, app_id)`` for
    the session's lifetime (#145 AC2) — this gate runs on every 5s poll, not
    once per track, so without memoization the same browser track would walk
    the waterfall repeatedly."""
    key = _admission_memo_key(candidate)
    if key in admission_memo:
        return admission_memo[key]
    identity = resolver.resolve(
        title=candidate.info.title,
        artist=candidate.info.artist,
        fingerprint=None,
    )
    verdict = identity.level != "name"
    admission_memo[key] = verdict
    return verdict


def _select_now_playing(
    candidates: list[_SmtcCandidate],
    *,
    resolver: LiveIdentityResolver,
    admission_memo: dict[tuple[str, str, str], bool] | None = None,
    on_drop: Callable[[NowPlayingInfo], None] | None = None,
    spotify_stem: str = DEFAULT_SPOTIFY_APP_STEM,
    browser_stems: frozenset[str] = DEFAULT_BROWSER_APP_STEMS,
) -> NowPlayingInfo | None:
    """Pick one track out of every enumerated SMTC session (#138).

    Filters to allowlisted (Spotify or a known browser), currently-``Playing``
    candidates; browser candidates are additionally gated on identity
    resolvability. Ties (multiple simultaneously-Playing candidates) prefer
    Spotify, else the first Playing candidate in enumeration order — SMTC
    exposes no reliable recency signal, so no recency heuristic is used.

    ``admission_memo`` defaults to a fresh, call-scoped dict when omitted
    (as every pre-#145-AC2 caller does) — only a caller holding one memo
    across repeated calls (:class:`SmtcNowPlayingSource`) gets the
    once-per-session memoization.

    ``on_drop`` is called with a rejected browser candidate's
    :class:`NowPlayingInfo` (#145 AC3) — a transparent-rejection hook so a
    caller can persist a breadcrumb without this function knowing about file
    I/O. Never called for the memo's cached rejections (only the resolver
    call site knows it's a fresh verdict) — see
    :class:`SmtcNowPlayingSource` for the JSONL sink.
    """
    if admission_memo is None:
        admission_memo = {}
    allowlisted = []
    for candidate in candidates:
        if not candidate.is_playing:
            continue
        app_class = _classify_app(
            candidate.info.app_id, spotify_stem=spotify_stem, browser_stems=browser_stems
        )
        if app_class == "other":
            continue
        if app_class == "browser":
            was_cached = _admission_memo_key(candidate) in admission_memo
            if not _is_resolvable(candidate, resolver, admission_memo):
                if on_drop is not None and not was_cached:
                    on_drop(candidate.info)
                continue
        allowlisted.append((app_class, candidate))

    for app_class, candidate in allowlisted:
        if app_class == "spotify":
            return candidate.info
    if allowlisted:
        return allowlisted[0][1].info
    return None


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

    stem = _app_stem(app_id)

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
