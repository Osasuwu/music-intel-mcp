"""Tests for now-playing detection (#124 AC4): SMTC seam + identity wiring."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from music_intel_mcp import nowplaying as np_module
from music_intel_mcp.identity import (
    IdentityResolver,
    InMemoryIsrcMbidIndex,
    InMemorySpotifyIsrcSource,
)
from music_intel_mcp.nowplaying import (
    InMemoryNowPlayingSource,
    NowPlayingInfo,
    _select_now_playing,
    _SmtcCandidate,
    resolve_now_playing,
)


# AC4: track correctly identified via now-playing metadata -> identity waterfall.
def test_resolve_now_playing_walks_the_identity_waterfall() -> None:
    source = InMemoryNowPlayingSource(
        NowPlayingInfo(
            title="Around the World",
            artist="Daft Punk",
            app_id="Spotify.exe",
            process_id=4242,
            spotify_id="1pKYYY0dkg23sQQXi0Q5zN",
        )
    )
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"FRDM19700001": "mbid-around-the-world"}),
        spotify_source=InMemorySpotifyIsrcSource({"1pKYYY0dkg23sQQXi0Q5zN": "FRDM19700001"}),
    )

    identified = resolve_now_playing(source, resolver)

    assert identified is not None
    now_playing, identity = identified
    assert now_playing.process_id == 4242
    assert identity.resolved
    assert identity.mbid == "mbid-around-the-world"
    assert identity.name == "Around the World"
    assert identity.artist == "Daft Punk"


def test_resolve_now_playing_none_when_nothing_playing() -> None:
    source = InMemoryNowPlayingSource(None)
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    assert resolve_now_playing(source, resolver) is None


# #138 AC1/AC2/AC4: session selection now enumerates every SMTC session (not
# just get_current_session()'s single guess), filters to the app-id
# allowlist, and picks the Playing one deterministically.
def _candidate(
    *, title: str, artist: str, app_id: str, is_playing: bool, spotify_id: str | None = None
) -> _SmtcCandidate:
    return _SmtcCandidate(
        info=NowPlayingInfo(title=title, artist=artist, app_id=app_id, spotify_id=spotify_id),
        is_playing=is_playing,
    )


def test_select_now_playing_finds_playing_session_behind_non_media_foreground_app() -> None:
    # Spotify is Playing in the background; some unrelated non-media app (not
    # even SMTC-visible, so not modeled here) holds OS foreground focus.
    # get_sessions() must still surface the Spotify session (get_current_session()
    # would not, since it guesses based on the OS's own foreground notion).
    candidates = [
        _candidate(title="Discovery", artist="Daft Punk", app_id="Spotify.exe", is_playing=True),
    ]
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.title == "Discovery"


def test_select_now_playing_drops_unresolvable_browser_session() -> None:
    # A browser tab playing something that isn't identifiable through the
    # waterfall (e.g. a podcast, a call) must never be captured — a browser's
    # AUMID can't distinguish it from a music tab, so resolvability is the gate.
    candidates = [
        _candidate(title="Episode 42", artist="Some Podcast", app_id="chrome.exe", is_playing=True),
    ]
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    assert _select_now_playing(candidates, resolver=resolver) is None


def test_select_now_playing_admits_resolvable_browser_session() -> None:
    # A browser session IS admitted once its track resolves past bare-name level.
    candidates = [
        _candidate(
            title="Around the World",
            artist="Daft Punk",
            app_id="chrome.exe",
            is_playing=True,
            spotify_id="1pKYYY0dkg23sQQXi0Q5zN",
        ),
    ]
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"FRDM19700001": "mbid-around-the-world"}),
        spotify_source=InMemorySpotifyIsrcSource({"1pKYYY0dkg23sQQXi0Q5zN": "FRDM19700001"}),
    )

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.title == "Around the World"


def test_select_now_playing_ignores_non_allowlisted_app() -> None:
    candidates = [
        _candidate(title="Some Notification", artist="", app_id="explorer.exe", is_playing=True),
    ]
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    assert _select_now_playing(candidates, resolver=resolver) is None


def test_select_now_playing_ignores_non_playing_session() -> None:
    candidates = [
        _candidate(title="Discovery", artist="Daft Punk", app_id="Spotify.exe", is_playing=False),
    ]
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    assert _select_now_playing(candidates, resolver=resolver) is None


def test_select_now_playing_prefers_spotify_when_multiple_sessions_playing() -> None:
    # Deterministic tie-break: Spotify wins over any other allowlisted app
    # when both are simultaneously Playing (#138 AC4) — no recency heuristic,
    # SMTC exposes no reliable activity timestamp.
    candidates = [
        _candidate(
            title="Episode 1",
            artist="Podcast",
            app_id="chrome.exe",
            is_playing=True,
            spotify_id="track-2",
        ),
        _candidate(title="Discovery", artist="Daft Punk", app_id="Spotify.exe", is_playing=True),
    ]
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"ISRC2": "mbid-2"}),
        spotify_source=InMemorySpotifyIsrcSource({"track-2": "ISRC2"}),
    )

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.app_id == "Spotify.exe"


def test_select_now_playing_first_playing_when_no_spotify_among_ties() -> None:
    # No recency signal exists, so when two non-Spotify allowlisted sessions
    # are both Playing, enumeration order breaks the tie.
    candidates = [
        _candidate(
            title="First",
            artist="A",
            app_id="chrome.exe",
            is_playing=True,
            spotify_id="track-a",
        ),
        _candidate(
            title="Second",
            artist="B",
            app_id="msedge.exe",
            is_playing=True,
            spotify_id="track-b",
        ),
    ]
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"ISRC_A": "mbid-a", "ISRC_B": "mbid-b"}),
        spotify_source=InMemorySpotifyIsrcSource({"track-a": "ISRC_A", "track-b": "ISRC_B"}),
    )

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.title == "First"


# AC1: per-process capture needs the right PID scoped to the target app.
@dataclass
class _FakeProcInfo:
    pid: int
    name: str | None
    ppid: int | None = None


class _PsutilError(Exception):
    pass


class _FakePsutilProcess:
    def __init__(self, ppid: int | None) -> None:
        self._ppid = ppid

    def ppid(self) -> int:
        if self._ppid is None:
            raise _PsutilError("no parent")
        return self._ppid


def _fake_psutil(monkeypatch: pytest.MonkeyPatch, procs: list[_FakeProcInfo]) -> None:
    class _Proc:
        def __init__(self, info: _FakeProcInfo) -> None:
            self.info = {"pid": info.pid, "name": info.name}
            self._ppid = info.ppid

    class _FakePsutilModule:
        Error = _PsutilError

        @staticmethod
        def process_iter(_fields):
            return [_Proc(p) for p in procs]

        @staticmethod
        def Process(pid):
            for p in procs:
                if p.pid == pid:
                    return _FakePsutilProcess(p.ppid)
            raise _PsutilError(f"no such pid {pid}")

    monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutilModule)


def test_process_id_for_app_strips_exe_extension_and_matches_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_psutil(
        monkeypatch,
        [_FakeProcInfo(pid=100, name="Spotify.exe", ppid=1)],
    )

    pid = np_module._process_id_for_app("SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify")

    assert pid == 100


def test_process_id_for_app_picks_ancestor_most_process_among_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three sibling Spotify.exe processes (typical of the UWP-packaged build):
    # 100 is the root (its parent, 1, is not itself a match), 200 and 300 are
    # its descendants. Only the ancestor-most match maximizes process-tree
    # coverage for PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE.
    _fake_psutil(
        monkeypatch,
        [
            _FakeProcInfo(pid=200, name="Spotify.exe", ppid=100),
            _FakeProcInfo(pid=100, name="Spotify.exe", ppid=1),
            _FakeProcInfo(pid=300, name="Spotify.exe", ppid=100),
        ],
    )

    pid = np_module._process_id_for_app("Spotify")

    assert pid == 100


def test_process_id_for_app_none_when_no_match(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_psutil(monkeypatch, [_FakeProcInfo(pid=1, name="explorer.exe", ppid=None)])

    assert np_module._process_id_for_app("Spotify") is None


def test_process_id_for_app_skips_access_denied_processes_with_none_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # psutil.process_iter(["pid", "name"]) sets "name" to None (not a missing
    # key) for protected processes it can't read (AccessDenied/ZombieProcess --
    # e.g. PID 4 "System", "Registry", "Secure System" are always present on
    # Windows). Path(None) must not be reached.
    _fake_psutil(
        monkeypatch,
        [
            _FakeProcInfo(pid=4, name=None, ppid=0),
            _FakeProcInfo(pid=100, name="Spotify.exe", ppid=1),
        ],
    )

    pid = np_module._process_id_for_app("Spotify")

    assert pid == 100
