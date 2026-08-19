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
from music_intel_mcp.nowplaying import InMemoryNowPlayingSource, NowPlayingInfo, resolve_now_playing


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


# AC1: per-process capture needs the right PID scoped to the target app.
@dataclass
class _FakeProcInfo:
    pid: int
    name: str
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
