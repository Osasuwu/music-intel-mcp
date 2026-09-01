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
from music_intel_mcp.live_identity import (
    InMemoryMusicBrainzNameSearchSource,
    LiveIdentityResolver,
)
from music_intel_mcp.nowplaying import (
    InMemoryNowPlayingSource,
    NowPlayingInfo,
    _app_stem,
    _classify_app,
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
    resolver = LiveIdentityResolver()

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
    resolver = LiveIdentityResolver()

    assert _select_now_playing(candidates, resolver=resolver) is None


def test_select_now_playing_admits_resolvable_browser_session() -> None:
    # A browser session IS admitted once its track resolves past bare-name level
    # via the live waterfall's text-only rungs (#145) — pre-capture there is no
    # fingerprint yet, so this exercises the mb_name rung, not AcoustID.
    candidates = [
        _candidate(
            title="Around the World",
            artist="Daft Punk",
            app_id="chrome.exe",
            is_playing=True,
        ),
    ]
    resolver = LiveIdentityResolver(
        mb_name_search=InMemoryMusicBrainzNameSearchSource(
            {("Around the World", "Daft Punk"): "mbid-around-the-world"}
        ),
    )

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.title == "Around the World"


# #145 AC2: the gate runs on every 5s poll, so a session-lifetime admission
# memo keyed by (title, artist, app_id) must keep the waterfall from being
# walked more than once per unique track per session.
def test_select_now_playing_admission_memo_avoids_repeat_resolver_calls() -> None:
    candidates = [
        _candidate(
            title="Around the World",
            artist="Daft Punk",
            app_id="chrome.exe",
            is_playing=True,
        ),
    ]
    mb_name_search = InMemoryMusicBrainzNameSearchSource(
        {("Around the World", "Daft Punk"): "mbid-around-the-world"}
    )
    resolver = LiveIdentityResolver(mb_name_search=mb_name_search)
    memo: dict[tuple[str, str, str], bool] = {}

    _select_now_playing(candidates, resolver=resolver, admission_memo=memo)
    _select_now_playing(candidates, resolver=resolver, admission_memo=memo)

    assert len(mb_name_search.calls) == 1


# #145 AC3: rejected browser candidates are a transparent rejection, not a
# silent drop — recorded via a caller-supplied sink so SmtcNowPlayingSource
# can persist it as a JSONL breadcrumb without _select_now_playing knowing
# about file I/O.
def test_select_now_playing_reports_drop_breadcrumb_on_rejection() -> None:
    candidates = [
        _candidate(title="Episode 42", artist="Some Podcast", app_id="chrome.exe", is_playing=True),
    ]
    resolver = LiveIdentityResolver()
    drops: list[NowPlayingInfo] = []

    result = _select_now_playing(candidates, resolver=resolver, on_drop=drops.append)

    assert result is None
    assert len(drops) == 1
    assert drops[0].title == "Episode 42"
    assert drops[0].artist == "Some Podcast"


# Review finding on PR #147 (#145): on_drop must fire once per fresh
# rejection, not on every poll of a memo-cached rejection — else
# smtc_drops.jsonl grows unbounded for a single persistently-rejected track.
def test_select_now_playing_reports_drop_once_for_memo_cached_rejection() -> None:
    candidates = [
        _candidate(title="Episode 42", artist="Some Podcast", app_id="chrome.exe", is_playing=True),
    ]
    resolver = LiveIdentityResolver()
    memo: dict[tuple[str, str, str], bool] = {}
    drops: list[NowPlayingInfo] = []

    _select_now_playing(candidates, resolver=resolver, admission_memo=memo, on_drop=drops.append)
    _select_now_playing(candidates, resolver=resolver, admission_memo=memo, on_drop=drops.append)

    assert len(drops) == 1


def test_select_now_playing_ignores_non_allowlisted_app() -> None:
    candidates = [
        _candidate(title="Some Notification", artist="", app_id="explorer.exe", is_playing=True),
    ]
    resolver = LiveIdentityResolver()

    assert _select_now_playing(candidates, resolver=resolver) is None


def test_select_now_playing_ignores_non_playing_session() -> None:
    candidates = [
        _candidate(title="Discovery", artist="Daft Punk", app_id="Spotify.exe", is_playing=False),
    ]
    resolver = LiveIdentityResolver()

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
        ),
        _candidate(title="Discovery", artist="Daft Punk", app_id="Spotify.exe", is_playing=True),
    ]
    resolver = LiveIdentityResolver(
        mb_name_search=InMemoryMusicBrainzNameSearchSource({("Episode 1", "Podcast"): "mbid-2"}),
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
        ),
        _candidate(
            title="Second",
            artist="B",
            app_id="msedge.exe",
            is_playing=True,
        ),
    ]
    resolver = LiveIdentityResolver(
        mb_name_search=InMemoryMusicBrainzNameSearchSource(
            {("First", "A"): "mbid-a", ("Second", "B"): "mbid-b"}
        ),
    )

    result = _select_now_playing(candidates, resolver=resolver)

    assert result is not None
    assert result.title == "First"


# Real-world multi-profile Chromium AUMIDs (e.g. "MSEdge.UserData.Profile2")
# have no "!" separator and multiple dot segments, so Path(name).stem only
# strips the last segment ("MSEdge.UserData") instead of collapsing to the
# bare process stem ("msedge"). Confirmed live: a real Edge SMTC session was
# silently dropped at classification and never reached the resolvability gate.
def test_app_stem_handles_multi_segment_browser_aumid() -> None:
    assert _app_stem("MSEdge.UserData.Profile2") == "msedge"


def test_classify_app_recognizes_multi_segment_edge_aumid_as_browser() -> None:
    assert _classify_app("MSEdge.UserData.Profile2") == "browser"


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


def test_process_id_for_app_matches_multi_segment_browser_aumid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: _process_id_for_app used to compute its own stem via
    # Path(name).stem, which only strips the LAST dot segment and left
    # "msedge.userdata" for a real multi-profile Edge AUMID -- matching no
    # real psutil process and silently disabling WASAPI capture for exactly
    # the browser sessions #145's admission-gate fix was meant to admit.
    _fake_psutil(
        monkeypatch,
        [_FakeProcInfo(pid=100, name="msedge.exe", ppid=1)],
    )

    pid = np_module._process_id_for_app("MSEdge.UserData.Profile2")

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
