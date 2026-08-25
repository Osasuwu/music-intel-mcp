"""Continuous capture loop tests (#136) — track-change dedup and per-track
failure isolation, entirely against injected fakes."""

from __future__ import annotations

import threading

import numpy as np

from music_intel_mcp.capture import FakeLoopbackCapture
from music_intel_mcp.continuous_capture import run_continuous_capture
from music_intel_mcp.identity import IdentityResolver, InMemoryIsrcMbidIndex
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.nowplaying import NowPlayingInfo
from music_intel_mcp.store import UserStore

_TRACK_A = NowPlayingInfo(title="A", artist="Artist", app_id="Spotify.exe", process_id=111)
_TRACK_B = NowPlayingInfo(title="B", artist="Artist", app_id="Spotify.exe", process_id=222)


class _SequenceNowPlayingSource:
    """Returns each queued value in order, then repeats the last forever."""

    def __init__(self, sequence: list[NowPlayingInfo | None]) -> None:
        self._sequence = sequence
        self._i = 0

    def current(self) -> NowPlayingInfo | None:
        value = self._sequence[min(self._i, len(self._sequence) - 1)]
        self._i += 1
        return value


def _make_deps(tmp_path):
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())
    embedding_model = InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32))
    classifier = InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9}))
    store = UserStore(root=tmp_path)
    return resolver, embedding_model, classifier, store


def _stopping_sleep(stop_event: threading.Event, after: int):
    calls = {"n": 0}

    def sleep(_seconds: float) -> None:
        calls["n"] += 1
        if calls["n"] >= after:
            stop_event.set()

    return sleep


def test_same_track_polled_twice_captures_once(tmp_path) -> None:
    resolver, embedding_model, classifier, store = _make_deps(tmp_path)
    source = _SequenceNowPlayingSource([_TRACK_A, _TRACK_A, _TRACK_A])
    stop_event = threading.Event()
    captures: list[FakeLoopbackCapture] = []

    def capture_factory(_info: NowPlayingInfo) -> FakeLoopbackCapture:
        c = FakeLoopbackCapture(sample_rate=16000, channels=1)
        captures.append(c)
        return c

    run_continuous_capture(
        now_playing_source=source,
        identity_resolver=resolver,
        capture_factory=capture_factory,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        capture_duration_s=0.05,
        stop_event=stop_event,
        sleep=_stopping_sleep(stop_event, after=3),
    )

    assert len(captures) == 1


def test_track_change_captures_each_track(tmp_path) -> None:
    resolver, embedding_model, classifier, store = _make_deps(tmp_path)
    source = _SequenceNowPlayingSource([_TRACK_A, _TRACK_B])
    stop_event = threading.Event()
    captures: list[FakeLoopbackCapture] = []

    def capture_factory(_info: NowPlayingInfo) -> FakeLoopbackCapture:
        c = FakeLoopbackCapture(sample_rate=16000, channels=1)
        captures.append(c)
        return c

    run_continuous_capture(
        now_playing_source=source,
        identity_resolver=resolver,
        capture_factory=capture_factory,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        capture_duration_s=0.05,
        stop_event=stop_event,
        sleep=_stopping_sleep(stop_event, after=2),
    )

    assert len(captures) == 2


def test_nothing_playing_resets_dedup_and_does_not_capture(tmp_path) -> None:
    resolver, embedding_model, classifier, store = _make_deps(tmp_path)
    source = _SequenceNowPlayingSource([None, None])
    stop_event = threading.Event()
    captures: list[FakeLoopbackCapture] = []

    def capture_factory(_info: NowPlayingInfo) -> FakeLoopbackCapture:
        c = FakeLoopbackCapture()
        captures.append(c)
        return c

    run_continuous_capture(
        now_playing_source=source,
        identity_resolver=resolver,
        capture_factory=capture_factory,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        stop_event=stop_event,
        sleep=_stopping_sleep(stop_event, after=2),
    )

    assert captures == []


def test_capture_failure_is_isolated_and_loop_continues(tmp_path) -> None:
    resolver, embedding_model, classifier, store = _make_deps(tmp_path)
    source = _SequenceNowPlayingSource([_TRACK_A, _TRACK_B])
    stop_event = threading.Event()
    errors: list[tuple[NowPlayingInfo, Exception]] = []
    results: list[tuple[NowPlayingInfo, object]] = []

    class _ExplodingCapture:
        def start(self) -> None:
            raise RuntimeError("device unavailable")

        def read(self, duration_s: float):  # pragma: no cover - never reached
            raise AssertionError

        def stop(self) -> None:  # pragma: no cover - never reached
            raise AssertionError

    def capture_factory(info: NowPlayingInfo):
        if info is _TRACK_A:
            return _ExplodingCapture()
        return FakeLoopbackCapture(sample_rate=16000, channels=1)

    run_continuous_capture(
        now_playing_source=source,
        identity_resolver=resolver,
        capture_factory=capture_factory,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        capture_duration_s=0.05,
        stop_event=stop_event,
        on_error=lambda info, exc: errors.append((info, exc)),
        on_result=lambda info, result: results.append((info, result)),
        sleep=_stopping_sleep(stop_event, after=2),
    )

    assert len(errors) == 1
    assert errors[0][0] is _TRACK_A
    assert len(results) == 1
    assert results[0][0] is _TRACK_B


def test_unresolved_process_id_is_skipped(tmp_path) -> None:
    resolver, embedding_model, classifier, store = _make_deps(tmp_path)
    unresolved = NowPlayingInfo(title="C", artist="Artist", app_id="chrome.exe", process_id=None)
    source = _SequenceNowPlayingSource([unresolved])
    stop_event = threading.Event()
    captures: list[FakeLoopbackCapture] = []

    def capture_factory(_info: NowPlayingInfo) -> FakeLoopbackCapture:
        c = FakeLoopbackCapture()
        captures.append(c)
        return c

    run_continuous_capture(
        now_playing_source=source,
        identity_resolver=resolver,
        capture_factory=capture_factory,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        stop_event=stop_event,
        sleep=_stopping_sleep(stop_event, after=2),
    )

    assert captures == []
