"""Replay queue file + resume from the journal (#277)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from music_intel_mcp.capture import AudioFrame
from music_intel_mcp.cli import main
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.replay_capture import (
    ReplayJournalEntry,
    append_replay_journal_entry,
    process_replay_queue,
)
from music_intel_mcp.replay_queue import read_replay_queue_file, write_replay_queue_file
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore


class _Capture:
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read(self, duration_s: float) -> AudioFrame:
        t = np.arange(800) / 16000
        tone = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32).reshape(-1, 1)
        return AudioFrame(samples=tone, sample_rate=16000)


class _Driver:
    def __init__(self) -> None:
        self.played: list[TrackRef] = []

    def play(self, track: TrackRef) -> None:
        self.played.append(track)

    def pause(self) -> None: ...


def _t(n: int, **ids) -> TrackRef:
    return TrackRef(name=f"T{n}", artist="A", **ids)


def _journal(path: Path, track: TrackRef, outcome: str) -> None:
    append_replay_journal_entry(
        path,
        ReplayJournalEntry(
            track_id=canonical_track_id(track),
            outcome=outcome,
            started_at="2026-01-01T00:00:00+00:00",
            ended_at="2026-01-01T00:00:01+00:00",
        ),
    )


def _analyse(store: UserStore, key: str) -> None:
    store.write_audio_analysis(
        track_id=key, embedding=np.array([0.1], dtype=np.float32), tags={"x": 0.5}
    )


def _run(store, queue, journal, driver, *, resume=True):
    return process_replay_queue(
        capture_route="per_app_output",
        queue=queue,
        track_duration_s=lambda t: 0.05,
        capture=_Capture(),
        driver=driver,
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"g": 0.9})),
        max_window_s=0.05,
        journal_path=journal,
        resume=resume,
        max_silence_wait_s=0.02,
        poll_interval_s=0.01,
    )


def test_queue_file_round_trips_tracks_in_order(tmp_path) -> None:
    tracks = [
        _t(3, spotify_id="s3"),
        _t(1, mbid="m1", isrc="I1"),
        _t(2, youtube_id="y2"),
        _t(0),
    ]
    path = tmp_path / "sub" / "q.jsonl"

    assert write_replay_queue_file(path, tracks) == 4

    assert read_replay_queue_file(path) == tracks
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4


def test_resume_skips_ok_and_twice_failed_but_not_once_failed(tmp_path) -> None:
    store = UserStore(root=tmp_path)
    journal = tmp_path / "j.jsonl"
    ok, twice_silent, silent_short, once_silent, fresh = (
        _t(1, spotify_id="a"),
        _t(2, spotify_id="b"),
        _t(3, spotify_id="c"),
        _t(4, spotify_id="d"),
        _t(5, spotify_id="e"),
    )
    _journal(journal, ok, "ok")
    for o in ("silent", "requeued", "silent"):
        _journal(journal, twice_silent, o)
    for o in ("silent", "short"):
        _journal(journal, silent_short, o)
    _journal(journal, once_silent, "silent")
    driver = _Driver()

    _run(store, [ok, twice_silent, silent_short, once_silent, fresh], journal, driver)

    assert [t.spotify_id for t in driver.played] == ["d", "e"]


def test_without_resume_journal_does_not_filter(tmp_path) -> None:
    store = UserStore(root=tmp_path)
    journal = tmp_path / "j.jsonl"
    track = _t(1, spotify_id="a")
    _journal(journal, track, "ok")
    driver = _Driver()

    _run(store, [track], journal, driver, resume=False)

    assert driver.played == [track]


def test_track_analysed_under_spotify_or_mbid_key_is_not_played(tmp_path) -> None:
    store = UserStore(root=tmp_path)
    by_spotify = _t(1, spotify_id="sp1")
    by_mbid = _t(2, spotify_id="sp2", mbid="mb2")  # canonical key is mbid:
    by_spotify_with_mbid = _t(3, spotify_id="sp3", mbid="mb3")  # record under spotify: key
    fresh = _t(4, spotify_id="sp4")
    _analyse(store, "spotify:sp1")
    _analyse(store, "mbid:mb2")
    _analyse(store, "spotify:sp3")
    driver = _Driver()

    results = _run(
        store, [by_spotify, by_mbid, by_spotify_with_mbid, fresh], None, driver, resume=False
    )

    assert driver.played == [fresh]
    assert [r.outcome for r in results] == ["ok"]


def test_analysis_appearing_mid_run_is_seen_before_the_next_play(tmp_path) -> None:
    store = UserStore(root=tmp_path)
    first, second = _t(1, spotify_id="a"), _t(2, spotify_id="b")

    class _AnalysingDriver(_Driver):
        def play(self, track: TrackRef) -> None:
            super().play(track)
            _analyse(store, "spotify:b")  # e.g. another process finished track b

    driver = _AnalysingDriver()
    _run(store, [first, second], None, driver, resume=False)

    assert driver.played == [first]


def test_cli_replay_queue_writes_queue_file(tmp_path, capsys) -> None:
    events = [
        ListenEvent(
            track=TrackRef(name=f"N{n}", artist="Ar", spotify_id=f"s{n}"),
            played_at=f"2026-01-0{i}T00:00:00Z",
            source="test",
            context=PlayContext(ms_played=180_000),
        )
        for n in (1, 2)
        for i in (1, 2, 3)
    ]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8"
    )
    qfile = tmp_path / "out" / "queue.jsonl"

    rc = main(["replay-queue", "--data-dir", str(tmp_path), "--queue-file", str(qfile)])

    assert rc == 0
    assert str(qfile) in capsys.readouterr().out
    rows = [json.loads(line) for line in qfile.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["spotify_id"] for r in rows) == ["s1", "s2"]

    main(["replay-queue", "--data-dir", str(tmp_path)])
    assert (tmp_path / "replay_queue.jsonl").exists()
