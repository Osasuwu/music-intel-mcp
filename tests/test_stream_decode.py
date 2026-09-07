"""YouTube stream-decode capture (#170 AC1).

Pure in-memory tests against ``FakeStreamDecodeSource`` (mirrors
``test_capture.py``'s convention for ``LoopbackSource``/``FakeLoopbackCapture``).
The real ``YtDlpStreamDecodeSource`` backend is exercised only in the live
smoke session -- same caveat as ``WasapiProcessLoopbackCapture``
(``capture.py``) and the ONNX model classes (``inference.py``) -- except for
its optional-dependency-absence branch, which *is* unit-tested here via a
simulated ``ImportError`` (no real yt-dlp call happens either way).
"""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import TrackRef
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore
from music_intel_mcp.stream_decode import (
    FakeStreamDecodeSource,
    VideoUnavailableError,
    YtDlpNotInstalledError,
    YtDlpStreamDecodeSource,
    decode_and_run_inference,
    journaled_unavailable_track_ids,
    process_stream_decode_queue,
    run_stream_decode_capture,
)


def test_fake_stream_decode_source_yields_a_stereo_audio_frame():
    source = FakeStreamDecodeSource()

    frame = source.decode("yt-abc123")

    assert frame.samples.ndim == 2
    assert frame.samples.shape[1] == 2
    assert frame.samples.dtype == np.float32


# --- AC1: the decode -> run_inference seam must not downmix -------------- #
# run_inference's own frontend (inference._mel_patches) already reduces
# stereo to mono internally -- the seam this orchestration owns must hand it
# the stereo PCM unreduced, not pre-mix it away before the call.


class _ChannelRecordingEmbeddingModel:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector
        self.received_pcm: np.ndarray | None = None

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        self.received_pcm = pcm
        return self._vector


def test_decode_and_run_inference_feeds_run_inference_the_stereo_pcm_unreduced():
    source = FakeStreamDecodeSource(channels=2)
    embedding_model = _ChannelRecordingEmbeddingModel(np.array([1.0, 2.0]))
    classifier = InMemoryClassifier(ClassifierResult(tags={"genre---rock": 0.9}))

    decode_and_run_inference(
        source, "yt-abc123", embedding_model=embedding_model, classifier=classifier
    )

    assert embedding_model.received_pcm is not None
    assert embedding_model.received_pcm.ndim == 2
    assert embedding_model.received_pcm.shape[1] == 2


def test_decode_and_run_inference_returns_the_inference_result():
    source = FakeStreamDecodeSource(channels=2)
    vector = np.array([1.0, 2.0, 3.0])
    embedding_model = InMemoryEmbeddingModel(vector)
    tags = {"genre---rock": 0.9}
    classifier = InMemoryClassifier(ClassifierResult(tags=tags))

    result = decode_and_run_inference(
        source, "yt-abc123", embedding_model=embedding_model, classifier=classifier
    )

    assert np.array_equal(result.embedding, vector)
    assert result.tags == tags


# --- AC1: yt-dlp is optional -- its absence degrades the branch rather ---- #
# than breaking the package. The module itself must stay importable without
# yt-dlp installed (proven simply by every test above collecting/running
# without yt-dlp on this machine); only calling decode() without it should
# fail, with a clear, catchable error rather than a raw ImportError leaking
# out of yt-dlp's internals.


def test_ytdlp_source_degrades_with_a_clear_error_when_ytdlp_is_not_installed(monkeypatch):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "yt_dlp" or name.startswith("yt_dlp."):
            raise ImportError("No module named 'yt_dlp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    source = YtDlpStreamDecodeSource()

    with pytest.raises(YtDlpNotInstalledError):
        source.decode("yt-abc123")


# --- code review on PR #196: an ffmpeg decode failure after a successful --- #
# yt-dlp extraction was previously uncaught -- it must degrade the same way
# an extractor failure does (VideoUnavailableError), not crash the batch.


class _FakeYoutubeDL:
    def __init__(self, opts) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def extract_info(self, url, download=False):
        return {"url": "https://example.invalid/stream"}


def test_ytdlp_source_raises_video_unavailable_when_ffmpeg_decode_fails(monkeypatch):
    fake_yt_dlp = types.ModuleType("yt_dlp")
    fake_yt_dlp.YoutubeDL = _FakeYoutubeDL
    fake_utils = types.ModuleType("yt_dlp.utils")
    fake_utils.DownloadError = type("DownloadError", (Exception,), {})
    fake_yt_dlp.utils = fake_utils
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_yt_dlp)

    def _raise_called_process_error(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", _raise_called_process_error)
    source = YtDlpStreamDecodeSource()

    with pytest.raises(VideoUnavailableError):
        source.decode("yt-abc123")


def test_ytdlp_source_raises_video_unavailable_when_stream_is_truncated(monkeypatch):
    """code review on PR #196: a ``googlevideo`` stream URL that expires or
    drops mid-transfer can leave ffmpeg exiting 0 with a truncated stdout
    buffer -- one whose byte count isn't an exact multiple of
    ``channels * 4`` (float32). ``_decode_stream_to_pcm``'s
    ``raw.reshape(-1, channels)`` raises ``ValueError`` in that case, which
    must degrade through the same VideoUnavailableError contract as every
    other decode failure, not crash the batch."""
    fake_yt_dlp = types.ModuleType("yt_dlp")
    fake_yt_dlp.YoutubeDL = _FakeYoutubeDL
    fake_utils = types.ModuleType("yt_dlp.utils")
    fake_utils.DownloadError = type("DownloadError", (Exception,), {})
    fake_yt_dlp.utils = fake_utils
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_yt_dlp)

    class _TruncatedCompletedProcess:
        # 3 float32 samples (12 bytes) is not a multiple of channels=2 --
        # reshape(-1, 2) raises ValueError, simulating a stream cut short
        # mid-transfer that ffmpeg still exited 0 for.
        stdout = b"\x00" * 12

    def _return_truncated_stdout(cmd, **kwargs):
        return _TruncatedCompletedProcess()

    monkeypatch.setattr(subprocess, "run", _return_truncated_stdout)
    source = YtDlpStreamDecodeSource()

    with pytest.raises(VideoUnavailableError):
        source.decode("yt-abc123")


# --- AC2: one journal line per decode attempt; extractor failure for a ----- #
# removed/private/region-locked video journals outcome "unavailable" and is
# not re-queued; already-analyzed tracks are skipped without decoding.


class _UnavailableStreamDecodeSource:
    """Fake source simulating a removed/private/region-locked video -- the
    real ``YtDlpStreamDecodeSource`` raises the same ``VideoUnavailableError``
    from a caught ``yt_dlp.utils.DownloadError`` (not exercised here, same
    live-smoke-only caveat as the rest of the real backend)."""

    def decode(self, youtube_id: str) -> None:
        raise VideoUnavailableError(f"{youtube_id}: Video unavailable")


def test_run_stream_decode_capture_journals_ok_and_writes_analysis(tmp_path):
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    source = FakeStreamDecodeSource(channels=2)
    vector = np.array([1.0, 2.0, 3.0])
    embedding_model = InMemoryEmbeddingModel(vector)
    classifier = InMemoryClassifier(ClassifierResult(tags={"genre---rock": 0.9}))

    result = run_stream_decode_capture(
        track_id="youtube:abc123",
        youtube_id="abc123",
        source=source,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        journal_path=journal_path,
    )

    assert result.outcome == "ok"
    assert store.has_audio_analysis("youtube:abc123")
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["track_id"] == "youtube:abc123"
    assert entry["outcome"] == "ok"


def test_run_stream_decode_capture_journals_unavailable_on_extractor_failure(tmp_path):
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    source = _UnavailableStreamDecodeSource()
    classifier = InMemoryClassifier(ClassifierResult(tags={}))

    result = run_stream_decode_capture(
        track_id="youtube:gone",
        youtube_id="gone",
        source=source,
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=classifier,
        store=store,
        journal_path=journal_path,
    )

    assert result.outcome == "unavailable"
    assert not store.has_audio_analysis("youtube:gone")
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["track_id"] == "youtube:gone"
    assert entry["outcome"] == "unavailable"


def test_run_stream_decode_capture_skips_already_analyzed_track_without_decoding(tmp_path):
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="youtube:known", embedding=np.array([1.0]), tags={})
    journal_path = tmp_path / "journal.jsonl"

    class _SpyStreamDecodeSource:
        def __init__(self) -> None:
            self.called = False

        def decode(self, youtube_id: str):
            self.called = True
            raise AssertionError("decode should not be called for an already-analyzed track")

    source = _SpyStreamDecodeSource()

    result = run_stream_decode_capture(
        track_id="youtube:known",
        youtube_id="known",
        source=source,
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        journal_path=journal_path,
    )

    assert result.outcome == "skipped"
    assert source.called is False
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["outcome"] == "skipped"


# --- AC2: "not re-queued" -- a journaled-unavailable track_id must not be -- #
# offered again as a replay-queue candidate on a later run.


def test_journaled_unavailable_track_ids_reads_only_unavailable_outcomes(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    store = UserStore(root=tmp_path)
    run_stream_decode_capture(
        track_id="youtube:gone",
        youtube_id="gone",
        source=_UnavailableStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        journal_path=journal_path,
    )
    run_stream_decode_capture(
        track_id="youtube:ok1",
        youtube_id="ok1",
        source=FakeStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        journal_path=journal_path,
    )

    ids = journaled_unavailable_track_ids(journal_path)

    assert ids == {"youtube:gone"}


def test_journaled_unavailable_track_ids_empty_when_journal_missing(tmp_path):
    ids = journaled_unavailable_track_ids(tmp_path / "does_not_exist.jsonl")

    assert ids == set()


# --- AC9: one participant's queue runs end-to-end -- a batch driver that ---- #
# walks a list[TrackRef] queue through run_stream_decode_capture once per
# track, with no live API calls (FakeStreamDecodeSource only) and no
# requeue logic (unlike replay_capture.process_replay_queue's silent/short
# requeue for the loopback path -- AC2's "not re-queued" semantics for an
# "unavailable" outcome are enforced by excluding that track_id from a
# *future* queue selection, not by an in-loop retry here).


def test_process_stream_decode_queue_journals_and_writes_analysis_for_each_track(tmp_path):
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    queue = [
        TrackRef(name="Track One", artist="Artist A", youtube_id="yt1"),
        TrackRef(name="Track Two", artist="Artist B", youtube_id="yt2"),
    ]
    source = FakeStreamDecodeSource(channels=2)
    embedding_model = InMemoryEmbeddingModel(np.array([1.0, 2.0]))
    classifier = InMemoryClassifier(ClassifierResult(tags={"genre---rock": 0.9}))

    results = process_stream_decode_queue(
        queue=queue,
        source=source,
        embedding_model=embedding_model,
        classifier=classifier,
        store=store,
        journal_path=journal_path,
    )

    assert [r.outcome for r in results] == ["ok", "ok"]
    for track in queue:
        assert store.has_audio_analysis(canonical_track_id(track))
    lines = journal_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_process_stream_decode_queue_decodes_exactly_the_queued_ids_no_live_calls(tmp_path):
    store = UserStore(root=tmp_path)
    queue = [
        TrackRef(name="Track One", artist="Artist A", youtube_id="yt1"),
        TrackRef(name="Track Two", artist="Artist B", youtube_id="yt2"),
    ]
    source = FakeStreamDecodeSource(channels=2)

    process_stream_decode_queue(
        queue=queue,
        source=source,
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        journal_path=tmp_path / "journal.jsonl",
    )

    assert source.decoded_youtube_ids == ["yt1", "yt2"]


def test_process_stream_decode_queue_skips_tracks_with_no_youtube_id_without_journaling(tmp_path):
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    queue = [TrackRef(name="No Youtube", artist="Artist C", spotify_id="s1")]

    results = process_stream_decode_queue(
        queue=queue,
        source=FakeStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        journal_path=journal_path,
    )

    assert results == []
    assert not journal_path.exists()
