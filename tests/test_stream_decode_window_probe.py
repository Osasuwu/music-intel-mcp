"""Pre-pilot whole-track-vs-30s stream-decode bias probe (#201).

#169 measured whether the loopback leg's 120 s capture window biases
embeddings relative to a 30 s window. #201 asks the analogous question for
the *other* pilot leg -- YouTube stream decode (#170) never captures a fixed
120 s window at all; it decodes the whole track. Decision (recorded via
``record_decision``, #201): the comparison here is exactly two points --
whole-track embedding (the decode's existing inference embedding, zero extra
decode) vs a 30 s truncation -- reusing ``window_probe.py``'s leg-agnostic
machinery unchanged rather than adding a third 120 s point that would need
its own extra decode/inference pass and would blur this leg's numbers with
#169's 120 s-anchored ones (AC6).
"""

from __future__ import annotations

import numpy as np
import pytest

from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import TrackRef
from music_intel_mcp.store import UserStore
from music_intel_mcp.stream_decode import (
    FakeStreamDecodeSource,
    VideoUnavailableError,
    process_stream_decode_queue,
    run_stream_decode_capture,
)
from music_intel_mcp.window_probe import (
    build_window_probe_report,
    load_window_pairs,
    make_window_probe_recorder,
    render_window_probe_report,
    stream_decode_window_probe_path,
)

# --- AC1: the observer hook fires only on the accepted "ok" outcome ------ #


def test_on_capture_analyzed_fires_on_an_accepted_capture(tmp_path):
    store = UserStore(root=tmp_path)
    journal_path = tmp_path / "journal.jsonl"
    vector = np.array([1.0, 2.0, 3.0])
    calls = []

    def recorder(**kwargs):
        calls.append(kwargs)

    result = run_stream_decode_capture(
        track_id="youtube:abc123",
        youtube_id="abc123",
        source=FakeStreamDecodeSource(channels=2),
        embedding_model=InMemoryEmbeddingModel(vector),
        classifier=InMemoryClassifier(ClassifierResult(tags={"genre---rock": 0.9})),
        store=store,
        journal_path=journal_path,
        on_capture_analyzed=recorder,
    )

    assert result.outcome == "ok"
    assert len(calls) == 1
    call = calls[0]
    assert call["track_id"] == "youtube:abc123"
    assert call["sample_rate"] == FakeStreamDecodeSource().sample_rate
    assert isinstance(call["pcm"], np.ndarray)
    assert list(call["embedding"]) == list(vector)
    assert call["tags"] == {"genre---rock": 0.9}


def test_on_capture_analyzed_does_not_fire_on_a_skipped_track(tmp_path):
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="youtube:known", embedding=np.array([1.0]), tags={})
    calls = []

    result = run_stream_decode_capture(
        track_id="youtube:known",
        youtube_id="known",
        source=FakeStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        on_capture_analyzed=lambda **kw: calls.append(kw),
    )

    assert result.outcome == "skipped"
    assert calls == []


class _UnavailableSource:
    def decode(self, youtube_id: str):
        raise VideoUnavailableError(f"{youtube_id}: gone")


def test_on_capture_analyzed_does_not_fire_on_an_unavailable_video(tmp_path):
    store = UserStore(root=tmp_path)
    calls = []

    result = run_stream_decode_capture(
        track_id="youtube:gone",
        youtube_id="gone",
        source=_UnavailableSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        on_capture_analyzed=lambda **kw: calls.append(kw),
    )

    assert result.outcome == "unavailable"
    assert calls == []


def test_on_capture_analyzed_is_optional_and_defaults_to_off(tmp_path):
    """Additive, default-off (AC1) -- omitting the hook must not change
    behaviour or raise."""
    store = UserStore(root=tmp_path)

    result = run_stream_decode_capture(
        track_id="youtube:noop",
        youtube_id="noop",
        source=FakeStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
    )

    assert result.outcome == "ok"


def test_process_stream_decode_queue_passes_the_hook_through(tmp_path):
    store = UserStore(root=tmp_path)
    calls = []
    queue = [TrackRef(youtube_id="abc123", name="t", artist="a")]

    process_stream_decode_queue(
        queue=queue,
        source=FakeStreamDecodeSource(),
        embedding_model=InMemoryEmbeddingModel(np.array([1.0, 2.0])),
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        on_capture_analyzed=lambda **kw: calls.append(kw),
    )

    assert len(calls) == 1


# --- AC2: reuse make_window_probe_recorder, one extra inference pass ----- #


def test_make_window_probe_recorder_rides_stream_decode_capture_with_no_extra_decode(tmp_path):
    store = UserStore(root=tmp_path)
    path = stream_decode_window_probe_path(store)
    embedding_model = InMemoryEmbeddingModel(np.array([1.0, 0.0]))
    recorder = make_window_probe_recorder(store=store, embedding_model=embedding_model, path=path)
    source = FakeStreamDecodeSource(duration_s=2.0)

    result = run_stream_decode_capture(
        track_id="youtube:abc123",
        youtube_id="abc123",
        source=source,
        embedding_model=embedding_model,
        classifier=InMemoryClassifier(ClassifierResult(tags={})),
        store=store,
        on_capture_analyzed=recorder,
    )

    assert result.outcome == "ok"
    assert source.decoded_youtube_ids == ["abc123"]  # exactly one decode
    pairs = load_window_pairs(path)
    assert len(pairs) == 1
    assert pairs[0].track_id == "youtube:abc123"
    assert pairs[0].long_window_s == pytest.approx(2.0, abs=1e-3)
    assert pairs[0].short_window_s == pytest.approx(2.0, abs=1e-3)  # shorter than 30s window


def test_stream_decode_window_probe_path_is_separate_from_the_loopback_one(tmp_path):
    from music_intel_mcp.window_probe import window_probe_path

    store = UserStore(root=tmp_path)
    assert stream_decode_window_probe_path(store) != window_probe_path(store)


# --- AC6/AC7: the report distinguishes this leg from #169's loopback one - #


def test_stream_decode_report_title_and_labels_are_distinct_from_the_loopback_leg():
    from tests.test_window_probe import _pair  # reuse the test fixture helper

    pairs = [_pair(f"t{i}", [1.0, 0.0], [1.0, 0.0]) for i in range(100)]
    report = build_window_probe_report(pairs)

    loopback_text = render_window_probe_report(report)
    stream_text = render_window_probe_report(
        report,
        title="whole-track vs 30 s stream-decode probe (#201)",
        long_label="whole-track leg",
        short_label="30 s leg",
        capture_noun="decode",
    )

    assert loopback_text != stream_text
    assert "120 s" in loopback_text
    assert "120 s" not in stream_text
    assert "whole-track leg" in stream_text
    assert "#201" in stream_text
    assert "#169" not in stream_text


def test_stream_decode_report_default_rendering_is_unchanged_for_the_loopback_leg():
    """No regression: calling render_window_probe_report with no new kwargs
    must render byte-identical to before this issue's changes."""
    pairs = [
        __import__("tests.test_window_probe", fromlist=["_pair"])._pair(
            f"t{i}", [1.0, 0.0], [1.0, 0.0]
        )
        for i in range(100)
    ]
    report = build_window_probe_report(pairs)
    text = render_window_probe_report(report)

    assert text.startswith("120 s vs 30 s capture-window probe (#169)")
    assert "one capture" in text
    assert "capture-to-capture" in text
