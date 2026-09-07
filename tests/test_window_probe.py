"""Pre-pilot 120 s-vs-30 s window-bias probe (#169).

Entirely against injected fakes — a length-sensitive
:class:`~music_intel_mcp.inference.AudioEmbeddingModel` double lets a test
prove *which* slice of PCM reached the model without touching onnxruntime,
following the Protocol+fake idiom of ``test_replay_capture.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from music_intel_mcp.capture import AudioFrame
from music_intel_mcp.inference import ClassifierResult, InMemoryClassifier, InMemoryEmbeddingModel
from music_intel_mcp.models import TrackRef
from music_intel_mcp.replay_capture import process_replay_queue, run_replay_capture
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import UserStore
from music_intel_mcp.window_probe import (
    MIN_PROBE_SAMPLE_SIZE,
    WindowPair,
    append_window_pair,
    build_window_probe_report,
    compare_cluster_assignments,
    embed_window_pair,
    load_window_pairs,
    make_window_probe_recorder,
    pair_cosine_distance,
    render_window_probe_report,
    summarize_distances,
    window_probe_path,
)


class _LengthSensitiveEmbedding:
    """Returns ``[duration_s, mean_amplitude]`` so a test can read back the
    duration and content of the buffer the model was actually handed."""

    def __init__(self) -> None:
        self.durations_s: list[float] = []

    def embed(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = pcm.mean(axis=1) if pcm.ndim == 2 else pcm
        duration_s = mono.shape[0] / sample_rate
        self.durations_s.append(duration_s)
        return np.array([duration_s, float(mono.mean())], dtype=float)


def _ramp_pcm(duration_s: float, *, sample_rate: int = 16000) -> np.ndarray:
    """A monotonically rising ramp — the mean over the first 30 s differs from
    the mean over the full 120 s, so a truncation that silently kept the whole
    buffer is visible in the embedding, not just in the duration."""
    n = int(duration_s * sample_rate)
    return np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)


def test_embed_window_pair_embeds_full_window_and_front_truncation():
    model = _LengthSensitiveEmbedding()

    pair = embed_window_pair(
        track_id="spotify:abc",
        pcm=_ramp_pcm(120.0),
        sample_rate=16000,
        embedding_model=model,
        short_window_s=30.0,
    )

    assert pair.track_id == "spotify:abc"
    assert pair.long_window_s == 120.0
    assert pair.short_window_s == 30.0
    # The long leg saw the whole buffer, the short leg only its first 30 s.
    assert model.durations_s == [120.0, 30.0]
    assert pair.long_embedding[0] == 120.0
    assert pair.short_embedding[0] == 30.0
    # Ramp means differ, so the short leg is genuinely a different signal.
    assert pair.short_embedding[1] < pair.long_embedding[1]


def _pair(track_id: str, long_vec: list[float], short_vec: list[float]) -> WindowPair:
    return WindowPair(
        track_id=track_id,
        long_window_s=120.0,
        short_window_s=30.0,
        long_embedding=long_vec,
        short_embedding=short_vec,
    )


def test_pair_cosine_distance_spans_identical_to_orthogonal():
    identical = _pair("a", [1.0, 0.0], [2.0, 0.0])  # same direction, different norm
    orthogonal = _pair("b", [1.0, 0.0], [0.0, 1.0])

    assert pair_cosine_distance(identical) == 0.0
    assert pair_cosine_distance(orthogonal) == 1.0


def test_summarize_distances_reports_the_per_track_distribution():
    # Distances 0.0, 0.0, 1.0, 1.0 by construction.
    pairs = [
        _pair("a", [1.0, 0.0], [1.0, 0.0]),
        _pair("b", [1.0, 0.0], [3.0, 0.0]),
        _pair("c", [1.0, 0.0], [0.0, 1.0]),
        _pair("d", [0.0, 1.0], [1.0, 0.0]),
    ]

    summary = summarize_distances(pairs)

    assert summary.n == 4
    assert summary.mean == 0.5
    assert summary.median == 0.5
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0
    assert summary.p90 == 1.0


def test_summarize_distances_on_no_pairs_is_honest_empty():
    summary = summarize_distances([])

    assert summary.n == 0
    assert summary.mean is None
    assert summary.maximum is None


def _clustered_pairs(short_vectors: list[list[float]]) -> list[WindowPair]:
    """Six tracks whose *long* embeddings sit in two tight, well-separated
    groups of three (HDBSCAN ``min_cluster_size=3`` finds exactly two); the
    short leg is supplied per test."""
    long_vectors = [
        [0.0, 0.0],
        [0.0, 0.1],
        [0.1, 0.0],
        [10.0, 10.0],
        [10.0, 10.1],
        [10.1, 10.0],
    ]
    return [
        _pair(f"t{i}", long_vec, short_vec)
        for i, (long_vec, short_vec) in enumerate(zip(long_vectors, short_vectors, strict=True))
    ]


def test_cluster_agreement_is_perfect_when_both_windows_cluster_alike():
    # Short leg reproduces the same two groups (shifted, but same partition).
    pairs = _clustered_pairs(
        [[1.0, 1.0], [1.0, 1.1], [1.1, 1.0], [20.0, 20.0], [20.0, 20.1], [20.1, 20.0]]
    )

    agreement = compare_cluster_assignments(pairs, min_cluster_size=3)

    assert agreement.n_tracks == 6
    assert agreement.long_cluster_count == 2
    assert agreement.short_cluster_count == 2
    assert agreement.adjusted_rand_index == 1.0


def test_cluster_agreement_drops_when_the_short_window_regroups_tracks():
    # Short leg splits the groups across the two clusters — same cluster count,
    # different membership, which is exactly the failure the gate looks for.
    pairs = _clustered_pairs(
        [[1.0, 1.0], [1.0, 1.1], [20.0, 20.0], [1.1, 1.0], [20.0, 20.1], [20.1, 20.0]]
    )

    agreement = compare_cluster_assignments(pairs, min_cluster_size=3)

    assert agreement.adjusted_rand_index is not None
    assert agreement.adjusted_rand_index < 1.0


def test_cluster_agreement_without_clusters_reports_no_index():
    # Two tracks cannot form a min_cluster_size=3 cluster on either leg, so
    # there is no partition to compare — honest-empty, not a spurious 1.0.
    pairs = [_pair("a", [1.0, 0.0], [1.0, 0.0]), _pair("b", [0.0, 1.0], [0.0, 1.0])]

    agreement = compare_cluster_assignments(pairs, min_cluster_size=3)

    assert agreement.long_cluster_count == 0
    assert agreement.short_cluster_count == 0
    assert agreement.adjusted_rand_index is None


def test_cluster_agreement_does_not_treat_noise_as_a_shared_cluster():
    """Both legs find two clusters and leave three tracks as noise — but they
    disagree about *which* three. Collapsing HDBSCAN noise into one pseudo-
    cluster would make these two partitions identical (a spurious ARI of 1.0);
    scoring each noise track as its own singleton keeps the disagreement
    visible, which is the whole point of the gate."""
    tight_a = [[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]]
    tight_b = [[10.0, 10.0], [10.0, 10.1], [10.1, 10.0]]
    scattered = [[50.0, -50.0], [-60.0, 40.0], [70.0, 80.0]]

    long_vectors = tight_a + tight_b + scattered
    short_vectors = scattered + tight_b + tight_a  # noise set moves from t6-t8 to t0-t2
    pairs = [
        _pair(f"t{i}", long_vec, short_vec)
        for i, (long_vec, short_vec) in enumerate(zip(long_vectors, short_vectors, strict=True))
    ]

    agreement = compare_cluster_assignments(pairs, min_cluster_size=3)

    assert agreement.long_cluster_count == 2
    assert agreement.short_cluster_count == 2
    assert agreement.long_noise == 3
    assert agreement.short_noise == 3
    assert agreement.adjusted_rand_index is not None
    assert agreement.adjusted_rand_index < 0.9


def _many_pairs(n: int) -> list[WindowPair]:
    return [_pair(f"t{i}", [1.0, 0.0], [1.0, 0.0]) for i in range(n)]


def test_report_below_the_minimum_sample_is_not_a_passed_gate():
    """AC1 asks for >=100 tracks. A 12-track run must report itself as an
    under-powered sample rather than quietly presenting its statistics as the
    gate result -- unmeasured must never read as passed."""
    report = build_window_probe_report(_many_pairs(12))

    assert report.n_tracks == 12
    assert report.min_sample_size == MIN_PROBE_SAMPLE_SIZE == 100
    assert report.sample_size_ok is False
    assert "under-powered" in render_window_probe_report(report).lower()


def test_report_at_the_minimum_sample_meets_the_gate():
    report = build_window_probe_report(_many_pairs(MIN_PROBE_SAMPLE_SIZE))

    assert report.sample_size_ok is True
    assert report.distances.n == MIN_PROBE_SAMPLE_SIZE


def test_rendered_report_states_the_single_capture_limitation():
    """The probe truncates one capture instead of capturing twice, so its
    distances exclude capture-to-capture variance. The rendered report is what
    the owner judges the gate on, so the caveat has to travel with it."""
    text = render_window_probe_report(build_window_probe_report(_many_pairs(100)))

    assert "one capture" in text
    assert "capture-to-capture" in text
    assert "noise floor" in text


def test_window_pairs_round_trip_through_the_journal(tmp_path):
    store = UserStore(root=tmp_path)
    path = window_probe_path(store)
    written = [
        _pair("a", [1.0, 0.0], [0.9, 0.1]),
        _pair("b", [0.0, 1.0], [0.1, 0.9]),
    ]
    for pair in written:
        append_window_pair(path, pair)

    assert load_window_pairs(path) == written


def test_loading_an_absent_journal_is_honest_empty(tmp_path):
    assert load_window_pairs(window_probe_path(UserStore(root=tmp_path))) == []


def test_probe_journal_lives_under_the_gitignored_data_root(tmp_path):
    """AC4: no captured audio or embeddings may be committed. Embeddings only
    ever land under ``UserStore.root`` (the gitignored ``data/`` tree), never
    anywhere inside the repo -- so the write path is asserted, not assumed."""
    store = UserStore(root=tmp_path)
    path = window_probe_path(store)
    append_window_pair(path, _pair("a", [1.0, 0.0], [0.9, 0.1]))

    assert path.parent == store.root
    assert path.exists()
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root not in path.resolve().parents


def _loud_second(sample_rate: int = 16000) -> np.ndarray:
    """One second of steady 0.2-amplitude signal — comfortably above
    ``RMS_SILENCE_THRESHOLD``, so the replay contract accepts the capture."""
    return (0.2 * np.ones((sample_rate, 1))).astype(np.float32)


class _FrameListCapture:
    """Replays a fixed list of frames, one per ``read()``, then empties out."""

    def __init__(self, frames: list[np.ndarray], sample_rate: int = 16000) -> None:
        self._frames = [AudioFrame(samples=f, sample_rate=sample_rate) for f in frames]
        self._sample_rate = sample_rate

    def start(self) -> None: ...

    def read(self, duration_s: float) -> AudioFrame:
        if self._frames:
            return self._frames.pop(0)
        return AudioFrame(samples=np.zeros((0, 1), dtype=np.float32), sample_rate=self._sample_rate)

    def stop(self) -> None: ...


class _NullDriver:
    def play(self, track: TrackRef) -> None: ...

    def pause(self) -> None: ...


def test_replay_capture_feeds_the_probe_recorder_without_a_second_capture(tmp_path):
    """AC1/AC4 wiring: the probe rides on an ordinary replay session through
    ``run_replay_capture``'s ``on_capture_analyzed`` hook — the same PCM buffer
    the pilot embeds, truncated, so no extra replay hour is spent and the pair
    lands in the gitignored data root."""
    store = UserStore(root=tmp_path)
    pcm = _loud_second()
    probe_model = _LengthSensitiveEmbedding()

    outcome = run_replay_capture(
        track=TrackRef(mbid="M-1", name="Around the World", artist="Daft Punk"),
        duration_s=1.0,
        capture=_FrameListCapture([pcm]),
        driver=_NullDriver(),
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={"genre---electronic": 0.9})),
        poll_interval_s=1.0,
        on_capture_analyzed=make_window_probe_recorder(
            store=store, embedding_model=probe_model, short_window_s=0.5
        ),
    )

    assert outcome.outcome == "ok"
    pairs = load_window_pairs(window_probe_path(store))
    assert len(pairs) == 1
    assert pairs[0].track_id == outcome.track_id
    assert pairs[0].long_window_s == 1.0
    assert pairs[0].short_window_s == 0.5
    # The long leg reuses the inference embedding already computed by the
    # replay path; only the short leg costs an extra pass.
    assert probe_model.durations_s == [0.5]
    assert pairs[0].long_embedding == pytest.approx([0.1, 0.2])


def test_replay_capture_without_a_probe_hook_writes_no_probe_journal(tmp_path):
    store = UserStore(root=tmp_path)
    pcm = _loud_second()

    run_replay_capture(
        track=TrackRef(mbid="M-1", name="Around the World", artist="Daft Punk"),
        duration_s=1.0,
        capture=_FrameListCapture([pcm]),
        driver=_NullDriver(),
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={})),
        poll_interval_s=1.0,
    )

    assert not window_probe_path(store).exists()


def test_a_discarded_capture_never_reaches_the_probe(tmp_path):
    """A capture that anchors on a brief blip and is then near-silent fails the
    replay contract's RMS gate and is discarded, not embedded. It must not reach
    the probe either: a rejected buffer in the journal would bias the very
    distribution the gate is read from."""
    store = UserStore(root=tmp_path)
    blip = (0.02 * np.ones((160, 1))).astype(np.float32)  # anchors, 0.01 s
    quiet = np.zeros((16000, 1), dtype=np.float32)
    probe_model = _LengthSensitiveEmbedding()

    outcome = run_replay_capture(
        track=TrackRef(mbid="M-1", name="Around the World", artist="Daft Punk"),
        duration_s=1.0,
        capture=_FrameListCapture([blip, quiet]),
        driver=_NullDriver(),
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={})),
        poll_interval_s=1.0,
        on_capture_analyzed=make_window_probe_recorder(
            store=store, embedding_model=probe_model, short_window_s=0.5
        ),
    )

    assert outcome.outcome == "silent"
    assert probe_model.durations_s == []
    assert load_window_pairs(window_probe_path(store)) == []


def test_the_probe_rides_through_the_whole_replay_queue(tmp_path):
    """The pilot drives ``process_replay_queue``, not single captures. The hook
    reaches every track of the queue through it, so a pilot session accumulates
    the >=100-track sample AC1 asks for without a separate probe run."""
    store = UserStore(root=tmp_path)
    pcm = _loud_second()
    tracks = [
        TrackRef(mbid="M-1", name="Around the World", artist="Daft Punk"),
        TrackRef(mbid="M-2", name="Da Funk", artist="Daft Punk"),
    ]
    queue = list(tracks)

    process_replay_queue(
        queue=queue,
        track_duration_s=lambda t: 1.0,
        capture=_FrameListCapture([pcm, pcm, pcm, pcm]),
        driver=_NullDriver(),
        store=store,
        embedding_model=InMemoryEmbeddingModel(vector=np.array([0.1, 0.2], dtype=np.float32)),
        classifier=InMemoryClassifier(result=ClassifierResult(tags={})),
        poll_interval_s=1.0,
        on_capture_analyzed=make_window_probe_recorder(
            store=store, embedding_model=_LengthSensitiveEmbedding(), short_window_s=0.5
        ),
    )

    assert [p.track_id for p in load_window_pairs(window_probe_path(store))] == [
        canonical_track_id(t) for t in tracks
    ]
