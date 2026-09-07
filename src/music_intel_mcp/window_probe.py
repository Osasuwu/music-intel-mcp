"""Pre-pilot 120 s-vs-30 s capture-window bias probe (#169).

The pilot's replay capture contract uses a ``min(track duration, 120 s)``
window (#166, decision ``2e17aafa``) because it fits ~30 tracks/h instead of
~17. The MTG models behind ``inference.py`` were trained on short clips and
``_mel_patches`` mean-pools over whatever it is handed, so a 120 s window may
sit off the models' training distribution — unmeasured, and if it does bias
embeddings, pilot cluster-quality failures would be indistinguishable from
method failures (decision ``87277764``, runbook gate "Pre-pilot measurement
gate").

**Both legs come from one capture.** The 30 s leg is a front truncation of the
same RMS-anchored PCM buffer the 120 s leg embeds, not a second capture of the
same track. Two independent captures would vary in *content* (playback offset,
loudness state, loopback conditions) and confound the one variable under
measurement; truncating holds content fixed so the reported distance is a pure
window-length effect. The flip side, stated here because the report must not
be over-read: this measures **no** capture-to-capture variance, so the cosine
distances below are not a total-noise floor.

Consequently the probe is a passive rider on an ordinary replay session — see
:func:`make_window_probe_recorder`, wired through ``run_replay_capture``'s
``on_capture_analyzed`` hook. It costs one extra embedding pass per track and
zero extra replay hours.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .inference import AudioEmbeddingModel
from .store import UserStore

DEFAULT_SHORT_WINDOW_S = 30.0


@dataclass(frozen=True)
class WindowPair:
    """One track embedded twice from a single capture: the full pilot window
    and its front ``short_window_s`` truncation."""

    track_id: str
    long_window_s: float
    short_window_s: float
    long_embedding: list[float]
    short_embedding: list[float]


def truncate_pcm(pcm: np.ndarray, sample_rate: int, seconds: float) -> np.ndarray:
    """Front ``seconds`` of ``pcm``. A buffer already shorter than ``seconds``
    is returned whole — the caller records the real duration rather than
    padding, since padding is exactly the ``_mel_patches`` behaviour whose
    effect this probe exists to measure."""
    n = int(seconds * sample_rate)
    return pcm[:n]


def _duration_s(pcm: np.ndarray, sample_rate: int) -> float:
    return pcm.shape[0] / sample_rate


def embed_window_pair(
    *,
    track_id: str,
    pcm: np.ndarray,
    sample_rate: int,
    embedding_model: AudioEmbeddingModel,
    short_window_s: float = DEFAULT_SHORT_WINDOW_S,
    long_embedding: Sequence[float] | None = None,
) -> WindowPair:
    """Embed ``pcm`` whole and truncated to ``short_window_s``.

    ``long_embedding`` lets a caller that already ran inference over the full
    buffer (the replay path always has) pass that vector in rather than paying
    for a second identical embedding pass."""
    long_pcm_duration_s = _duration_s(pcm, sample_rate)
    if long_embedding is None:
        long_vector = np.asarray(embedding_model.embed(pcm, sample_rate), dtype=float)
    else:
        long_vector = np.asarray(long_embedding, dtype=float)

    short_pcm = truncate_pcm(pcm, sample_rate, short_window_s)
    short_vector = np.asarray(embedding_model.embed(short_pcm, sample_rate), dtype=float)

    return WindowPair(
        track_id=track_id,
        long_window_s=long_pcm_duration_s,
        short_window_s=_duration_s(short_pcm, sample_rate),
        long_embedding=[float(x) for x in long_vector],
        short_embedding=[float(x) for x in short_vector],
    )


@dataclass(frozen=True)
class DistanceSummary:
    """Per-track cosine-distance distribution between the two window legs
    (#169 AC1). Every statistic is ``None`` on an empty sample rather than
    ``0.0`` — an unmeasured gate must not read as a passed one."""

    n: int
    mean: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    minimum: float | None = None
    maximum: float | None = None


def pair_cosine_distance(pair: WindowPair) -> float:
    """``1 - cos(long, short)``. A zero vector on either side yields ``1.0``
    (maximally distant) rather than a divide-by-zero: an embedding that
    collapsed to zero is a failed measurement, not a perfect match."""
    long_vec = np.asarray(pair.long_embedding, dtype=float)
    short_vec = np.asarray(pair.short_embedding, dtype=float)
    long_norm = float(np.linalg.norm(long_vec))
    short_norm = float(np.linalg.norm(short_vec))
    if long_norm == 0.0 or short_norm == 0.0:
        return 1.0
    return float(1.0 - (long_vec @ short_vec) / (long_norm * short_norm))


def summarize_distances(pairs: Sequence[WindowPair]) -> DistanceSummary:
    if not pairs:
        return DistanceSummary(n=0)
    distances = np.array([pair_cosine_distance(p) for p in pairs], dtype=float)
    return DistanceSummary(
        n=len(pairs),
        mean=float(distances.mean()),
        median=float(np.median(distances)),
        p90=float(np.percentile(distances, 90)),
        p95=float(np.percentile(distances, 95)),
        minimum=float(distances.min()),
        maximum=float(distances.max()),
    )


@dataclass(frozen=True)
class ClusterAgreement:
    """How far the two window legs disagree about *structure*, not just about
    individual vectors (#169 AC2).

    ``adjusted_rand_index`` is ``None`` when either leg produced no cluster at
    all: there is no partition to compare, and the all-singleton fallback
    would score a meaningless 1.0. ``*_cluster_count`` is also the timbre
    **root** count, since ``derive_timbre_roots`` emits one root per cluster."""

    n_tracks: int
    adjusted_rand_index: float | None
    long_cluster_count: int
    short_cluster_count: int
    long_noise: int
    short_noise: int


def _cluster_labels(
    pairs: Sequence[WindowPair], *, leg: str, min_cluster_size: int
) -> tuple[list[int], int, int]:
    """Run the *production* derivation over one leg and flatten it back into a
    per-track label array aligned with ``pairs``.

    Noise points each get their own singleton label rather than a shared
    ``-1``: HDBSCAN noise is "no structure found here", and collapsing it into
    one pseudo-cluster would let two derivations that agree on nothing but
    which tracks are noise score a high ARI."""
    from .store import AudioAnalysisRecord
    from .timbre import derive_timbre_clusters

    records = [
        AudioAnalysisRecord(
            track_id=p.track_id,
            embedding=(p.long_embedding if leg == "long" else p.short_embedding),
            tags={},
        )
        for p in pairs
    ]
    derivation = derive_timbre_clusters(records, min_cluster_size=min_cluster_size)

    labels_by_track = {
        track_id: index
        for index, cluster in enumerate(derivation.clusters)
        for track_id in cluster.member_ids
    }
    next_singleton = len(derivation.clusters)
    labels: list[int] = []
    noise = 0
    for pair in pairs:
        if pair.track_id in labels_by_track:
            labels.append(labels_by_track[pair.track_id])
        else:
            labels.append(next_singleton)
            next_singleton += 1
            noise += 1
    return labels, len(derivation.clusters), noise


def compare_cluster_assignments(
    pairs: Sequence[WindowPair], *, min_cluster_size: int = 3
) -> ClusterAgreement:
    """Derive timbre clusters from each window leg and score their agreement.

    Deliberately routed through :func:`~music_intel_mcp.timbre.
    derive_timbre_clusters` rather than calling HDBSCAN here: the gate asks
    whether the *pilot's own* derivation moves, so a re-implementation with
    subtly different parameters would answer a different question."""
    from sklearn.metrics import adjusted_rand_score

    long_labels, long_clusters, long_noise = _cluster_labels(
        pairs, leg="long", min_cluster_size=min_cluster_size
    )
    short_labels, short_clusters, short_noise = _cluster_labels(
        pairs, leg="short", min_cluster_size=min_cluster_size
    )

    if long_clusters == 0 or short_clusters == 0:
        index = None
    else:
        index = float(adjusted_rand_score(long_labels, short_labels))

    return ClusterAgreement(
        n_tracks=len(pairs),
        adjusted_rand_index=index,
        long_cluster_count=long_clusters,
        short_cluster_count=short_clusters,
        long_noise=long_noise,
        short_noise=short_noise,
    )


# AC1 asks for a sample of at least this many tracks. Below it the two legs'
# statistics are still computed -- they are useful while a run is in progress --
# but the report says so out loud, because a thin sample that reads as a passed
# gate is exactly the failure mode #169 exists to prevent.
MIN_PROBE_SAMPLE_SIZE = 100


@dataclass(frozen=True)
class WindowProbeReport:
    """Everything the owner needs to judge the gate: the per-track distance
    distribution (AC1), the cluster agreement (AC2), and whether the sample was
    large enough for either to mean anything."""

    n_tracks: int
    min_sample_size: int
    sample_size_ok: bool
    distances: DistanceSummary
    agreement: ClusterAgreement


def build_window_probe_report(
    pairs: Sequence[WindowPair],
    *,
    min_cluster_size: int = 3,
    min_sample_size: int = MIN_PROBE_SAMPLE_SIZE,
) -> WindowProbeReport:
    return WindowProbeReport(
        n_tracks=len(pairs),
        min_sample_size=min_sample_size,
        sample_size_ok=len(pairs) >= min_sample_size,
        distances=summarize_distances(pairs),
        agreement=compare_cluster_assignments(pairs, min_cluster_size=min_cluster_size),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def render_window_probe_report(
    report: WindowProbeReport,
    *,
    title: str = "120 s vs 30 s capture-window probe (#169)",
    long_label: str = "120 s leg",
    short_label: str = "30 s leg",
    capture_noun: str = "capture",
) -> str:
    """Human-readable gate result. The single-capture caveat travels with the
    numbers because the numbers are what gets over-read without it.

    ``title``/``long_label``/``short_label``/``capture_noun`` default to the
    #169 loopback leg's original wording (unchanged) but let a caller render
    a different leg's report without its numbers being mistakable for #169's
    (#201 AC6) -- e.g. the stream-decode leg passes ``long_label="whole-track
    leg"`` since its long leg is a whole decoded track, not a 120 s window."""
    d = report.distances
    a = report.agreement
    lines = [
        title,
        "",
        f"tracks: {report.n_tracks} (minimum for the gate: {report.min_sample_size})",
    ]
    if not report.sample_size_ok:
        lines.append(
            f"  WARNING: under-powered sample -- {report.n_tracks} < {report.min_sample_size} "
            "tracks. These figures do not yet decide the gate."
        )
    lines += [
        "",
        f"cosine distance, {long_label} vs {short_label}:",
        f"  mean {_fmt(d.mean)}  median {_fmt(d.median)}  p90 {_fmt(d.p90)}  p95 {_fmt(d.p95)}",
        f"  min {_fmt(d.minimum)}  max {_fmt(d.maximum)}",
        "",
        "timbre cluster agreement:",
        f"  adjusted Rand index: {_fmt(a.adjusted_rand_index)}",
        f"  roots/clusters: {long_label} {a.long_cluster_count}, "
        f"{short_label} {a.short_cluster_count}",
        f"  unclustered tracks: {long_label} {a.long_noise}, {short_label} {a.short_noise}",
        "",
        f"Caveat: both legs are derived from one {capture_noun} (the {short_label} is a front",
        "truncation of the same buffer), so these distances contain no",
        f"{capture_noun}-to-{capture_noun} variance and are not a total noise floor -- they",
        "isolate window length alone.",
    ]
    return "\n".join(lines)


def window_probe_path(store: UserStore) -> Path:
    """Under the gitignored per-user data root, never in the repo (#169 AC4:
    no captured audio or embeddings committed)."""
    return store.root / "window_probe.jsonl"


def stream_decode_window_probe_path(store: UserStore) -> Path:
    """The #201 stream-decode leg's journal -- separate file from
    :func:`window_probe_path`'s #169 loopback leg (AC6/AC9: never mixed with
    the loopback pairs, never committed)."""
    return store.root / "stream_decode_window_probe.jsonl"


def append_window_pair(path: Path, pair: WindowPair) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(pair)) + "\n")


def load_window_pairs(path: Path) -> list[WindowPair]:
    if not path.exists():
        return []
    pairs: list[WindowPair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pairs.append(WindowPair(**json.loads(line)))
    return pairs


def make_window_probe_recorder(
    *,
    store: UserStore,
    embedding_model: AudioEmbeddingModel,
    short_window_s: float = DEFAULT_SHORT_WINDOW_S,
    path: Path | None = None,
) -> Callable[..., None]:
    """An ``on_capture_analyzed`` hook for :func:`~music_intel_mcp.
    replay_capture.run_replay_capture` that appends one :class:`WindowPair` per
    successful capture.

    The long leg is the embedding the replay path already computed, so the
    probe's marginal cost is a single extra inference pass over the first
    ``short_window_s`` — no second capture, no extra replay hours."""
    journal_path = path if path is not None else window_probe_path(store)

    def _record(
        *,
        track_id: str,
        pcm: np.ndarray,
        sample_rate: int,
        embedding: Sequence[float],
        tags: dict[str, float],
    ) -> None:
        pair = embed_window_pair(
            track_id=track_id,
            pcm=pcm,
            sample_rate=sample_rate,
            embedding_model=embedding_model,
            short_window_s=short_window_s,
            long_embedding=embedding,
        )
        append_window_pair(journal_path, pair)

    return _record
