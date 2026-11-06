"""Timbre pipeline (#125, wired into RootProfile per #162) — per-user
clustering over Discogs-EffNet embeddings.

:func:`derive_timbre_clusters` runs HDBSCAN directly over the raw
(~1280-dim in production) embedding vectors from
:meth:`music_intel_mcp.store.UserStore.list_audio_analyses` — no z-scoring,
since a per-dimension z-score assumes each embedding dim has independent
meaningful scale/units the way bpm/energy/valence do, which a learned
embedding space does not (decision 0c762eec). Classifier tags are
descriptive only: used to label clusters after the fact via nearest-tag
lookup, never fed back into clustering.

:func:`select_timbre_records` (#264) picks what a profile may cluster:
(root ∪ pool) ∩ this participant's history by alias-resolved canonical
track id, current embedding version only — never the whole pool, which
would leak other participants' tracks into this profile — and measures
timbre coverage as the share of valid plays with a selected record.
:func:`derive_timbre_roots` (#162, decision 945459ac reopened) is a pure
function of the records it is given and emits one
``Root(category="timbre")`` per resulting ``TimbreCluster``;
``analyzer.analyze`` is its caller. No ``Validator`` exists yet for this
category (unlike audio/scene/temporal), so every emitted root is
unconditionally ``classification="root"`` with
``coverage_pass``/``confidence_pass`` both ``True`` and caveats marking
it proxy / preliminary and noting the missing floor system.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN

from .inference import EMBEDDING_SPACE_VERSION
from .models import Evidence, ListenEvent, Root, TemporalStability, ValidationScores
from .shared_store import canonical_track_id
from .store import AudioAnalysisRecord
from .temporal import _is_valid

DEFAULT_TOP_TAGS_COUNT = 5
DEFAULT_SAMPLE_TRACK_COUNT = 3


@dataclass(frozen=True)
class TimbreCluster:
    cluster_id: str
    member_ids: list[str]
    cluster_size: int
    confidence: float
    top_tags: list[dict] = field(default_factory=list)
    sample_tracks: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class TimbreDerivation:
    clusters: list[TimbreCluster]
    n_clustered: int
    n_noise: int


def derive_timbre_clusters(
    analyses: Sequence[AudioAnalysisRecord],
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    top_tags_count: int = DEFAULT_TOP_TAGS_COUNT,
    sample_track_count: int = DEFAULT_SAMPLE_TRACK_COUNT,
) -> TimbreDerivation:
    ordered = sorted(analyses, key=lambda a: a.track_id)
    n = len(ordered)
    if n < min_cluster_size:
        return TimbreDerivation(clusters=[], n_clustered=n, n_noise=n)

    x = np.array([a.embedding for a in ordered], dtype=float)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True)
    labels = clusterer.fit_predict(x)
    probabilities = clusterer.probabilities_

    cluster_labels = sorted(set(labels) - {-1})
    built = []
    for label in cluster_labels:
        members = [i for i, lb in enumerate(labels) if lb == label]
        built.append(
            _build_timbre_cluster(
                members,
                ordered,
                x,
                probabilities,
                top_tags_count=top_tags_count,
                sample_track_count=sample_track_count,
            )
        )
    built.sort(key=lambda c: (-c["cluster_size"], c["min_id"]))

    clusters = [
        TimbreCluster(
            cluster_id=f"timbre-{rank}",
            member_ids=c["member_ids"],
            cluster_size=c["cluster_size"],
            confidence=c["confidence"],
            top_tags=c["top_tags"],
            sample_tracks=c["sample_tracks"],
        )
        for rank, c in enumerate(built, start=1)
    ]
    n_noise = int(np.sum(labels == -1))
    return TimbreDerivation(clusters=clusters, n_clustered=n, n_noise=n_noise)


def _build_timbre_cluster(
    members: list[int],
    ordered: list[AudioAnalysisRecord],
    x: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_tags_count: int,
    sample_track_count: int,
) -> dict:
    member_records = [ordered[i] for i in members]
    size = len(members)

    centroid = x[members].mean(axis=0)
    distances = np.linalg.norm(x[members] - centroid, axis=1)
    order = sorted(range(size), key=lambda k: (float(distances[k]), member_records[k].track_id))
    samples = [
        {
            "track_id": member_records[k].track_id,
            "distance_to_centroid": round(float(distances[k]), 4),
        }
        for k in order[:sample_track_count]
    ]

    tag_totals: dict[str, float] = {}
    for record in member_records:
        for tag, score in record.tags.items():
            tag_totals[tag] = tag_totals.get(tag, 0.0) + score
    ranked_tags = sorted(tag_totals.items(), key=lambda kv: (-kv[1], kv[0]))
    top_tags = [
        {"tag": tag, "score": round(total / size, 6)} for tag, total in ranked_tags[:top_tags_count]
    ]

    return {
        "cluster_size": size,
        "min_id": min(r.track_id for r in member_records),
        "member_ids": [r.track_id for r in member_records],
        "confidence": round(float(probabilities[members].mean()), 6),
        "top_tags": top_tags,
        "sample_tracks": samples,
    }


def _identity(key: str) -> str:
    return key


@dataclass(frozen=True)
class TimbreSource:
    """Where a profile's timbre records come from (#264 AC1).

    ``records`` is every analysis the processing node holds for this
    participant's store: the pool's first, then the root's, so on a duplicate
    track key the pool record wins. ``resolve_key`` maps a canonical track id
    through the recorded aliases (#178), applied to history keys and record
    keys alike. ``embedding_version`` is the only embedding space clustered
    together: records from another space, or with no version at all, are not
    comparable vectors and are dropped."""

    records: Sequence[AudioAnalysisRecord]
    resolve_key: Callable[[str], str] = _identity
    embedding_version: str = EMBEDDING_SPACE_VERSION


@dataclass(frozen=True)
class TimbreSelection:
    """The records a profile may cluster, and what was dropped on the way."""

    records: list[AudioAnalysisRecord]
    coverage: float
    n_candidates: int
    n_stale_version: int
    n_outside_history: int


def select_timbre_records(events: Iterable[ListenEvent], source: TimbreSource) -> TimbreSelection:
    """#264 AC1/AC4: (root ∪ pool) ∩ this participant's history, at
    ``source.embedding_version`` only — never the whole pool, which would
    surface other participants' tracks in this profile.

    Only the first record per resolved track id is kept, so a track analysed
    twice is one point in the clustering, not two. ``coverage`` is the share of valid plays
    (``temporal._is_valid``) whose track has a selected record; 0.0 when there
    are no valid plays."""
    resolve = source.resolve_key
    events = list(events)
    history_keys = {resolve(canonical_track_id(event.track)) for event in events}

    selected: dict[str, AudioAnalysisRecord] = {}
    n_stale = 0
    n_outside = 0
    for record in source.records:
        if record.model_version != source.embedding_version:
            n_stale += 1
            continue
        key = resolve(record.track_id)
        if key not in history_keys:
            n_outside += 1
            continue
        if key not in selected:
            selected[key] = record

    valid_keys = [
        resolve(canonical_track_id(event.track)) for event in events if _is_valid(event.context)
    ]
    covered = sum(1 for key in valid_keys if key in selected)
    coverage = covered / len(valid_keys) if valid_keys else 0.0

    return TimbreSelection(
        records=list(selected.values()),
        coverage=coverage,
        n_candidates=len(source.records),
        n_stale_version=n_stale,
        n_outside_history=n_outside,
    )


def derive_timbre_roots(
    records: Iterable[AudioAnalysisRecord],
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    top_tags_count: int = DEFAULT_TOP_TAGS_COUNT,
    sample_track_count: int = DEFAULT_SAMPLE_TRACK_COUNT,
) -> list[Root]:
    """#162: one ``Root(category="timbre")`` per embedding cluster of
    ``records``. #264 AC1: a pure function of the records it is given — the
    caller decides which (``select_timbre_records``); no records, no roots."""
    return derive_timbre_roots_with_stats(
        records,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        top_tags_count=top_tags_count,
        sample_track_count=sample_track_count,
    )[0]


def derive_timbre_roots_with_stats(
    records: Iterable[AudioAnalysisRecord],
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    top_tags_count: int = DEFAULT_TOP_TAGS_COUNT,
    sample_track_count: int = DEFAULT_SAMPLE_TRACK_COUNT,
) -> tuple[list[Root], TimbreDerivation]:
    """``derive_timbre_roots`` plus the clustering counts, so the analyzer can
    log a selection that produced no cluster (transparent rejection)."""
    derivation = derive_timbre_clusters(
        list(records),
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        top_tags_count=top_tags_count,
        sample_track_count=sample_track_count,
    )
    roots = [
        _build_timbre_root(cluster, rank, n_clustered=derivation.n_clustered)
        for rank, cluster in enumerate(derivation.clusters, start=1)
    ]
    return roots, derivation


def _humanize_tag(tag: str) -> str:
    segment = tag.split("---")[-1]
    return segment.replace("-", " ").replace("_", " ").title()


def _build_timbre_root(cluster: TimbreCluster, rank: int, *, n_clustered: int) -> Root:
    humanized_tags = [_humanize_tag(t["tag"]) for t in cluster.top_tags]
    label = humanized_tags[0] if humanized_tags else f"Timbre Cluster {rank}"
    if humanized_tags:
        explanation = (
            f"{cluster.cluster_size} tracks in your history share a distinct sound "
            f"profile, most associated with {', '.join(humanized_tags[:3])}."
        )
    else:
        explanation = (
            f"{cluster.cluster_size} tracks in your history share a distinct sound "
            "profile with no dominant descriptive tags."
        )
    share = cluster.cluster_size / n_clustered if n_clustered else 0.0
    return Root(
        id=f"r-timbre-{rank}",
        category="timbre",
        classification="root",
        structural_descriptor={
            "label": label,
            "explanation": explanation,
            "top_tags": cluster.top_tags,
        },
        evidence=Evidence(
            cluster_size=cluster.cluster_size,
            cluster_share=share,
            evidence_count=cluster.cluster_size,
            coverage=share,
            sample_tracks=cluster.sample_tracks,
        ),
        validation_scores=ValidationScores(
            confidence=cluster.confidence,
            temporal_stability=TemporalStability(status="not_evaluated", score=None),
            coverage_pass=True,
            confidence_pass=True,
        ),
        caveats=[
            "proxy / preliminary: one embedding space only, not a full root model (#264)",
            "timbre roots are unvalidated: no confidence/coverage floor system "
            "implemented yet (#162)",
        ],
    )
