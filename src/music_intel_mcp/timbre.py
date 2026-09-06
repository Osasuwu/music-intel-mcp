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

:func:`derive_timbre_roots` (#162, decision 945459ac reopened) is the
pool ∩ participant-history bridge into ``Root.category == "timbre"``: it
intersects this participant's own listen history against the node-level
anonymous pool (#161) by canonical track id — never clustering over the
whole pool, which would leak other participants' tracks into this
profile — then emits one ``Root`` per resulting ``TimbreCluster``. No
``Validator`` exists yet for this category (unlike audio/scene/temporal),
so every emitted root is unconditionally ``classification="root"`` with
``coverage_pass``/``confidence_pass`` both ``True`` and an explicit
caveat noting the missing floor system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN

from .models import Evidence, Root, TemporalStability, ValidationScores
from .shared_store import canonical_track_id
from .store import AudioAnalysisRecord, UserStore

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


def derive_timbre_roots(
    store: UserStore,
    *,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    top_tags_count: int = DEFAULT_TOP_TAGS_COUNT,
    sample_track_count: int = DEFAULT_SAMPLE_TRACK_COUNT,
) -> list[Root]:
    """#162: derive ``Root(category="timbre")`` entries from the pool ∩
    participant-history intersection — never the whole pool, which would
    surface other participants' tracks in this profile. A profile with no
    history and/or no matching pool analyses derives no timbre roots at
    all (same honest-empty shape as before this issue existed)."""
    history_ids = {canonical_track_id(event.track) for event in store.load_history()}
    if not history_ids:
        return []

    pool_analyses = [a for a in store.list_pool_audio_analyses() if a.track_id in history_ids]
    if not pool_analyses:
        return []

    derivation = derive_timbre_clusters(
        pool_analyses,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        top_tags_count=top_tags_count,
        sample_track_count=sample_track_count,
    )
    return [
        _build_timbre_root(cluster, rank, n_clustered=derivation.n_clustered)
        for rank, cluster in enumerate(derivation.clusters, start=1)
    ]


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
            "timbre roots are unvalidated: no confidence/coverage floor system "
            "implemented yet (#162)"
        ],
    )
