"""Timbre pipeline (#125) — per-user clustering over Discogs-EffNet embeddings.

Standalone module, not wired into ``RootProfile``/``Root.category``/
``MethodParams``/``Validator`` (decision 945459ac) — the issue's acceptance
criteria don't require that integration.

:func:`derive_timbre_clusters` runs HDBSCAN directly over the raw
(~1280-dim in production) embedding vectors from
:meth:`music_intel_mcp.store.UserStore.list_audio_analyses` — no z-scoring,
since a per-dimension z-score assumes each embedding dim has independent
meaningful scale/units the way bpm/energy/valence do, which a learned
embedding space does not (decision 0c762eec). Classifier tags are
descriptive only: used to label clusters after the fact via nearest-tag
lookup, never fed back into clustering.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN

from .store import AudioAnalysisRecord

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
