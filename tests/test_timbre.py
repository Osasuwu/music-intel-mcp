"""Embedding-space clustering for the timbre pipeline (#125).

Per-user HDBSCAN over raw Discogs-EffNet embedding vectors (not z-scored,
not tags/scalars — decision 0c762eec). Mirrors audio.py's HDBSCAN/
_build_cluster conventions but is intentionally NOT integrated into
RootProfile/Root.category/MethodParams/Validator (decision 945459ac) — the
issue's ACs don't require that wiring.
"""

from __future__ import annotations

import numpy as np

from music_intel_mcp.store import AudioAnalysisRecord
from music_intel_mcp.timbre import derive_timbre_clusters


def _record(
    track_id: str, embedding: list[float], tags: dict[str, float] | None = None
) -> AudioAnalysisRecord:
    return AudioAnalysisRecord(track_id=track_id, embedding=embedding, tags=tags or {})


# AC2: HDBSCAN runs over the embedding vectors themselves, not over tags/scalars.
# Two tight embedding neighborhoods, far apart in embedding space, with tags
# that (if used for clustering) would produce the *same* grouping here — this
# test only proves clustering follows the embedding geometry, not the tags.
def test_derive_timbre_clusters_groups_by_embedding_not_tags():
    rng = np.random.default_rng(0)
    cluster_a = [
        _record(f"a{i}", (np.array([0.0, 0.0]) + rng.normal(scale=0.01, size=2)).tolist())
        for i in range(4)
    ]
    cluster_b = [
        _record(f"b{i}", (np.array([10.0, 10.0]) + rng.normal(scale=0.01, size=2)).tolist())
        for i in range(4)
    ]

    result = derive_timbre_clusters(cluster_a + cluster_b, min_cluster_size=3)

    assert len(result.clusters) == 2
    ids_by_cluster = [set(c.member_ids) for c in result.clusters]
    assert {r.track_id for r in cluster_a} in ids_by_cluster
    assert {r.track_id for r in cluster_b} in ids_by_cluster


# AC3: clusters are labeled via nearest-tag lookup (aggregated member tags,
# descriptive only) + representative sample tracks (nearest to centroid).
def test_derive_timbre_clusters_labels_via_tags_and_sample_tracks():
    records = [
        _record("a0", [0.0, 0.0], tags={"genre---electronic": 0.9, "mood---dark": 0.4}),
        _record("a1", [0.01, 0.0], tags={"genre---electronic": 0.7}),
        _record("a2", [0.0, 0.01], tags={"genre---electronic": 0.8}),
        _record("a3", [10.0, 10.0], tags={"genre---rock": 0.6}),
        _record("a4", [10.01, 10.0], tags={"genre---rock": 0.5}),
        _record("a5", [10.0, 10.01], tags={"genre---rock": 0.7}),
    ]

    result = derive_timbre_clusters(records, min_cluster_size=3, top_tags_count=2)

    electronic_cluster = next(c for c in result.clusters if "a0" in c.member_ids)
    top_tag_names = [t["tag"] for t in electronic_cluster.top_tags]
    assert top_tag_names[0] == "genre---electronic"
    assert electronic_cluster.sample_tracks  # non-empty, nearest-to-centroid tracks
    assert all(
        s["track_id"] in electronic_cluster.member_ids for s in electronic_cluster.sample_tracks
    )


# AC4: a genre-bending / cross-vocabulary track — tagged with vocabulary that
# shares nothing with either cluster's tags — still lands in the coherent
# embedding-space cluster its audio actually belongs to. Tag-only clustering
# (grouping by shared tag vocabulary) would have missed or mislabeled it,
# since it has zero tag overlap with the electronic cluster it embeds into.
def test_derive_timbre_clusters_places_genre_bending_track_by_embedding():
    electronic = [
        _record(f"e{i}", [0.0 + 0.01 * i, 0.0], tags={"genre---electronic": 0.9}) for i in range(4)
    ]
    rock = [_record(f"r{i}", [10.0 + 0.01 * i, 10.0], tags={"genre---rock": 0.8}) for i in range(4)]
    # No tag overlap with "genre---electronic" at all, yet its embedding sits
    # squarely inside the electronic cluster's tight neighborhood.
    genre_bender = _record("cross-vocab", [0.005, 0.0], tags={"experimental---jazz-fusion": 0.95})

    result = derive_timbre_clusters(electronic + rock + [genre_bender], min_cluster_size=3)

    electronic_cluster = next(c for c in result.clusters if "e0" in c.member_ids)
    assert "cross-vocab" in electronic_cluster.member_ids
    rock_cluster = next(c for c in result.clusters if "r0" in c.member_ids)
    assert "cross-vocab" not in rock_cluster.member_ids


def test_derive_timbre_clusters_below_min_size_returns_empty():
    records = [_record("a", [0.0, 0.0]), _record("b", [0.0, 0.0])]

    result = derive_timbre_clusters(records, min_cluster_size=3)

    assert result.clusters == []
    assert result.n_clustered == 2
    assert result.n_noise == 2
