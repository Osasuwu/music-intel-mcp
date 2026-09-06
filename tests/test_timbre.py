"""Embedding-space clustering for the timbre pipeline (#125, wired into
RootProfile per #162).

Per-user HDBSCAN over raw Discogs-EffNet embedding vectors (not z-scored,
not tags/scalars — decision 0c762eec). Mirrors audio.py's HDBSCAN/
_build_cluster conventions. ``derive_timbre_roots`` (#162, decision
945459ac reopened) is the pool ∩ participant-history bridge into
``Root.category == "timbre"`` — see ``timbre.py``'s module docstring for
the full picture.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from music_intel_mcp.models import ListenEvent, TrackRef
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import AudioAnalysisRecord, UserStore
from music_intel_mcp.timbre import derive_timbre_clusters, derive_timbre_roots


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


# --------------------------------------------------------------------------- #
# derive_timbre_roots (#162) — pool ∩ participant-history -> Root bridge
# --------------------------------------------------------------------------- #


def _store(tmp_path, *, pool: bool = True) -> UserStore:
    return UserStore(
        root=tmp_path / "participant",
        pool_root=(tmp_path / "pool") if pool else None,
    )


def _track_ref(spotify_id: str) -> TrackRef:
    return TrackRef(spotify_id=spotify_id, name=spotify_id, artist="artist")


def _listen(spotify_id: str) -> ListenEvent:
    return ListenEvent(
        track=_track_ref(spotify_id),
        played_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="test",
    )


def _write_pool_track(store: UserStore, spotify_id: str, embedding: list[float]) -> None:
    # pool records are keyed by the same canonical id the live pipeline
    # resolves from a TrackRef -- not a bare "spotify_id" string.
    store.write_audio_analysis(
        track_id=canonical_track_id(_track_ref(spotify_id)), embedding=embedding, tags={}
    )


# AC2: a track present in the pool but NOT in this participant's own history
# must never surface in any derived root, even if it would otherwise cluster
# tightly with the participant's own tracks -- reading the whole pool would
# leak other participants' tracks into every profile. Two separated clusters
# (a and b) are both in history so HDBSCAN has genuine density contrast to
# extract from -- a single blob with nothing else is indistinguishable from
# noise to HDBSCAN's default extraction (no allow_single_cluster).
def test_derive_timbre_roots_excludes_pool_tracks_outside_history(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.append_events([_listen(f"a{i}")])
        store.append_events([_listen(f"b{i}")])
        _write_pool_track(store, f"a{i}", [0.0 + 0.001 * i, 0.0])
        _write_pool_track(store, f"b{i}", [10.0 + 0.001 * i, 10.0])
    # foreign pool track: tight to the "a" history cluster in embedding
    # space, but never listened to by this participant -- must never leak in.
    _write_pool_track(store, "z", [0.0005, 0.0])

    roots = derive_timbre_roots(store, min_cluster_size=3)

    assert len(roots) == 2
    z_id = canonical_track_id(_track_ref("z"))
    for root in roots:
        # z sits inside the "a" cluster's embedding neighborhood; if the pool
        # weren't filtered down to history first, that cluster's evidence
        # would reflect a cluster of 4, not 3.
        assert root.evidence.cluster_size == 3
        sample_ids = {s.get("track_id") for s in root.evidence.sample_tracks}
        assert z_id not in sample_ids


# AC3: real cluster -> Root emission carries a human-readable label/
# explanation derived from the cluster's top tags, plus sample tracks.
def test_derive_timbre_roots_emits_label_explanation_and_sample_tracks(tmp_path):
    store = _store(tmp_path)
    for i in range(4):
        store.append_events([_listen(f"e{i}")])
        store.write_audio_analysis(
            track_id=canonical_track_id(_track_ref(f"e{i}")),
            embedding=[0.0 + 0.001 * i, 0.0],
            tags={"genre---electronic": 0.9},
        )
    for i in range(4):
        store.append_events([_listen(f"r{i}")])
        store.write_audio_analysis(
            track_id=canonical_track_id(_track_ref(f"r{i}")),
            embedding=[10.0 + 0.001 * i, 10.0],
            tags={"genre---rock": 0.8},
        )

    roots = derive_timbre_roots(store, min_cluster_size=3)

    assert len(roots) == 2
    for root in roots:
        assert root.category == "timbre"
        assert root.structural_descriptor["label"] in {"Electronic", "Rock"}
        assert root.structural_descriptor["explanation"]
        assert root.structural_descriptor["top_tags"]
        assert root.evidence.sample_tracks
        assert all(
            s["track_id"] in {canonical_track_id(_track_ref(f"e{i}")) for i in range(4)}
            or s["track_id"] in {canonical_track_id(_track_ref(f"r{i}")) for i in range(4)}
            for s in root.evidence.sample_tracks
        )


# AC4: no prior pilot-topology derivation exists for this store shape, so
# "same as current derivation" means the honest-empty shape -- [] -- for
# empty history, for no pool configured, and for a disjoint pool/history.
def test_derive_timbre_roots_returns_empty_for_empty_history(tmp_path):
    store = _store(tmp_path)
    # a real, clusterable pool (>= min_cluster_size) -- so this only stays
    # empty because history is empty, not because the pool is too small to
    # cluster on its own.
    for i in range(4):
        _write_pool_track(store, f"a{i}", [0.0 + 0.001 * i, 0.0])

    assert derive_timbre_roots(store) == []


def test_derive_timbre_roots_returns_empty_when_pool_not_configured(tmp_path):
    store = _store(tmp_path, pool=False)
    for i in range(4):
        store.append_events([_listen(f"a{i}")])

    assert derive_timbre_roots(store) == []


def test_derive_timbre_roots_returns_empty_for_disjoint_pool_and_history(tmp_path):
    store = _store(tmp_path)
    for i in range(4):
        store.append_events([_listen(f"a{i}")])
    # two separated blobs in the pool, both outside history -- HDBSCAN can
    # extract real clusters from these (unlike a lone blob), so this only
    # stays empty because the history-intersection filter drops them, not
    # because the pool itself is unclusterable.
    for i in range(4):
        _write_pool_track(store, f"z{i}", [0.0 + 0.001 * i, 0.0])
        _write_pool_track(store, f"w{i}", [10.0 + 0.001 * i, 10.0])

    assert derive_timbre_roots(store) == []
