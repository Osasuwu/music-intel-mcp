"""Embedding-space clustering for the timbre pipeline (#125, wired into
RootProfile per #162).

Per-user HDBSCAN over raw Discogs-EffNet embedding vectors (not z-scored,
not tags/scalars — decision 0c762eec). Mirrors audio.py's HDBSCAN/
_build_cluster conventions. ``select_timbre_records`` (#264) picks the
records a profile may cluster — (root ∪ pool) ∩ history at the current
embedding version — and ``derive_timbre_roots`` (#162) turns them into
``Root.category == "timbre"``; see ``timbre.py``'s module docstring.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from music_intel_mcp.inference import EMBEDDING_SPACE_VERSION
from music_intel_mcp.models import ListenEvent, PlayContext, TrackRef
from music_intel_mcp.store import AudioAnalysisRecord
from music_intel_mcp.timbre import (
    TimbreSource,
    derive_timbre_clusters,
    derive_timbre_roots,
    select_timbre_records,
)


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
# select_timbre_records (#264) -- (root ∪ pool) ∩ history at the current
# embedding version, plus timbre coverage as a share of valid plays
# --------------------------------------------------------------------------- #

OLD_VERSION = "discogs-effnet-bsdynamic-1"  # a pre-#194 embedding space


def _ref(spotify_id: str) -> TrackRef:
    return TrackRef(spotify_id=spotify_id, name=spotify_id, artist="artist")


def _listen(
    spotify_id: str, *, ms_played: int | None = None, skipped: bool | None = None
) -> ListenEvent:
    context = None
    if ms_played is not None or skipped is not None:
        context = PlayContext(ms_played=ms_played, skipped=skipped)
    return ListenEvent(
        track=_ref(spotify_id),
        played_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="test",
        context=context,
    )


def _rec(
    track_id: str,
    embedding: list[float],
    *,
    version: str | None = EMBEDDING_SPACE_VERSION,
    tags: dict[str, float] | None = None,
) -> AudioAnalysisRecord:
    return AudioAnalysisRecord(
        track_id=track_id, embedding=embedding, tags=tags or {}, model_version=version
    )


# AC1: the input to timbre derivation is (root ∪ pool) ∩ this participant's
# history, restricted to the current embedding version. A pool record and a
# root record of listened tracks both count; a record from an older embedding
# space, a record with no version at all, and a record of a track this
# participant never played are each dropped, and each drop is counted.
def test_select_timbre_records_keeps_current_version_history_records_from_root_and_pool():
    events = [_listen("a"), _listen("b"), _listen("c"), _listen("d")]
    pool = [
        _rec("spotify:a", [0.0, 0.0]),
        _rec("spotify:c", [0.0, 0.0], version=OLD_VERSION),
        _rec("spotify:z", [0.0, 0.0]),  # another participant's track
    ]
    root = [
        _rec("spotify:b", [1.0, 1.0]),
        _rec("spotify:d", [1.0, 1.0], version=None),  # pre-#194 root record
    ]

    selection = select_timbre_records(events, TimbreSource(records=pool + root))

    assert sorted(r.track_id for r in selection.records) == ["spotify:a", "spotify:b"]
    assert selection.n_candidates == 5
    assert selection.n_stale_version == 2
    assert selection.n_outside_history == 1


# AC1: one record per track. A track analysed into the root before the pool
# existed and again into the pool is one track, not two points in the
# clustering; the first record in the source (the pool's, by the CLI's order)
# wins. The same holds when the two keys differ only through an alias.
def test_select_timbre_records_keeps_the_first_record_per_resolved_key():
    events = [_listen("a"), _listen("b")]
    aliases = {"spotify:b": "mbid:bbb"}
    records = [
        _rec("spotify:a", [1.0, 1.0]),
        _rec("mbid:bbb", [3.0, 3.0]),
        _rec("spotify:a", [2.0, 2.0]),
        _rec("spotify:b", [4.0, 4.0]),
    ]

    selection = select_timbre_records(
        events,
        TimbreSource(records=records, resolve_key=lambda key: aliases.get(key, key)),
    )

    assert sorted((r.track_id, list(r.embedding)) for r in selection.records) == [
        ("mbid:bbb", [3.0, 3.0]),
        ("spotify:a", [1.0, 1.0]),
    ]


# AC4: timbre coverage is a share of valid plays, not of tracks: 5 valid
# plays (a0 x3, a1, b), of which the 4 on a0/a1 have a selected record ->
# 0.8. The skipped a0 play and the 10 s b play are not valid plays and count
# on neither side. Counting tracks would give 2/3; counting every play 5/7.
def test_select_timbre_records_reports_coverage_as_share_of_valid_plays():
    events = [
        _listen("a0"),
        _listen("a0"),
        _listen("a0", ms_played=200_000),
        _listen("a0", skipped=True),
        _listen("a1"),
        _listen("b"),
        _listen("b", ms_played=10_000),
    ]
    records = [_rec("spotify:a0", [0.0, 0.0]), _rec("spotify:a1", [0.0, 0.0])]

    selection = select_timbre_records(events, TimbreSource(records=records))

    assert selection.coverage == 0.8


def test_select_timbre_records_without_valid_plays_reports_zero_coverage():
    events = [_listen("a", skipped=True)]

    selection = select_timbre_records(events, TimbreSource(records=[_rec("spotify:a", [0.0])]))

    assert selection.coverage == 0.0


# --------------------------------------------------------------------------- #
# derive_timbre_roots (#162, #264) -- records -> Root(category="timbre")
# --------------------------------------------------------------------------- #


# #162 AC3, moved onto the record iterable by #264 AC1: a real cluster becomes
# a Root whose label and explanation come from the cluster's top tags, with
# sample tracks drawn from its own members.
def test_derive_timbre_roots_emits_label_explanation_and_sample_tracks():
    electronic = [
        _rec(f"spotify:e{i}", [0.0 + 0.001 * i, 0.0], tags={"genre---electronic": 0.9})
        for i in range(4)
    ]
    rock = [
        _rec(f"spotify:r{i}", [10.0 + 0.001 * i, 10.0], tags={"genre---rock": 0.8})
        for i in range(4)
    ]

    roots = derive_timbre_roots(electronic + rock, min_cluster_size=3)

    assert len(roots) == 2
    by_label = {root.structural_descriptor["label"]: root for root in roots}
    assert set(by_label) == {"Electronic", "Rock"}
    assert all(root.category == "timbre" for root in roots)
    assert by_label["Electronic"].structural_descriptor["explanation"] == (
        "4 tracks in your history share a distinct sound profile, most associated with Electronic."
    )
    assert by_label["Electronic"].evidence.cluster_size == 4
    electronic_samples = {s["track_id"] for s in by_label["Electronic"].evidence.sample_tracks}
    assert len(electronic_samples) == 3
    assert electronic_samples <= {f"spotify:e{i}" for i in range(4)}
