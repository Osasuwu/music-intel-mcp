"""Scene root pipeline (#64): tag enrichment buckets + NMF topic derivation,
K auto-selection by coherence, the coherence floor, and honest-empty.

NMF is deterministic here (``init="nndsvd"`` + pinned ``random_state`` + sorted
docs/vocab), so these assert exact roots, not just shapes."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from music_intel_mcp.models import SceneParams, ValidationParams
from music_intel_mcp.scene import (
    CompositeTagSource,
    DiscogsStyleSource,
    InMemoryTagSource,
    MusicBrainzGenreSource,
    SceneDerivation,
    TagEnrichmentReport,
    derive_scene_roots,
    enrich_tags,
)
from music_intel_mcp.shared_store import InMemorySharedStore, TrackMetadataRecord, TrackTag
from music_intel_mcp.validation import DatasetContext

NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
EARLY = datetime(2025, 1, 20, tzinfo=UTC)
LATE = datetime(2025, 5, 20, tzinfo=UTC)

# Relaxed thresholds open the temporal gate + lower the floors on small fixtures.
VP_OPEN = ValidationParams(
    N_THRESHOLD=5, T_THRESHOLD_DAYS=5, evidence_count_floor=5, confidence_floor=0.5
)
CTX = DatasetContext(n_unique_tracks=12, history_span_days=150)

# Two coherent tag blobs: "metal" tracks share {metal,thrash,heavy}, "jazz"
# tracks share {jazz,bebop,swing}. Distinct artist per blob → top_artists is
# meaningful. Each track is played once in each history half (balanced → high
# temporal stability), mirroring test_audio's _two_blobs.
_BLOBS = {
    "metal": (["metal", "thrash", "heavy"], "MetalBand"),
    "jazz": (["jazz", "bebop", "swing"], "JazzCombo"),
}


def _record(track_id: str, name: str, artist: str, tags: list[str] | None) -> TrackMetadataRecord:
    return TrackMetadataRecord(
        track_id=track_id,
        name=name,
        artist=artist,
        tags=[TrackTag(tag=t, weight=1.0, source="synthetic") for t in (tags or [])],
        fetched_at=NOW,
    )


def _two_tag_blobs() -> tuple[dict[str, TrackMetadataRecord], dict[str, list[datetime]]]:
    records: dict[str, TrackMetadataRecord] = {}
    plays: dict[str, list[datetime]] = {}
    for label, (tags, artist) in _BLOBS.items():
        for i in range(6):
            tid = f"mbid:{label}-{i}"
            records[tid] = _record(tid, f"{label} {i}", artist, tags)
            plays[tid] = [EARLY, LATE]
    return records, plays


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #


def test_enrich_tags_buckets_and_coverage():
    store = InMemorySharedStore(
        [
            _record("a", "Track A", "Artist X", None),  # will be tagged
            _record("b", "Track B", "Artist Y", ["jazz"]),  # already tagged
            _record("c", "Track C", "Artist Z", None),  # source has nothing
        ]
    )
    source = InMemoryTagSource(
        {("Artist X", "Track A"): [TrackTag(tag="metal", weight=80.0, source="lastfm")]}
    )

    report = enrich_tags(["a", "b", "c"], store, source, now=NOW)

    assert report.enriched == ["a"]
    assert report.already_present == ["b"]
    assert report.missing_tags == ["c"]
    assert report.total_considered == 3
    assert report.coverage == 2 / 3
    # write-back: 'a' now carries the looked-up tag
    assert [t.tag for t in store.get_tracks(["a"])["a"].tags] == ["metal"]
    # already-tagged 'b' was never queried
    assert ("Artist Y", "Track B") not in source.lookups


def test_enrich_tags_skips_already_tagged_no_lookup():
    store = InMemorySharedStore([_record("a", "Track A", "Artist X", ["metal", "thrash"])])
    source = InMemoryTagSource()
    report = enrich_tags(["a"], store, source, now=NOW)
    assert report.already_present == ["a"]
    assert source.lookups == []


# --------------------------------------------------------------------------- #
# CompositeTagSource (#122)
# --------------------------------------------------------------------------- #


def test_composite_returns_first_non_empty_source():
    key = ("artist x", "track a")
    first = InMemoryTagSource({key: [TrackTag(tag="metal", weight=1.0, source="lastfm")]})
    second = InMemoryTagSource({key: [TrackTag(tag="thrash", weight=1.0, source="musicbrainz")]})
    composite = CompositeTagSource([first, second])

    tags = composite.lookup("Artist X", "Track A", None)

    assert [t.tag for t in tags] == ["metal"]
    # second source is never queried once the first hits
    assert second.lookups == []


def test_composite_falls_through_to_next_source_on_miss():
    key = ("artist x", "track a")
    first = InMemoryTagSource()  # empty -> always None
    second = InMemoryTagSource({key: [TrackTag(tag="thrash", weight=1.0, source="musicbrainz")]})
    composite = CompositeTagSource([first, second])

    tags = composite.lookup("Artist X", "Track A", None)

    assert [t.tag for t in tags] == ["thrash"]
    assert first.lookups == [("Artist X", "Track A")]


def test_composite_returns_none_when_all_sources_miss():
    composite = CompositeTagSource([InMemoryTagSource(), InMemoryTagSource()])
    assert composite.lookup("Artist X", "Track A", None) is None


# --------------------------------------------------------------------------- #
# MusicBrainzGenreSource (#122)
# --------------------------------------------------------------------------- #

_MB_MBID = "1fc545d3-94b6-4a3d-bff3-bb8b861943d6"
_MB_URL = f"https://musicbrainz.org/ws/2/recording/{_MB_MBID}"


def test_musicbrainz_genre_source_merges_genres_and_tags_deduped():
    source = MusicBrainzGenreSource(app_contact="me@example.com")
    body = {
        "genres": [{"name": "Metal", "count": 5}],
        "tags": [{"name": "metal", "count": 2}, {"name": "thrash", "count": 1}],
    }
    with respx.mock(assert_all_called=False) as router:
        router.get(_MB_URL).mock(return_value=httpx.Response(200, json=body))
        tags = source.lookup("Artist X", "Track A", _MB_MBID)

    # 'Metal' (genre) wins over the casefold-duplicate 'metal' (tag); 'thrash' survives.
    assert [t.tag for t in tags] == ["Metal", "thrash"]
    assert all(t.source == "musicbrainz" for t in tags)


def test_musicbrainz_genre_source_no_mbid_short_circuits_without_network():
    source = MusicBrainzGenreSource(app_contact="me@example.com")
    with respx.mock(assert_all_called=False) as router:
        route = router.get(_MB_URL).mock(return_value=httpx.Response(200, json={}))
        result = source.lookup("Artist X", "Track A", None)
    assert result is None
    assert route.call_count == 0


def test_musicbrainz_genre_source_requires_app_contact():
    source = MusicBrainzGenreSource(app_contact=None)
    with pytest.raises(RuntimeError, match="MUSICBRAINZ_APP_CONTACT"):
        source.lookup("Artist X", "Track A", _MB_MBID)


# --------------------------------------------------------------------------- #
# DiscogsStyleSource (#122)
# --------------------------------------------------------------------------- #

_DISCOGS_URL = "https://api.discogs.com/database/search"


def test_discogs_style_source_returns_first_result_styles():
    source = DiscogsStyleSource(token="tok")
    body = {"results": [{"style": ["Thrash", "Speed Metal"]}]}
    with respx.mock(assert_all_called=False) as router:
        router.get(_DISCOGS_URL).mock(return_value=httpx.Response(200, json=body))
        tags = source.lookup("Artist X", "Track A", None)
    assert [t.tag for t in tags] == ["Thrash", "Speed Metal"]
    assert all(t.source == "discogs" for t in tags)


def test_discogs_style_source_empty_results_returns_none():
    source = DiscogsStyleSource(token="tok")
    with respx.mock(assert_all_called=False) as router:
        router.get(_DISCOGS_URL).mock(return_value=httpx.Response(200, json={"results": []}))
        result = source.lookup("Artist X", "Track A", None)
    assert result is None


def test_discogs_style_source_requires_token():
    source = DiscogsStyleSource(token=None)
    with pytest.raises(RuntimeError, match="DISCOGS_TOKEN"):
        source.lookup("Artist X", "Track A", None)


# --------------------------------------------------------------------------- #
# Derivation — happy path
# --------------------------------------------------------------------------- #


def test_derive_finds_two_scene_topics():
    records, plays = _two_tag_blobs()
    params = SceneParams(K_grid_explored=[2])

    d = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    assert isinstance(d, SceneDerivation)
    assert d.k_selected == 2
    assert d.coverage == 1.0
    assert d.n_tagged == 12
    assert len(d.outcome.roots) == 2
    assert {r.id for r in d.outcome.roots} == {"r-scene-1", "r-scene-2"}
    assert {r.category for r in d.outcome.roots} == {"scene"}

    # each topic recovers exactly its 3 defining tags, coherently
    tag_sets = {
        frozenset(td["tag"] for td in r.structural_descriptor["top_tags"]) for r in d.outcome.roots
    }
    assert tag_sets == {
        frozenset({"metal", "thrash", "heavy"}),
        frozenset({"jazz", "bebop", "swing"}),
    }
    for r in d.outcome.roots:
        assert r.structural_descriptor["coherence_score"] >= 0.15
        assert r.structural_descriptor["topic_index_in_K"] in {0, 1}
        # one artist per blob → one top_artist with all 6 tracks
        assert r.structural_descriptor["top_artists"] == [
            {"name": r.evidence.sample_tracks[0]["artist"], "count_in_topic": 6}
        ]
        # sample tracks carry topic_weight (not distance_to_centroid)
        assert all("topic_weight" in s for s in r.evidence.sample_tracks)


def test_derive_is_deterministic():
    records, plays = _two_tag_blobs()
    params = SceneParams(K_grid_explored=[2])
    a = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )
    b = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )
    assert [r.model_dump() for r in a.outcome.roots] == [r.model_dump() for r in b.outcome.roots]
    assert a.k_coherences == b.k_coherences


def test_stop_tags_are_filtered_out():
    records, plays = _two_tag_blobs()
    # pollute every metal track with library-management noise
    for i in range(6):
        rec = records[f"mbid:metal-{i}"]
        rec.tags.append(TrackTag(tag="seen live", weight=1.0, source="synthetic"))
        rec.tags.append(TrackTag(tag="favorite", weight=1.0, source="synthetic"))
    params = SceneParams(K_grid_explored=[2])

    d = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    all_tags = {td["tag"] for r in d.outcome.roots for td in r.structural_descriptor["top_tags"]}
    assert "seen live" not in all_tags
    assert "favorite" not in all_tags
    assert {"metal", "thrash", "heavy"} <= all_tags


# --------------------------------------------------------------------------- #
# Derivation — coherence floor + honest-empty
# --------------------------------------------------------------------------- #


def test_low_coherence_topic_rejected_to_quality_log():
    """Two coherent blobs + a scatter group of single-tag tracks. At K=3 the
    scatter collapses into one incoherent topic (its top tags never co-occur)
    → rejected to quality_log; the two real topics survive as roots."""
    records, plays = _two_tag_blobs()
    for i in range(5):
        tid = f"mbid:scatter-{i}"
        records[tid] = _record(tid, f"scatter {i}", "Misc", [f"oneoff{i}"])
        plays[tid] = [EARLY, LATE]
    params = SceneParams(K_grid_explored=[3], coherence_floor=0.15)

    d = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    assert d.k_selected == 3
    rejected = [q for q in d.outcome.quality_log if q.failed_test == "coherence_floor"]
    assert rejected, "expected at least one coherence-floor rejection"
    for q in rejected:
        assert q.category == "scene"
        assert q.details["K"] == 3
        assert q.details["coherence"] < 0.15
    # the two coherent blobs still surface as roots
    root_tags = {
        frozenset(td["tag"] for td in r.structural_descriptor["top_tags"]) for r in d.outcome.roots
    }
    assert frozenset({"metal", "thrash", "heavy"}) in root_tags
    assert frozenset({"jazz", "bebop", "swing"}) in root_tags


def test_no_k_passes_floor_is_honest_empty():
    """All tracks carry mutually-exclusive single tags → no co-occurrence at any
    K → no topic clears the floor → honest empty (no fabricated scene)."""
    records: dict[str, TrackMetadataRecord] = {}
    plays: dict[str, list[datetime]] = {}
    for i in range(12):
        tid = f"mbid:u-{i}"
        records[tid] = _record(tid, f"track {i}", f"Artist {i}", [f"unique{i}"])
        plays[tid] = [EARLY, LATE]
    params = SceneParams(K_grid_explored=[3, 5], coherence_floor=0.15)

    d = derive_scene_roots(
        records, plays, params=params, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    assert d.k_selected is None
    assert d.outcome.roots == []
    assert d.outcome.tendencies == []
    assert d.outcome.quality_log == []  # nothing selected → no per-topic rejection
    assert d.k_coherences  # but we still record what each K scored


def test_no_tags_at_all_is_honest_empty_with_zero_coverage():
    records: dict[str, TrackMetadataRecord] = {
        f"mbid:n-{i}": _record(f"mbid:n-{i}", f"t{i}", "A", None) for i in range(8)
    }
    plays = {tid: [EARLY, LATE] for tid in records}
    d = derive_scene_roots(
        records, plays, params=SceneParams(), validation_params=VP_OPEN, dataset_ctx=CTX
    )
    assert d.coverage == 0.0
    assert d.n_tagged == 0
    assert d.k_selected is None
    assert d.outcome.roots == []


def test_partial_tag_coverage_is_reported():
    records, plays = _two_tag_blobs()
    # drop tags from half the metal blob → 9/12 tagged
    for i in range(3):
        records[f"mbid:metal-{i}"].tags = []
    d = derive_scene_roots(
        records,
        plays,
        params=SceneParams(K_grid_explored=[2]),
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert d.n_tagged == 9
    assert d.coverage == 9 / 12


def test_report_is_a_dataclass_instance():
    # guards the public surface used by analyzer wiring
    assert isinstance(TagEnrichmentReport(), TagEnrichmentReport)
