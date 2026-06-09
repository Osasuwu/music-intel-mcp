"""Audio root pipeline (#63): AB-dump enricher → HDBSCAN → structural descriptor.

Two halves, both verified on *synthetic* features (never the real AB dump):

1. **Enricher** — looks audio features up by MBID and writes them back to the
   shared store, reporting coverage and transparently listing what it could not
   enrich (no MBID / no dump entry).
2. **Derivation** — z-scores the configured cluster dims, fits HDBSCAN, turns
   each cluster 1:1 into a ``Candidate`` with a complete structural descriptor
   (bands + raw centroid), evidence, a chronological-split stability score, and
   sample tracks, then hands the batch to the #62 validator.

Default thresholds gate the temporal-stability test off below 1000 tracks, so
these small fixtures pass a relaxed ``ValidationParams`` to open the gate — the
floors are params, calibrated for real in #66.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.audio import (
    AcousticBrainzDump,
    AudioDerivation,
    AudioEnrichmentReport,
    InMemoryAudioFeatureSource,
    derive_audio_roots,
    enrich_audio_features,
)
from music_intel_mcp.models import AudioParams, ValidationParams
from music_intel_mcp.shared_store import (
    AudioFeatures,
    InMemorySharedStore,
    TrackMetadataRecord,
)
from music_intel_mcp.validation import DatasetContext

NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
EARLY = datetime(2025, 1, 1, tzinfo=UTC)
LATE = datetime(2025, 6, 1, tzinfo=UTC)
# One play in each history half -> balanced -> stability ~1.0 (well above floor).
BALANCED_PLAYS = [datetime(2025, 1, 20, tzinfo=UTC), datetime(2025, 5, 20, tzinfo=UTC)]

# Relaxed validation params: open the temporal gate and lower the evidence floor
# so a 6-track fixture cluster can legitimately reach `root`.
VP_OPEN = ValidationParams(
    N_THRESHOLD=5,
    T_THRESHOLD_DAYS=5,
    evidence_count_floor=5,
    confidence_floor=0.5,
)
AP = AudioParams(min_cluster_size=5, min_samples=2)
CTX = DatasetContext(n_unique_tracks=12, history_span_days=150)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _features(bpm, energy, valence, dance, *, acoustic=0.5, instr=0.1):
    return AudioFeatures(
        bpm=bpm,
        energy=energy,
        valence=valence,
        danceability=dance,
        acousticness=acoustic,
        instrumentalness=instr,
        source="synthetic",
    )


def _rec(tid, features=None, *, mbid=None, name="N", artist="A"):
    return TrackMetadataRecord(
        track_id=tid,
        mbid=mbid,
        name=name,
        artist=artist,
        audio_features=features,
        fetched_at=NOW,
    )


def _two_blobs(*, full_coverage=True):
    """12 tracks: a tight low/low/low/low blob and a tight high/high/high/high
    blob, far apart after z-scoring -> two clean HDBSCAN clusters, no noise."""
    records: dict[str, TrackMetadataRecord] = {}
    for i in range(6):
        tid = f"mbid:low-{i}"
        records[tid] = _rec(
            tid,
            _features(78 + i * 0.4, 0.18 + i * 0.004, 0.22 + i * 0.004, 0.28 + i * 0.004),
            mbid=f"low-{i}",
        )
    for i in range(6):
        tid = f"mbid:high-{i}"
        records[tid] = _rec(
            tid,
            _features(170 + i * 0.4, 0.88 + i * 0.004, 0.80 + i * 0.004, 0.83 + i * 0.004),
            mbid=f"high-{i}",
        )
    return records


def _balanced_plays(records):
    return {tid: list(BALANCED_PLAYS) for tid in records}


def _by_band(roots, band_value):
    return next(r for r in roots if r.structural_descriptor["bpm_band"] == band_value)


# --------------------------------------------------------------------------- #
# Enricher
# --------------------------------------------------------------------------- #


def test_enricher_populates_features_by_mbid_and_reports_coverage():
    store = InMemorySharedStore(
        [
            _rec("mbid:a", mbid="a"),
            _rec("mbid:b", mbid="b"),
            _rec("mbid:c", mbid="c"),  # no dump entry -> missing_features
        ]
    )
    source = InMemoryAudioFeatureSource(
        {
            "a": _features(120, 0.5, 0.5, 0.5),
            "b": _features(90, 0.3, 0.4, 0.3),
        }
    )

    report = enrich_audio_features(["mbid:a", "mbid:b", "mbid:c"], store, source, now=NOW)

    assert isinstance(report, AudioEnrichmentReport)
    assert set(report.enriched) == {"mbid:a", "mbid:b"}
    assert report.missing_features == ["mbid:c"]
    assert report.coverage == 2 / 3
    # written back to the store
    stored = store.get_tracks(["mbid:a", "mbid:c"])
    assert stored["mbid:a"].audio_features is not None
    assert stored["mbid:a"].audio_features.bpm == 120
    assert stored["mbid:c"].audio_features is None


def test_enricher_flags_tracks_without_mbid_and_never_looks_them_up():
    store = InMemorySharedStore([_rec("name:x\x1fy")])  # name-only, no mbid
    source = InMemoryAudioFeatureSource({"a": _features(120, 0.5, 0.5, 0.5)})

    report = enrich_audio_features(["name:x\x1fy"], store, source, now=NOW)

    assert report.no_mbid == ["name:x\x1fy"]
    assert report.enriched == []
    assert source.lookups == []  # transparent skip, no wasted lookup
    assert report.coverage == 0.0


def test_enricher_skips_tracks_already_carrying_features():
    store = InMemorySharedStore([_rec("mbid:a", _features(120, 0.5, 0.5, 0.5), mbid="a")])
    source = InMemoryAudioFeatureSource({"a": _features(999, 0.1, 0.1, 0.1)})

    report = enrich_audio_features(["mbid:a"], store, source, now=NOW)

    assert source.lookups == []  # already enriched -> no lookup
    assert report.coverage == 1.0
    # existing features untouched (not overwritten with the dump's value)
    assert store.get_tracks(["mbid:a"])["mbid:a"].audio_features.bpm == 120


def test_acousticbrainz_dump_reads_jsonl_and_misses_gracefully(tmp_path: Path):
    dump = tmp_path / "ab.jsonl"
    dump.write_text(
        json.dumps({"mbid": "a", "bpm": 168, "energy": 0.81, "valence": 0.28, "danceability": 0.62})
        + "\n",
        encoding="utf-8",
    )
    src = AcousticBrainzDump(path=dump)

    hit = src.lookup("a")
    assert hit is not None and hit.bpm == 168 and hit.source == "acousticbrainz_dump"
    assert src.lookup("absent") is None  # unknown mbid


def test_acousticbrainz_dump_missing_file_is_empty_not_error(tmp_path: Path):
    # honest low coverage beats a crash when the dump isn't installed
    src = AcousticBrainzDump(path=tmp_path / "does-not-exist.jsonl")
    assert src.lookup("anything") is None


# --------------------------------------------------------------------------- #
# Derivation — clustering, descriptor, validation integration
# --------------------------------------------------------------------------- #


def test_two_blobs_yield_two_roots_with_complete_structural_descriptor():
    records = _two_blobs()
    result = derive_audio_roots(
        records,
        _balanced_plays(records),
        params=AP,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )

    assert isinstance(result, AudioDerivation)
    assert result.n_clusters == 2
    assert result.n_noise == 0
    assert len(result.outcome.roots) == 2
    assert result.outcome.tendencies == []
    assert result.outcome.quality_log == []

    high = _by_band(result.outcome.roots, "high")
    low = _by_band(result.outcome.roots, "low")

    # complete descriptor: four bands + raw centroid
    for root, band in ((high, "high"), (low, "low")):
        d = root.structural_descriptor
        assert d["bpm_band"] == band
        assert d["energy_band"] == band
        assert d["valence_band"] == band
        assert d["danceability_band"] == band
        assert set(d["centroid_raw"]) == {"bpm", "energy", "valence", "danceability"}

    # raw centroid reflects the blob it came from
    assert high.structural_descriptor["centroid_raw"]["bpm"] > 140
    assert low.structural_descriptor["centroid_raw"]["bpm"] < 100


def test_each_root_carries_evidence_validation_and_sample_tracks():
    records = _two_blobs()
    result = derive_audio_roots(
        records,
        _balanced_plays(records),
        params=AP,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    root = result.outcome.roots[0]

    assert root.evidence.cluster_size == 6
    assert root.evidence.evidence_count == 6
    assert 0.0 < root.evidence.cluster_share <= 0.5
    assert root.evidence.coverage == 1.0  # every fixture track has all 6 features
    # sample tracks carry identity + distance to centroid
    assert root.evidence.sample_tracks
    sample = root.evidence.sample_tracks[0]
    assert {"track_id", "name", "artist", "distance_to_centroid"} <= set(sample)
    # validated as a root: stability evaluated (open gate), both passes true
    assert root.validation_scores.temporal_stability.status == "evaluated"
    assert root.validation_scores.coverage_pass is True
    assert root.validation_scores.confidence_pass is True


def test_root_ids_rank_clusters_by_size_then_id_deterministically():
    records = _two_blobs()
    result = derive_audio_roots(
        records,
        _balanced_plays(records),
        params=AP,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert {r.id for r in result.outcome.roots} == {"r-audio-1", "r-audio-2"}


def test_derivation_is_reproducible_given_same_inputs():
    records = _two_blobs()
    plays = _balanced_plays(records)
    a = derive_audio_roots(records, plays, params=AP, validation_params=VP_OPEN, dataset_ctx=CTX)
    b = derive_audio_roots(records, plays, params=AP, validation_params=VP_OPEN, dataset_ctx=CTX)

    dump = lambda res: [r.model_dump() for r in res.outcome.roots]  # noqa: E731
    assert dump(a) == dump(b)


def test_time_skewed_cluster_fails_temporal_stability_and_becomes_tendency():
    records = _two_blobs()
    plays = _balanced_plays(records)
    # push the whole LOW blob into the first history half -> stability 0.0
    for i in range(6):
        plays[f"mbid:low-{i}"] = [datetime(2025, 1, 10, tzinfo=UTC)]

    result = derive_audio_roots(
        records, plays, params=AP, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    assert len(result.outcome.roots) == 1  # the balanced HIGH blob
    assert result.outcome.roots[0].structural_descriptor["bpm_band"] == "high"
    assert len(result.outcome.tendencies) == 1
    tend = result.outcome.tendencies[0]
    assert tend.structural_descriptor["bpm_band"] == "low"
    assert "temporal_stability" in tend.failed_tests


def test_evidence_count_floor_suppresses_small_clusters_to_quality_log():
    records = _two_blobs()
    # default evidence_count_floor=50 >> 6-track clusters -> artifact_suspect
    strict = ValidationParams(N_THRESHOLD=5, T_THRESHOLD_DAYS=5)
    result = derive_audio_roots(
        records, _balanced_plays(records), params=AP, validation_params=strict, dataset_ctx=CTX
    )

    assert result.outcome.roots == []
    assert result.outcome.tendencies == []
    assert len(result.outcome.quality_log) == 2
    assert all(q.failed_test == "calibration" for q in result.outcome.quality_log)


def test_partial_feature_coverage_downgrades_cluster_to_tendency():
    records = _two_blobs()
    # strip the two non-cluster dims from 4 of 6 HIGH-blob tracks -> coverage 2/6
    for i in range(2, 6):
        f = records[f"mbid:high-{i}"].audio_features
        records[f"mbid:high-{i}"].audio_features = f.model_copy(
            update={"acousticness": None, "instrumentalness": None}
        )

    result = derive_audio_roots(
        records, _balanced_plays(records), params=AP, validation_params=VP_OPEN, dataset_ctx=CTX
    )

    high = _by_band(result.outcome.tendencies, "high")
    assert high.evidence.coverage == 2 / 6
    assert "coverage_floor" in high.failed_tests
    assert high.validation_scores.coverage_pass is False
    # the fully-covered LOW blob is still a root
    assert any(r.structural_descriptor["bpm_band"] == "low" for r in result.outcome.roots)


def test_no_clusterable_records_is_honest_empty():
    # records missing a cluster dim cannot be placed -> nothing to cluster
    records = {f"mbid:x-{i}": _rec(f"mbid:x-{i}", None, mbid=f"x-{i}") for i in range(8)}
    result = derive_audio_roots(records, {}, params=AP, validation_params=VP_OPEN, dataset_ctx=CTX)
    assert result.n_clustered == 0
    assert result.coverage == 0.0
    assert result.outcome.roots == []
    assert result.outcome.tendencies == []
    assert result.outcome.quality_log == []


def test_coverage_is_fraction_of_records_with_usable_features():
    records = _two_blobs()  # 12 clusterable
    # add 4 records with no features -> 12/16 clusterable
    for i in range(4):
        records[f"mbid:none-{i}"] = _rec(f"mbid:none-{i}", None, mbid=f"none-{i}")
    result = derive_audio_roots(
        records, _balanced_plays(records), params=AP, validation_params=VP_OPEN, dataset_ctx=CTX
    )
    assert result.coverage == 12 / 16
