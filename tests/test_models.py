"""RootProfile / event-schema model conformance."""

from __future__ import annotations

from music_intel_mcp.models import (
    Evidence,
    GeneratedFrom,
    ListenEvent,
    MethodParams,
    PlayContext,
    Root,
    RootProfile,
    TemporalStability,
    TrackRef,
    ValidationScores,
)


def test_canonical_example_validates(canonical_profile_dict):
    """The canonical worked example must validate against the models —
    this is the structural contract test the issue requires."""
    profile = RootProfile.model_validate(canonical_profile_dict)
    assert profile.schema_version == "v0"
    assert profile.model_maturity == "proxy"
    assert len(profile.roots) == 3
    assert {r.category for r in profile.roots} == {"audio", "scene", "temporal"}
    assert len(profile.tendencies) == 1
    assert profile.tendencies[0].failed_tests == ["temporal_stability"]
    assert len(profile.epochs) == 2
    assert len(profile.quality_log) == 2


def test_canonical_example_round_trips(canonical_profile_dict):
    """Parse -> dump -> re-parse is lossless and re-validates."""
    profile = RootProfile.model_validate(canonical_profile_dict)
    again = RootProfile.model_validate_json(profile.model_dump_json())
    assert again == profile


def test_method_params_defaults_match_canonical():
    """Default method_params equal the canonical example's threshold set —
    they are the single V0 source of truth (calibrated later in #66)."""
    mp = MethodParams()
    assert mp.validation.N_THRESHOLD == 1000
    assert mp.validation.T_THRESHOLD_DAYS == 180
    assert mp.validation.confidence_floor == 0.6
    assert mp.validation.coverage_floors == {
        "root": 0.5,
        "tendency": 0.3,
        "artifact_suspect": 0.0,
    }
    assert mp.audio.algorithm == "HDBSCAN"
    assert mp.scene.K_grid_explored == [3, 5, 8, 13]
    assert mp.temporal.lift_floor == 1.3
    assert mp.epochs.window_days == 60


def test_honest_empty_profile_is_valid():
    profile = RootProfile(
        snapshot_id="u/2025-01-01T00:00:00Z",
        user_id="u",
        generated_from=GeneratedFrom(),
    )
    assert profile.roots == []
    assert profile.tendencies == []
    assert profile.epochs == []
    assert profile.quality_log == []
    # category_weights inert-default: audio/temporal/scene=1, others null
    assert profile.category_weights.audio == 1.0
    assert profile.category_weights.timbre is None


def test_root_requires_structural_descriptor_and_evidence():
    root = Root(
        id="r-audio-1",
        category="audio",
        classification="root",
        structural_descriptor={"bpm_band": "high"},
        evidence=Evidence(cluster_size=10, cluster_share=0.5, evidence_count=10, coverage=0.9),
        validation_scores=ValidationScores(
            confidence=0.8,
            temporal_stability=TemporalStability(status="evaluated", score=0.9),
            coverage_pass=True,
            confidence_pass=True,
        ),
    )
    assert root.curator_prose is None
    assert root.caveats == []


def test_track_ref_ids_all_optional_but_name_artist_required():
    t = TrackRef(name="X", artist="Y")
    assert t.spotify_id is None and t.isrc is None and t.mbid is None


def test_play_context_fields_independently_nullable():
    assert PlayContext(ms_played=1000).skipped is None
    assert PlayContext(skipped=True).ms_played is None


def test_listen_event_context_optional():
    ev = ListenEvent.model_validate(
        {
            "track": {"name": "X", "artist": "Y"},
            "played_at": "2025-01-01T00:00:00Z",
            "source": "lastfm",
        }
    )
    assert ev.context is None
