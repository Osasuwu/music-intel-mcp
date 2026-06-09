"""Validation / classification core (#62).

The category-agnostic deep module: a candidate (cluster_size / coverage /
confidence / temporal-split score) classifies into one of three —
``root`` / ``tendency`` / ``artifact_suspect`` — per the four V0 tests
(A confidence floor, D gated temporal stability, E coverage labelling,
G calibration). Verified against *synthetic* candidates; no real category
pipeline exists yet.
"""

from __future__ import annotations

from music_intel_mcp.models import ValidationParams
from music_intel_mcp.validation import Candidate, DatasetContext, Validator

# A dataset large/long enough to open the temporal-stability gate (test D):
#   n_unique_tracks > N_THRESHOLD (1000) AND history_span_days > T_THRESHOLD (180).
FULL_CTX = DatasetContext(n_unique_tracks=2000, history_span_days=400)
# Below either threshold -> gate closed -> temporal stability not_evaluated.
SMALL_CTX = DatasetContext(n_unique_tracks=100, history_span_days=30)


def candidate(**overrides) -> Candidate:
    """A would-be ``root``: passes every default floor. Tests override the one
    field under examination so the failing test is the only moving part."""
    base = dict(
        candidate_id="c-audio-1",
        category="audio",
        cluster_size=200,
        cluster_share=0.3,
        evidence_count=200,
        coverage=0.8,
        confidence=0.8,
        temporal_stability_score=0.9,
        structural_descriptor={"bpm_band": "high"},
    )
    base.update(overrides)
    return Candidate(**base)


# --------------------------------------------------------------------------- #
# Happy path — a candidate that clears every test is a root
# --------------------------------------------------------------------------- #


def test_clean_candidate_classifies_as_root():
    outcome = Validator().validate([candidate()], FULL_CTX)

    assert len(outcome.roots) == 1
    assert outcome.tendencies == []
    assert outcome.quality_log == []

    root = outcome.roots[0]
    assert root.classification == "root"
    assert root.category == "audio"
    assert root.validation_scores.coverage_pass is True
    assert root.validation_scores.confidence_pass is True
    assert root.validation_scores.temporal_stability.status == "evaluated"
    assert root.evidence.cluster_size == 200
    assert root.structural_descriptor == {"bpm_band": "high"}


# --------------------------------------------------------------------------- #
# Test A — confidence floor
# --------------------------------------------------------------------------- #


def test_below_confidence_floor_is_tendency_not_root():
    outcome = Validator().validate([candidate(confidence=0.4)], FULL_CTX)

    assert outcome.roots == []
    assert outcome.quality_log == []  # still a real pattern, not suppressed
    assert len(outcome.tendencies) == 1

    tend = outcome.tendencies[0]
    assert tend.classification == "tendency"
    assert "confidence_floor" in tend.failed_tests
    assert tend.validation_scores.confidence_pass is False
    assert tend.validation_scores.coverage_pass is True


# --------------------------------------------------------------------------- #
# Test D — temporal stability, gated by dataset size
# --------------------------------------------------------------------------- #


def test_temporal_stability_gated_off_forces_tendency():
    # Same clean candidate, but the dataset is too small/short to evaluate
    # stability -> not_evaluated -> forced to tendency (never a root).
    outcome = Validator().validate([candidate()], SMALL_CTX)

    assert outcome.roots == []
    assert len(outcome.tendencies) == 1
    tend = outcome.tendencies[0]
    assert tend.validation_scores.temporal_stability.status == "not_evaluated"
    assert tend.validation_scores.temporal_stability.score is None
    assert "temporal_stability_not_evaluated" in tend.failed_tests
    # coverage/confidence still genuinely passed — only stability is unproven
    assert tend.validation_scores.coverage_pass is True
    assert tend.validation_scores.confidence_pass is True


def test_temporal_stability_evaluated_but_low_is_tendency():
    outcome = Validator().validate([candidate(temporal_stability_score=0.2)], FULL_CTX)

    assert outcome.roots == []
    tend = outcome.tendencies[0]
    assert tend.validation_scores.temporal_stability.status == "evaluated"
    assert tend.validation_scores.temporal_stability.score == 0.2
    assert tend.failed_tests == ["temporal_stability"]


def test_temporal_stability_missing_score_under_open_gate_is_not_evaluated():
    # Gate open but the pipeline produced no split score -> cannot evaluate.
    outcome = Validator().validate([candidate(temporal_stability_score=None)], FULL_CTX)

    tend = outcome.tendencies[0]
    assert tend.validation_scores.temporal_stability.status == "not_evaluated"
    assert "temporal_stability_not_evaluated" in tend.failed_tests


# --------------------------------------------------------------------------- #
# Test E — coverage labelling (root floor / tendency floor / artifact floor)
# --------------------------------------------------------------------------- #


def test_coverage_below_artifact_floor_is_suppressed_artifact_suspect():
    # coverage 0.2 < tendency floor 0.3 -> too sparse to be a real pattern.
    outcome = Validator().validate([candidate(coverage=0.2)], FULL_CTX)

    assert outcome.roots == []
    assert outcome.tendencies == []  # suppressed from the user-facing artifact
    assert len(outcome.quality_log) == 1

    entry = outcome.quality_log[0]
    assert entry.candidate_id == "c-audio-1"
    assert entry.category == "audio"
    assert entry.failed_test == "coverage_floor"
    assert entry.details["coverage"] == 0.2
    assert entry.details["floor_artifact"] == 0.3


def test_coverage_between_tendency_and_root_floor_caps_at_tendency():
    # 0.3 <= coverage 0.4 < 0.5 -> real enough to surface, not enough for root.
    outcome = Validator().validate([candidate(coverage=0.4)], FULL_CTX)

    assert outcome.roots == []
    tend = outcome.tendencies[0]
    assert tend.validation_scores.coverage_pass is False
    assert "coverage_floor" in tend.failed_tests


# --------------------------------------------------------------------------- #
# Test G — calibration (evidence-count floor)
# --------------------------------------------------------------------------- #


def test_below_evidence_count_floor_is_artifact_suspect():
    # evidence_count 10 < floor 50 -> statistical artifact, suppressed + logged.
    outcome = Validator().validate([candidate(evidence_count=10)], FULL_CTX)

    assert outcome.roots == []
    assert outcome.tendencies == []
    assert len(outcome.quality_log) == 1
    entry = outcome.quality_log[0]
    assert entry.failed_test == "calibration"
    assert entry.details["evidence_count"] == 10
    assert entry.details["floor"] == 50


def test_calibration_checked_before_coverage_when_both_fail():
    # Fails both G (evidence) and E (coverage); calibration is the reported cause.
    outcome = Validator().validate([candidate(evidence_count=5, coverage=0.1)], FULL_CTX)
    assert outcome.quality_log[0].failed_test == "calibration"


# --------------------------------------------------------------------------- #
# Thresholds come from method_params, not hardcoded
# --------------------------------------------------------------------------- #


def test_thresholds_sourced_from_method_params():
    # A candidate that is a root under defaults (confidence 0.8) becomes a
    # tendency under a stricter custom confidence floor -> proves the floor is
    # read from params, not baked in.
    strict = ValidationParams(confidence_floor=0.9)
    outcome = Validator(strict).validate([candidate(confidence=0.8)], FULL_CTX)

    assert outcome.roots == []
    assert "confidence_floor" in outcome.tendencies[0].failed_tests


def test_custom_temporal_stability_floor_respected():
    lenient = ValidationParams(temporal_stability_floor=0.1)
    outcome = Validator(lenient).validate([candidate(temporal_stability_score=0.2)], FULL_CTX)
    # 0.2 >= custom floor 0.1 -> stability passes -> root
    assert len(outcome.roots) == 1


# --------------------------------------------------------------------------- #
# Honest-empty + transparent rejection over a batch
# --------------------------------------------------------------------------- #


def test_zero_candidates_is_honest_empty():
    outcome = Validator().validate([], FULL_CTX)
    assert outcome.roots == []
    assert outcome.tendencies == []
    assert outcome.quality_log == []


def test_all_candidates_rejected_yields_empty_roots_populated_quality_log():
    cands = [
        candidate(candidate_id="c-1", evidence_count=3),
        candidate(candidate_id="c-2", coverage=0.05),
    ]
    outcome = Validator().validate(cands, FULL_CTX)
    assert outcome.roots == []
    assert outcome.tendencies == []
    assert {e.candidate_id for e in outcome.quality_log} == {"c-1", "c-2"}


def test_mixed_batch_partitions_into_three_sections():
    cands = [
        candidate(candidate_id="c-root"),  # clean -> root
        candidate(candidate_id="c-tend", confidence=0.4),  # low conf -> tendency
        candidate(candidate_id="c-art", evidence_count=2),  # tiny -> artifact
    ]
    outcome = Validator().validate(cands, FULL_CTX)
    assert [r.id for r in outcome.roots] == ["c-root"]
    assert [t.id for t in outcome.tendencies] == ["c-tend"]
    assert [q.candidate_id for q in outcome.quality_log] == ["c-art"]


def test_multiple_failed_tests_accumulate_on_one_tendency():
    # Low confidence AND low (but evaluated) stability -> both recorded.
    outcome = Validator().validate(
        [candidate(confidence=0.4, temporal_stability_score=0.1)], FULL_CTX
    )
    failed = set(outcome.tendencies[0].failed_tests)
    assert {"confidence_floor", "temporal_stability"} <= failed
