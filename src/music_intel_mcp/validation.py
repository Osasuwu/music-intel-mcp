"""Validation / classification core (#62).

The category-agnostic deep module every derivation pipeline feeds candidates
into. A *candidate* is a category-neutral summary of a derived pattern
(cluster size, coverage, confidence, a chronological-split stability score);
this module classifies each into exactly one of three buckets per decision
``bce66b6e``:

- **``root``** — clears every V0 test; the confident, stable, well-covered
  patterns the artifact is built around.
- **``tendency``** — a *real* pattern (enough evidence and coverage to not be
  noise) that fell short on at least one promotion test (confidence floor,
  coverage→root floor, or temporal stability). Surfaced separately, labelled
  with the tests it failed — never called a root.
- **``artifact_suspect``** — fails a hard floor (too little evidence, or
  coverage below the floor that distinguishes signal from noise). Suppressed
  from the user-facing sections but recorded in ``quality_log[]`` with the
  failed test and details. **Transparent rejection** — nothing is silently
  dropped.

The four V0 tests (decision ``bce66b6e``):

- **A — confidence floor.** ``confidence >= confidence_floor`` to be a root.
- **D — temporal stability (gated).** Evaluated only when the dataset is large
  and long enough (``n_unique_tracks > N_THRESHOLD`` AND
  ``history_span_days > T_THRESHOLD_DAYS``) *and* the candidate carries a split
  score. Otherwise ``not_evaluated`` and the candidate is forced to
  ``tendency`` — we never assert stability we could not measure.
- **E — coverage labelling.** Below the tendency floor → ``artifact_suspect``;
  between the tendency and root floors → capped at ``tendency``; at/above the
  root floor → eligible for ``root``.
- **G — calibration.** ``evidence_count >= evidence_count_floor`` — clusters
  too small to be statistically meaningful are artifacts.

Every threshold is read from :class:`ValidationParams` (``method_params``),
never hardcoded — they are placeholders until calibrated on real data in #66.
The module is verified against synthetic candidates; the audio/scene/temporal
pipelines (#63–#65) construct real :class:`Candidate`s and hand them here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import (
    Evidence,
    QualityLogEntry,
    Root,
    TemporalStability,
    Tendency,
    ValidationParams,
    ValidationScores,
)

Verdict = Literal["root", "tendency", "artifact_suspect"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """A category-neutral summary of one derived pattern. The derivation
    pipelines fill ``structural_descriptor`` (category-specific) and the
    evidence scalars; the validator reads only the scalars to classify, and
    passes the descriptor + samples through onto the resulting root/tendency."""

    candidate_id: str
    category: str
    cluster_size: int
    cluster_share: float
    evidence_count: int
    coverage: float
    confidence: float
    structural_descriptor: dict[str, Any]
    temporal_stability_score: float | None = None
    sample_tracks: list[dict[str, Any]] = field(default_factory=list)
    actionability_hint: str | None = None


@dataclass(frozen=True)
class DatasetContext:
    """Dataset-wide facts the temporal-stability gate (test D) depends on.
    Constant across a run, so it is passed once to :meth:`Validator.validate`
    rather than carried per candidate."""

    n_unique_tracks: int
    history_span_days: int


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Classified:
    """The verdict for one candidate. Exactly one of ``item`` (root/tendency)
    or ``rejected`` (artifact_suspect) is populated."""

    verdict: Verdict
    item: Root | Tendency | None = None
    rejected: QualityLogEntry | None = None


@dataclass
class ValidationOutcome:
    """The three RootProfile sections this module owns, ready to drop straight
    into a :class:`~music_intel_mcp.models.RootProfile`."""

    roots: list[Root] = field(default_factory=list)
    tendencies: list[Tendency] = field(default_factory=list)
    quality_log: list[QualityLogEntry] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


class Validator:
    """Classifies candidates into root / tendency / artifact_suspect using the
    thresholds in :class:`ValidationParams`."""

    def __init__(self, params: ValidationParams | None = None) -> None:
        self.params = params or ValidationParams()

    def _gate_open(self, ctx: DatasetContext) -> bool:
        """Test D gate: enough unique tracks AND long enough span to trust a
        chronological-split stability measurement."""
        p = self.params
        return ctx.n_unique_tracks > p.N_THRESHOLD and ctx.history_span_days > p.T_THRESHOLD_DAYS

    def classify(self, cand: Candidate, ctx: DatasetContext) -> Classified:
        p = self.params
        tendency_floor = p.coverage_floors["tendency"]
        root_floor = p.coverage_floors["root"]

        # --- Hard floors → artifact_suspect (suppressed, logged). G before E:
        #     no evidence is a more fundamental failure than thin coverage. ---
        if cand.evidence_count < p.evidence_count_floor:
            return self._reject(
                cand,
                "calibration",
                {"evidence_count": cand.evidence_count, "floor": p.evidence_count_floor},
            )
        if cand.coverage < tendency_floor:
            return self._reject(
                cand,
                "coverage_floor",
                {"coverage": cand.coverage, "floor_artifact": tendency_floor},
            )

        # --- Survivors are at least a tendency. Run the promotion tests. ---
        failed: list[str] = []

        coverage_pass = cand.coverage >= root_floor
        if not coverage_pass:
            failed.append("coverage_floor")

        confidence_pass = cand.confidence >= p.confidence_floor
        if not confidence_pass:
            failed.append("confidence_floor")

        # D — gated temporal stability. Evaluated only with an open gate AND a
        # score; otherwise not_evaluated, which forbids promotion to root.
        if self._gate_open(ctx) and cand.temporal_stability_score is not None:
            stability = TemporalStability(status="evaluated", score=cand.temporal_stability_score)
            if cand.temporal_stability_score < p.temporal_stability_floor:
                failed.append("temporal_stability")
        else:
            stability = TemporalStability(status="not_evaluated", score=None)
            failed.append("temporal_stability_not_evaluated")

        scores = ValidationScores(
            confidence=cand.confidence,
            temporal_stability=stability,
            coverage_pass=coverage_pass,
            confidence_pass=confidence_pass,
        )
        evidence = Evidence(
            cluster_size=cand.cluster_size,
            cluster_share=cand.cluster_share,
            evidence_count=cand.evidence_count,
            sample_tracks=cand.sample_tracks,
            coverage=cand.coverage,
        )

        if not failed:
            return Classified(
                verdict="root",
                item=Root(
                    id=cand.candidate_id,
                    category=cand.category,
                    classification="root",
                    structural_descriptor=cand.structural_descriptor,
                    evidence=evidence,
                    validation_scores=scores,
                    actionability_hint=cand.actionability_hint,
                ),
            )
        return Classified(
            verdict="tendency",
            item=Tendency(
                id=cand.candidate_id,
                category=cand.category,
                classification="tendency",
                structural_descriptor=cand.structural_descriptor,
                evidence=evidence,
                validation_scores=scores,
                actionability_hint=cand.actionability_hint,
                failed_tests=failed,
            ),
        )

    def _reject(self, cand: Candidate, failed_test: str, details: dict[str, Any]) -> Classified:
        return Classified(
            verdict="artifact_suspect",
            rejected=QualityLogEntry(
                candidate_id=cand.candidate_id,
                category=cand.category,
                failed_test=failed_test,
                details=details,
            ),
        )

    def validate(self, candidates: Sequence[Candidate], ctx: DatasetContext) -> ValidationOutcome:
        """Classify a batch, binning into the three RootProfile sections.
        Order within each section follows input order (stable, inspectable)."""
        outcome = ValidationOutcome()
        for cand in candidates:
            result = self.classify(cand, ctx)
            if result.verdict == "root":
                assert result.item is not None
                outcome.roots.append(result.item)
            elif result.verdict == "tendency":
                assert isinstance(result.item, Tendency)
                outcome.tendencies.append(result.item)
            else:
                assert result.rejected is not None
                outcome.quality_log.append(result.rejected)
        return outcome
