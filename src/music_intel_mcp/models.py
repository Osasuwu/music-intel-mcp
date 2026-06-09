"""V0 data schema — the canonical pydantic models.

Two cohesive concerns live here because both *are* the V0 contract every slice
reads and writes:

1. **Event schema** — source-agnostic listening event (`ListenEvent`). Track
   identity + timestamp + source tag + nullable play-context. Designed so the
   IFTTT extended-history source (which carries ms_played/skipped) and a thin
   Last.fm scrobble (which does not) both land without a schema rewrite.
2. **`RootProfile`** — the versioned artifact that *is* V0 output. Mirrors
   ``schemas/root_profile.v0.example.json`` (the canonical worked example);
   ``tests/test_models.py`` asserts the example validates against these models.

``method_params`` defaults live here as the single source of truth for the V0
threshold set. They are placeholders until calibrated empirically in #66.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Event schema (source-agnostic listening event)
# --------------------------------------------------------------------------- #


class TrackRef(BaseModel):
    """Track identity. Every id is optional; identity resolution (#61) fills
    the waterfall spotify_id -> ISRC -> MBID. ``name``/``artist`` are the
    always-present human-readable fallback used when no id resolves."""

    model_config = ConfigDict(extra="forbid")

    spotify_id: str | None = None
    isrc: str | None = None
    mbid: str | None = None
    name: str
    artist: str


class PlayContext(BaseModel):
    """Per-play context. Nullable as a whole (a scrobble source may omit it)
    and field-nullable within (a source may have ms_played but not skipped)."""

    model_config = ConfigDict(extra="forbid")

    ms_played: int | None = None
    skipped: bool | None = None


class ListenEvent(BaseModel):
    """One listening event from any source. The unit of the per-user history
    store (``data/history.jsonl``, one JSON object per line)."""

    model_config = ConfigDict(extra="forbid")

    track: TrackRef
    played_at: datetime
    source: str
    context: PlayContext | None = None


# --------------------------------------------------------------------------- #
# RootProfile — method_params (defaults = single source of truth for V0)
# --------------------------------------------------------------------------- #


class ValidationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    N_THRESHOLD: int = 1000
    T_THRESHOLD_DAYS: int = 180
    confidence_floor: float = 0.6
    evidence_count_floor: int = 50
    coverage_floors: dict[str, float] = Field(
        default_factory=lambda: {"root": 0.5, "tendency": 0.3, "artifact_suspect": 0.0}
    )


class AudioParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str = "HDBSCAN"
    min_cluster_size: int = 50
    min_samples: int = 10
    feature_set: list[str] = Field(
        default_factory=lambda: [
            "bpm",
            "energy",
            "valence",
            "danceability",
            "acousticness",
            "instrumentalness",
        ]
    )


class SceneParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str = "NMF"
    K_selected: int | None = None
    K_grid_explored: list[int] = Field(default_factory=lambda: [3, 5, 8, 13])
    coherence_floor: float = 0.15
    tag_canonicalization: Any = None
    stop_tags_version: str = "v1"


def _default_temporal_calendar() -> dict[str, Any]:
    return {
        "day_parts": {
            "morning": [6, 12],
            "day": [12, 18],
            "evening": [18, 23],
            "night": [23, 6],
        },
        "weekday_kind": {"weekday": [1, 5], "weekend": [6, 7]},
        "seasons": {
            "winter": [12, 2],
            "spring": [3, 5],
            "summer": [6, 8],
            "autumn": [9, 11],
        },
    }


class TemporalParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_calendar: dict[str, Any] = Field(default_factory=_default_temporal_calendar)
    lift_floor: float = 1.3
    event_count_floor: int = 30


class EpochParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int = 60
    step_days: int = 14
    ks_significance_threshold: float = 0.01


class MethodParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioParams = Field(default_factory=AudioParams)
    scene: SceneParams = Field(default_factory=SceneParams)
    temporal: TemporalParams = Field(default_factory=TemporalParams)
    epochs: EpochParams = Field(default_factory=EpochParams)
    validation: ValidationParams = Field(default_factory=ValidationParams)


# --------------------------------------------------------------------------- #
# RootProfile — top-level sections
# --------------------------------------------------------------------------- #


class GeneratedFrom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_range: tuple[datetime, datetime] | None = None
    n_events: int = 0
    n_unique_tracks: int = 0
    history_span_days: int = 0
    data_sources: list[str] = Field(default_factory=list)
    coverage_per_category: dict[str, float] = Field(default_factory=dict)


class CategoryWeights(BaseModel):
    """Per-category 0..1 driver weight. Inert in V0 (recorded, unused —
    decision 7b3adb41). V0 default = {audio,temporal,scene}=1, others null."""

    model_config = ConfigDict(extra="forbid")

    audio: float | None = 1.0
    temporal: float | None = 1.0
    scene: float | None = 1.0
    timbre: float | None = None
    career_phase: float | None = None
    lyrics: float | None = None
    structural: float | None = None
    context_chain: float | None = None


class TemporalStability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["evaluated", "not_evaluated"]
    score: float | None = None


class ValidationScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: float
    temporal_stability: TemporalStability
    coverage_pass: bool
    confidence_pass: bool


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_size: int
    cluster_share: float
    evidence_count: int
    sample_tracks: list[dict[str, Any]] = Field(default_factory=list)
    coverage: float


class Root(BaseModel):
    """A validated root. ``structural_descriptor`` is category-specific and
    free-form (the audio/scene/temporal pipelines each define their own shape);
    it is the machine-readable source of truth for what the root *is*."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: Literal["audio", "scene", "temporal"]
    classification: Literal["root", "tendency"]
    structural_descriptor: dict[str, Any]
    evidence: Evidence
    validation_scores: ValidationScores
    epoch_presence: dict[str, float] = Field(default_factory=dict)
    actionability_hint: str | None = None
    curator_prose: str | None = None
    caveats: list[str] = Field(default_factory=list)


class Tendency(Root):
    """Structurally identical to ``Root`` plus ``failed_tests`` — separated by
    section for cheap downstream dispatch, not by flag."""

    failed_tests: list[str] = Field(default_factory=list)


class Epoch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    range: tuple[datetime, datetime]
    n_events: int
    change_point_in_significance: float | None = None
    dominant_roots: list[str] = Field(default_factory=list)


class QualityLogEntry(BaseModel):
    """A transparently-rejected candidate. Mandatory under the
    transparent-rejection invariant — nothing is silently dropped."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    category: str
    failed_test: str
    details: dict[str, Any] = Field(default_factory=dict)


class Analytics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    novelty_curve: Any = None
    cluster_share_over_time: Any = None


class RootProfile(BaseModel):
    """The V0 output artifact. Source of truth for every downstream consumer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["v0"] = "v0"
    model_maturity: Literal["proxy", "full"] = "proxy"
    snapshot_id: str
    user_id: str
    generated_from: GeneratedFrom
    category_weights: CategoryWeights = Field(default_factory=CategoryWeights)
    method_params: MethodParams = Field(default_factory=MethodParams)
    roots: list[Root] = Field(default_factory=list)
    tendencies: list[Tendency] = Field(default_factory=list)
    epochs: list[Epoch] = Field(default_factory=list)
    quality_log: list[QualityLogEntry] = Field(default_factory=list)
    analytics: Analytics = Field(default_factory=Analytics)
