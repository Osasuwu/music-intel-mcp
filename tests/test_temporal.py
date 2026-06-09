"""Temporal root pipeline (#65): lift-qualified roots, the calendar buckets,
sliding-window KS epoch detection, per-root epoch_presence, and the ordering
guard.

Temporal does not cluster in its own space — it conditions on roots already
derived upstream (passed here as a ``members`` map id→track-ids). These tests
feed that map directly so the temporal math is exercised in isolation from
audio/scene (decision 00f7e1eb).

Dates are pinned to known weekdays so bucket assignment is exact: January 02:00
on a Mon–Fri is ``(night, weekday, winter)``; July 14:00 on a Mon–Fri is
``(day, weekday, summer)``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from music_intel_mcp.models import EpochParams, TemporalParams, ValidationParams
from music_intel_mcp.temporal import (
    TemporalDerivation,
    bucketize,
    derive_temporal_roots,
)
from music_intel_mcp.validation import DatasetContext

# Open the temporal-stability gate on small fixtures and lower the floors so the
# synthetic counts (tens, not thousands) can promote to roots.
VP_OPEN = ValidationParams(
    N_THRESHOLD=2,
    T_THRESHOLD_DAYS=30,
    evidence_count_floor=5,
    confidence_floor=0.5,
)
# event_count_floor=5 matches the relaxed evidence floor; default 30 would reject
# every synthetic pair.
TP = TemporalParams(event_count_floor=5)
CTX = DatasetContext(n_unique_tracks=4, history_span_days=760)

# Mon–Fri winter (January) 02:00 nights, across two winters for a midpoint-
# balanced stability score.
_WN_2024 = [datetime(2024, 1, d, 2, tzinfo=UTC) for d in (8, 9, 10, 11, 12)]  # Mon–Fri
_WN_2025 = [datetime(2025, 1, d, 2, tzinfo=UTC) for d in (6, 7)]  # Mon, Tue
_WN_2026 = [datetime(2026, 1, d, 2, tzinfo=UTC) for d in (5, 6, 7, 8, 9)]  # Mon–Fri
# Mon–Fri summer (July) 14:00 days, across two summers.
_SD_2024 = [datetime(2024, 7, d, 14, tzinfo=UTC) for d in (15, 16, 17, 18, 19)]  # Mon–Fri
_SD_2025 = [datetime(2025, 7, d, 14, tzinfo=UTC) for d in (14, 15, 16, 17, 18)]  # Mon–Fri


# --------------------------------------------------------------------------- #
# bucketize — the configurable calendar
# --------------------------------------------------------------------------- #


def test_default_calendar_is_32_buckets():
    cal = TemporalParams().temporal_calendar
    n = len(cal["day_parts"]) * len(cal["weekday_kind"]) * len(cal["seasons"])
    assert n == 32


def test_bucketize_wraps_night_and_winter():
    cal = TemporalParams().temporal_calendar
    # 2025-01-06 02:00 is a Monday 2am in January — both night and winter wrap.
    b = bucketize(datetime(2025, 1, 6, 2, 0, tzinfo=UTC), cal)
    assert b == {"day_part": "night", "weekday_kind": "weekday", "season": "winter"}


def test_bucketize_half_open_hours_and_weekend():
    cal = TemporalParams().temporal_calendar
    # 18:00 exactly is evening (day is [12,18), evening is [18,23)).
    sun_evening = bucketize(datetime(2025, 4, 6, 18, 0, tzinfo=UTC), cal)  # Sunday
    assert sun_evening["day_part"] == "evening"
    assert sun_evening["weekday_kind"] == "weekend"
    assert sun_evening["season"] == "spring"


def test_bucketize_respects_a_custom_calendar():
    cal = {
        "day_parts": {"am": [0, 12], "pm": [12, 24]},
        "weekday_kind": {"any": [1, 7]},
        "seasons": {"all": [1, 12]},
    }
    b = bucketize(datetime(2025, 3, 3, 9, 0, tzinfo=UTC), cal)
    assert b == {"day_part": "am", "weekday_kind": "any", "season": "all"}


# --------------------------------------------------------------------------- #
# Lift -> temporal roots
# --------------------------------------------------------------------------- #


def _lift_fixture() -> tuple[dict[str, list[str]], dict[str, list[datetime]]]:
    """Two upstream roots. r-audio-1 fires on winter weekday nights (10 events,
    5+5 across two winters → balanced about the midpoint), r-audio-2 on summer
    weekday days (10, balanced). Each leaks 2 events into the other's bucket so
    neither is a degenerate 100% conditional and the bucket totals are 12."""
    members = {"r-audio-1": ["t1"], "r-audio-2": ["t2"]}
    t1 = [*_WN_2024, *_WN_2026, *_SD_2024[3:5]]  # 10 winter-night + 2 summer leak
    t2 = [*_SD_2024, *_SD_2025, *_WN_2025]  # 10 summer-day + 2 winter leak
    return members, {"t1": t1, "t2": t2}


def test_lift_above_floors_becomes_temporal_root():
    members, plays = _lift_fixture()
    d = derive_temporal_roots(
        members,
        plays,
        params=TP,
        epoch_params=EpochParams(),
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert isinstance(d, TemporalDerivation)
    assert not d.skipped

    by_cond = {r.structural_descriptor["conditioned_root_id"]: r for r in d.outcome.roots}
    assert "r-audio-1" in by_cond, "winter-night pair should promote to a root"
    root = by_cond["r-audio-1"]
    assert root.category == "temporal"
    assert root.classification == "root"
    desc = root.structural_descriptor
    assert desc["time_bucket"] == {
        "day_part": "night",
        "weekday_kind": "weekday",
        "season": "winter",
    }
    # P(R)=12/24=0.5; P(R|winter-night)=10/12=0.833; lift=1.667.
    assert desc["p_root_overall"] == 0.5
    assert round(desc["p_root_given_bucket"], 2) == 0.83
    assert round(desc["lift"], 2) == 1.67
    assert desc["n_events_in_bucket"] == 12  # 10 from r1 + 2 leaked from r2
    assert root.evidence.evidence_count == 10
    assert root.evidence.coverage == 1.0
    # balanced across two winters -> stability evaluated and high
    assert root.validation_scores.temporal_stability.status == "evaluated"
    assert "amplify root r-audio-1" in root.actionability_hint


def test_lift_below_floor_is_logged_not_dropped():
    # t1 is globally common; in the summer-day bucket it is rare relative to its
    # baseline → lift < 1 even though the pair clears the event-count floor.
    members = {"r-audio-1": ["t1"], "r-audio-2": ["t2"]}
    plays = {
        "t1": [_WN_2024[0]] * 30 + [_SD_2024[0]] * 5,  # 30 winter, 5 summer
        "t2": [_SD_2025[0]] * 20,  # 20 summer
    }
    d = derive_temporal_roots(
        members,
        plays,
        params=TP,
        epoch_params=EpochParams(),
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    lift_fails = [q for q in d.outcome.quality_log if q.failed_test == "lift_floor"]
    assert lift_fails, "an above-count but below-lift pair must be logged"
    for q in lift_fails:
        assert q.category == "temporal"
        assert q.details["lift"] < TP.lift_floor


# --------------------------------------------------------------------------- #
# Epochs — sliding-window KS, in epochs[] not roots[]
# --------------------------------------------------------------------------- #


def _two_regime_fixture() -> tuple[dict[str, list[str]], dict[str, list[datetime]]]:
    """Regime 1 (days 0-50): only r-audio-1 plays. Gap. Regime 2 (days 70-120):
    only r-audio-2. A clean distributional change with a buffer so the KS
    windows on either side of the boundary are pure."""
    base = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    members = {"r-audio-1": ["tA"], "r-audio-2": ["tB"]}
    tA = [base + timedelta(days=d) for d in range(0, 51, 2)]  # 26 events
    tB = [base + timedelta(days=d) for d in range(70, 121, 2)]  # 26 events
    return members, {"tA": tA, "tB": tB}


# Relaxed epoch params: a ~120-day fixture needs a window short enough to fit two
# regimes (the default 60-day window + 14-day step would barely see one boundary).
EP_TIGHT = EpochParams(window_days=20, step_days=10, ks_significance_threshold=0.01)


def test_epochs_detected_via_ks_and_live_in_epochs_not_roots():
    members, plays = _two_regime_fixture()
    # event_count_floor huge -> no temporal roots, isolating epoch logic.
    no_roots = TemporalParams(event_count_floor=10_000)
    d = derive_temporal_roots(
        members,
        plays,
        params=no_roots,
        epoch_params=EP_TIGHT,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert d.outcome.roots == []  # epochs are NOT roots
    assert len(d.epochs) >= 2
    # the boundary epoch carries a significant change point; the final one does not
    changed = [e for e in d.epochs if e.change_point_in_significance is not None]
    assert changed, "at least one detected change point"
    for e in changed:
        assert e.change_point_in_significance < EP_TIGHT.ks_significance_threshold
    assert d.epochs[-1].change_point_in_significance is None
    # regime 1 dominated by r-audio-1, regime 2 by r-audio-2
    assert d.epochs[0].dominant_roots[:1] == ["r-audio-1"]
    assert d.epochs[-1].dominant_roots[:1] == ["r-audio-2"]


def test_epoch_presence_populated_per_root():
    members, plays = _two_regime_fixture()
    no_roots = TemporalParams(event_count_floor=10_000)
    d = derive_temporal_roots(
        members,
        plays,
        params=no_roots,
        epoch_params=EP_TIGHT,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert "r-audio-1" in d.epoch_presence
    pres = d.epoch_presence["r-audio-1"]
    epoch_ids = {e.id for e in d.epochs}
    assert set(pres) == epoch_ids
    # r-audio-1 owns the first epoch and is absent from the last
    assert pres[d.epochs[0].id] > 0.9
    assert pres[d.epochs[-1].id] < 0.1
    assert all(0.0 <= v <= 1.0 for v in pres.values())


# --------------------------------------------------------------------------- #
# Ordering guard
# --------------------------------------------------------------------------- #


def test_ordering_guard_skips_when_upstream_empty():
    _, plays = _two_regime_fixture()
    d = derive_temporal_roots(
        {},  # no upstream roots/tendencies to condition on
        plays,
        params=TP,
        epoch_params=EP_TIGHT,
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
    )
    assert d.skipped is True
    assert d.outcome.roots == []
    assert d.outcome.tendencies == []
    assert d.epochs == []
    assert d.epoch_presence == {}
