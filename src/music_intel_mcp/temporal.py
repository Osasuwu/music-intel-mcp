"""Temporal root pipeline (#65) — the deep module behind the temporal category.

Temporal does **not** cluster in its own feature space. It *qualifies* roots that
audio/scene already derived (decision ``00f7e1eb``): given the upstream
``members`` map (root id → its member track ids) and the user's play timestamps,
it asks "when does this root fire more than its baseline?". Two halves:

1. **Lift-conditioned roots.** Every play is assigned a calendar *bucket*
   (day-part × weekday-kind × season — 32 by default, :func:`bucketize`). For
   each upstream root ``R`` and bucket ``b`` with enough in-bucket events
   (``event_count_floor`` — a *silent* precondition), the lift

       lift = P(R | b) / P(R)            (= conditional rate over baseline rate)

   measures over-representation. ``lift >= lift_floor`` promotes the (R, b) pair
   to a temporal :class:`~music_intel_mcp.validation.Candidate`
   (``r-temporal-N``) handed to the #62 validator; ``lift < lift_floor`` is
   *logged* (``q-temporal-N``, ``failed_test="lift_floor"``), never silently
   dropped (transparent-rejection invariant).

2. **Epochs.** A sliding-window two-sample KS test over the dominant-root-code
   timeline detects distribution change points; the contiguous segments between
   them become :class:`~music_intel_mcp.models.Epoch`s. Epochs live in
   ``RootProfile.epochs[]`` — they are *not* roots. Per Option B (matching the
   committed example), the change-point p-value is attached to the epoch the
   change point *ends* (the earlier of the adjacent pair); the final, most-recent
   epoch carries ``None``.

The ordering guard: temporal runs strictly after upstream derivation. With no
upstream members to condition on, the stage refuses (``skipped=True``) rather
than inventing a temporal space of its own.

Tests use synthetic play timestamps only — never a live API or dump.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from scipy.stats import ks_2samp

from .audio import _history_midpoint
from .models import Epoch, EpochParams, TemporalParams, ValidationParams
from .validation import (
    Candidate,
    DatasetContext,
    QualityLogEntry,
    ValidationOutcome,
    Validator,
)

# A KS window narrower than this carries too little signal to trust a p-value;
# the boundary is skipped rather than fed a degenerate two-sample test.
_MIN_WINDOW_EVENTS = 5


# --------------------------------------------------------------------------- #
# Calendar bucketing
# --------------------------------------------------------------------------- #


def _in_range(value: int, lo: int, hi: int, *, inclusive_end: bool) -> bool:
    """Membership in ``[lo, hi]`` (inclusive_end) or ``[lo, hi)`` (half-open),
    wrap-aware: when ``lo > hi`` the range wraps past the cycle boundary
    (e.g. night ``[23, 6)`` covers 23,0..5; winter ``[12, 2]`` covers 12,1,2)."""
    if lo <= hi:
        return lo <= value <= hi if inclusive_end else lo <= value < hi
    if inclusive_end:
        return value >= lo or value <= hi
    return value >= lo or value < hi


def _match(value: int, ranges: Mapping[str, Any], *, inclusive_end: bool) -> str:
    """First range in ``ranges`` containing ``value``; ``"unknown"`` if none
    (a mis-specified calendar degrades to an explicit unknown, not a crash)."""
    for name, bounds in ranges.items():
        lo, hi = bounds
        if _in_range(value, lo, hi, inclusive_end=inclusive_end):
            return name
    return "unknown"


def bucketize(played_at: datetime, calendar: Mapping[str, Any]) -> dict[str, str]:
    """Assign one play timestamp to its calendar bucket.

    Hours are half-open (``18:00`` is evening, not the tail of the afternoon);
    ISO weekday and month are inclusive. Each dimension falls back to
    ``"unknown"`` independently when the calendar leaves a gap.
    """
    return {
        "day_part": _match(played_at.hour, calendar["day_parts"], inclusive_end=False),
        "weekday_kind": _match(
            played_at.isoweekday(), calendar["weekday_kind"], inclusive_end=True
        ),
        "season": _match(played_at.month, calendar["seasons"], inclusive_end=True),
    }


def _balance(times: list[datetime], midpoint: datetime | None) -> float | None:
    """Chronological-split stability of a set of plays: ``1 - |first_half_frac -
    second_half_frac|``. ``None`` (→ not_evaluated) without a midpoint or events.
    Mirrors :func:`music_intel_mcp.audio._stability` but over an explicit times
    list (the in-bucket plays), not a member→plays lookup."""
    if midpoint is None or not times:
        return None
    before = sum(1 for t in times if t < midpoint)
    total = len(times)
    after = total - before
    return round(1 - abs(before / total - after / total), 4)


# --------------------------------------------------------------------------- #
# Epoch detection — sliding-window KS over the dominant-root-code timeline
# --------------------------------------------------------------------------- #


def _dominant_roots(seg_codes: list[int], code_to_root: dict[int, str], top: int = 2) -> list[str]:
    counts: dict[int, int] = {}
    for c in seg_codes:
        if c == 0:  # code 0 = "belongs to no surviving root" — not a dominant root
            continue
        counts[c] = counts.get(c, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], code_to_root[kv[0]]))
    return [code_to_root[c] for c, _ in ranked[:top]]


def _detect_epochs(
    times_sorted: list[datetime],
    codes_sorted: list[int],
    code_to_root: dict[int, str],
    params: EpochParams,
) -> tuple[list[Epoch], list[tuple[datetime, datetime, bool]]]:
    """Find distribution change points by sliding two adjacent ``window_days``
    windows in ``step_days`` strides and KS-testing their code distributions.
    Returns the epochs and parallel ``(lo, hi, is_last)`` range tuples (the
    callers need the half-open/closed semantics to count membership consistently).
    """
    n = len(times_sorted)
    if n == 0:
        return [], []
    start, end = times_sorted[0], times_sorted[-1]
    window = timedelta(days=params.window_days)
    step = timedelta(days=params.step_days)

    change_points: list[tuple[datetime, float]] = []
    last_cp = start
    b = start + step
    while b < end:
        lo = bisect.bisect_left(times_sorted, b - window)
        mid = bisect.bisect_left(times_sorted, b)
        hi = bisect.bisect_left(times_sorted, b + window)
        left = codes_sorted[lo:mid]
        right = codes_sorted[mid:hi]
        if len(left) >= _MIN_WINDOW_EVENTS and len(right) >= _MIN_WINDOW_EVENTS:
            p = float(ks_2samp(left, right, method="asymp").pvalue)
            # spacing guard: change points at least one window apart, so a single
            # transition is not double-counted by overlapping strides.
            if p < params.ks_significance_threshold and (b - last_cp) >= window:
                change_points.append((b, p))
                last_cp = b
        b += step

    boundaries = [start, *(cp[0] for cp in change_points), end]
    epochs: list[Epoch] = []
    ranges: list[tuple[datetime, datetime, bool]] = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        is_last = i == len(boundaries) - 2
        lo_idx = bisect.bisect_left(times_sorted, lo)
        hi_idx = (
            bisect.bisect_right(times_sorted, hi)
            if is_last
            else bisect.bisect_left(times_sorted, hi)
        )
        seg_codes = codes_sorted[lo_idx:hi_idx]
        # Option B: the epoch ENDING at a change point carries that p-value; the
        # final epoch (ends at `end`, not a change point) carries None.
        sig = change_points[i][1] if i < len(change_points) else None
        epochs.append(
            Epoch(
                id=f"e-{i + 1}",
                range=(lo, hi),
                n_events=len(seg_codes),
                change_point_in_significance=sig,
                dominant_roots=_dominant_roots(seg_codes, code_to_root),
            )
        )
        ranges.append((lo, hi, is_last))
    return epochs, ranges


def _count_in_range(times: list[datetime], lo: datetime, hi: datetime, *, is_last: bool) -> int:
    if is_last:
        return sum(1 for t in times if lo <= t <= hi)
    return sum(1 for t in times if lo <= t < hi)


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TemporalDerivation:
    """Result of :func:`derive_temporal_roots`.

    - ``outcome`` — validated temporal roots / tendencies + the lift-rejection
      quality log (prepended ahead of any validator rejections).
    - ``epochs`` — contiguous time segments for ``RootProfile.epochs[]``.
    - ``epoch_presence`` — upstream root id → {epoch id → fraction of that
      epoch's events belonging to the root}, to stamp onto each root.
    - ``skipped`` — ordering guard tripped (no upstream members); everything
      else is empty.
    """

    outcome: ValidationOutcome
    epochs: list[Epoch]
    epoch_presence: dict[str, dict[str, float]]
    skipped: bool = False


def derive_temporal_roots(
    members: Mapping[str, list[str]],
    track_plays: Mapping[str, list[datetime]],
    *,
    params: TemporalParams,
    epoch_params: EpochParams,
    validation_params: ValidationParams,
    dataset_ctx: DatasetContext,
) -> TemporalDerivation:
    """Qualify upstream roots by calendar bucket (lift) and detect epochs.

    ``members`` maps each surviving upstream root/tendency id to its member
    canonical track ids; ``track_plays`` is the full per-track timestamp map
    (its bucket totals form the lift denominators). With no members the ordering
    guard trips and the stage is skipped.
    """
    if not members:
        return TemporalDerivation(ValidationOutcome(), [], {}, skipped=True)

    calendar = params.temporal_calendar
    n_total = sum(len(v) for v in track_plays.values())
    midpoint = _history_midpoint(track_plays)

    # Bucket totals over ALL plays — the lift denominator P(R|b) = R-in-b / b-total.
    bucket_totals: dict[tuple[str, str, str], int] = {}
    bucket_dicts: dict[tuple[str, str, str], dict[str, str]] = {}
    for times in track_plays.values():
        for t in times:
            b = bucketize(t, calendar)
            key = (b["day_part"], b["weekday_kind"], b["season"])
            bucket_totals[key] = bucket_totals.get(key, 0) + 1
            bucket_dicts[key] = b

    root_ids_sorted = sorted(members)

    survivors: list[dict] = []
    rejects: list[dict] = []
    if n_total:
        for rid in root_ids_sorted:
            member_ids = list(dict.fromkeys(members[rid]))
            r_times = [t for tid in member_ids for t in track_plays.get(tid, [])]
            events_r = len(r_times)
            if events_r == 0:
                continue
            p_overall = events_r / n_total

            per_bucket: dict[tuple[str, str, str], list[datetime]] = {}
            for t in r_times:
                b = bucketize(t, calendar)
                per_bucket.setdefault((b["day_part"], b["weekday_kind"], b["season"]), []).append(t)

            for key, in_bucket in per_bucket.items():
                count = len(in_bucket)
                if count < params.event_count_floor:  # silent precondition, not logged
                    continue
                bucket_total = bucket_totals[key]
                p_rgb = count / bucket_total
                lift = p_rgb / p_overall
                record = {
                    "rid": rid,
                    "key": key,
                    "bucket": bucket_dicts[key],
                    "count": count,
                    "events_r": events_r,
                    "bucket_total": bucket_total,
                    "p_rgb": p_rgb,
                    "p_overall": p_overall,
                    "lift": lift,
                    "times_in": in_bucket,
                }
                (survivors if lift >= params.lift_floor else rejects).append(record)

    # Rank: strongest bucket first, ties broken deterministically.
    survivors.sort(key=lambda r: (-r["count"], r["rid"], r["key"]))
    rejects.sort(key=lambda r: (-r["count"], r["rid"], r["key"]))

    candidates: list[Candidate] = []
    for rank, r in enumerate(survivors, start=1):
        b = r["bucket"]
        descriptor = {
            "time_bucket": b,
            "conditioned_root_id": r["rid"],
            "lift": round(r["lift"], 4),
            "p_root_given_bucket": round(r["p_rgb"], 4),
            "p_root_overall": round(r["p_overall"], 4),
            "n_events_in_bucket": r["bucket_total"],
        }
        candidates.append(
            Candidate(
                candidate_id=f"r-temporal-{rank}",
                category="temporal",
                cluster_size=r["count"],
                cluster_share=round(r["count"] / r["events_r"], 6),
                evidence_count=r["count"],
                coverage=1.0,
                confidence=round(r["p_rgb"], 6),
                structural_descriptor=descriptor,
                temporal_stability_score=_balance(r["times_in"], midpoint),
                sample_tracks=[],
                actionability_hint=(
                    f"amplify root {r['rid']} with new candidates during "
                    f"{b['weekday_kind']} {b['season']} {b['day_part']}s"
                ),
            )
        )

    lift_rejections = [
        QualityLogEntry(
            candidate_id=f"q-temporal-{rank}",
            category="temporal",
            failed_test="lift_floor",
            details={
                "lift": round(r["lift"], 4),
                "floor": params.lift_floor,
                "conditioned_root_id": r["rid"],
                "time_bucket": r["bucket"],
                "p_root_given_bucket": round(r["p_rgb"], 4),
                "p_root_overall": round(r["p_overall"], 4),
                "n_events_in_bucket": r["bucket_total"],
                "events_in_bucket": r["count"],
            },
        )
        for rank, r in enumerate(rejects, start=1)
    ]

    outcome = Validator(validation_params).validate(candidates, dataset_ctx)
    outcome.quality_log = [*lift_rejections, *outcome.quality_log]

    # --- Epochs (independent of lift; run whenever there are events) --------- #
    code_of_root = {rid: i for i, rid in enumerate(root_ids_sorted, start=1)}
    code_to_root = {i: rid for rid, i in code_of_root.items()}
    code_of_track: dict[str, int] = {}
    for rid in root_ids_sorted:  # first sorted root a track belongs to wins its code
        for tid in members[rid]:
            code_of_track.setdefault(tid, code_of_root[rid])

    events: list[tuple[datetime, int]] = []
    for tid, times in track_plays.items():
        code = code_of_track.get(tid, 0)
        for t in times:
            events.append((t, code))
    events.sort(key=lambda e: e[0])
    times_sorted = [t for t, _ in events]
    codes_sorted = [c for _, c in events]

    epochs, ranges = _detect_epochs(times_sorted, codes_sorted, code_to_root, epoch_params)

    epoch_presence: dict[str, dict[str, float]] = {}
    for rid in root_ids_sorted:
        member_times = [t for tid in dict.fromkeys(members[rid]) for t in track_plays.get(tid, [])]
        pres: dict[str, float] = {}
        for epoch, (lo, hi, is_last) in zip(epochs, ranges, strict=True):
            total = epoch.n_events
            if total == 0:
                pres[epoch.id] = 0.0
                continue
            cnt = _count_in_range(member_times, lo, hi, is_last=is_last)
            pres[epoch.id] = round(cnt / total, 4)
        epoch_presence[rid] = pres

    return TemporalDerivation(outcome, epochs, epoch_presence, skipped=False)
