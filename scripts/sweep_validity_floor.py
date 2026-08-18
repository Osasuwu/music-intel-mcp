"""Offline threshold-sensitivity sweep for the play-validity floor (#95).

Reuses the production audio -> scene pipeline exactly once to reconstruct the
upstream ``members`` map (root id -> member canonical track ids), then re-runs
ONLY the temporal seed/derive step at each candidate ``MIN_VALID_MS`` floor
(:data:`FLOOR_CONFIGS`), rebuilding the validity-filtered play map via
:func:`music_intel_mcp.temporal._is_valid`'s ``min_valid_ms`` seam (decision
``c69c6f17``, AC1). This measures how sensitive the play-validity floor is —
it does NOT change any production threshold (AC9); that calibration call
belongs to #98, `/grill`-gated there.

Fully offline (AC2): reads ``history.jsonl`` + the local shared-metadata cache
+ the env-pointed AcousticBrainz/MusicBrainz dumps; the scene stage uses an
empty :class:`~music_intel_mcp.scene.InMemoryTagSource` (only tags already
cached in the shared store surface) so no Last.fm network call ever fires.

Usage::

    python scripts/sweep_validity_floor.py --data-dir data --user-id petr \\
        --out data/validity_floor_sweep_report.json

The report is written to a **local file under the data dir and never
committed** (history-never-leaves-the-machine invariant) — only a summary
table is meant to be posted as a GitHub comment on issue #98 (AC8).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from music_intel_mcp.analyzer import (
    _dataset_ctx,
    _generated_from,
    _group_and_seed,
    _localize,
    _modal_zone,
    _resolve_ref,
)
from music_intel_mcp.audio import AcousticBrainzDump, derive_audio_roots, enrich_audio_features
from music_intel_mcp.identity import IdentityCache, IdentityResolver, MusicBrainzIsrcIndex
from music_intel_mcp.models import ListenEvent, MethodParams, PlayContext, TrackRef
from music_intel_mcp.scene import InMemoryTagSource, derive_scene_roots, enrich_tags
from music_intel_mcp.shared_store import LocalSharedStore, canonical_track_id
from music_intel_mcp.store import UserStore
from music_intel_mcp.temporal import TemporalDerivation, bucketize, derive_temporal_roots
from music_intel_mcp.validation import DatasetContext

# The floors under sweep (AC3): eight ms thresholds plus one extra run at the
# production floor (30_000ms) with the skip-flag predicate ignored, so the
# cross-tab (AC4) can separate the two _is_valid predicates' individual effects.
FLOOR_CONFIGS: list[tuple[str, int, bool]] = [
    (f"{ms}ms", ms, False) for ms in (0, 5_000, 10_000, 20_000, 30_000, 45_000, 60_000, 90_000)
]
FLOOR_CONFIGS.append(("30000ms_ignore_skip", 30_000, True))

_EVENING_WINDOW_START = 16  # pre-registered 16-19h local evening peak (AC6.i)
_EVENING_WINDOW_TOLERANCE_H = 1  # #98 CRITIC loopback: exact-window match was too
# strict to survive a real 0.3%-margin near-tie (17h peak vs 16h window) and failed
# identically at every floor including floor=0 — carrying no discriminating signal.
# A cyclic ±1h tolerance keeps the check meaningful without loosening it into a no-op.


# --------------------------------------------------------------------------- #
# Validity predicate variants (measurement seam only — never used in production)
# --------------------------------------------------------------------------- #


def _is_valid_at(
    context: PlayContext | None, *, min_valid_ms: int, ignore_skip: bool = False
) -> bool:
    """Sweep-local mirror of ``temporal._is_valid`` with an extra ``ignore_skip``
    knob for the floor=30000/skip-ignored run (AC3). Never imported by
    production code — the seam production code calls is ``temporal._is_valid``
    itself (AC1)."""
    if context is None:
        return True
    if context.ms_played is not None and context.ms_played < min_valid_ms:
        return False
    if ignore_skip:
        return True
    return context.skipped is not True


# --------------------------------------------------------------------------- #
# One localization pass, reused across every floor (AC2)
# --------------------------------------------------------------------------- #


def _localized_rows(
    events: Sequence[ListenEvent],
    resolver: IdentityResolver | None,
    resolved_cache: dict[str, TrackRef],
) -> list[tuple[str, datetime, PlayContext | None, str]]:
    """(canonical id, local wall-clock datetime, context, source) per event —
    computed once via the production ``_localize``/``_modal_zone``, then reused
    to rebuild the filtered map cheaply at every floor."""
    modal_zone = _modal_zone(events)
    rows = []
    for event in events:
        cid = canonical_track_id(_resolve_ref(event.track, resolver, resolved_cache))
        local, _shifted = _localize(event, modal_zone)
        rows.append((cid, local, event.context, event.source))
    return rows


def _filtered_map(
    rows: list[tuple[str, datetime, PlayContext | None, str]],
    *,
    min_valid_ms: int,
    ignore_skip: bool = False,
) -> dict[str, list[datetime]]:
    out: dict[str, list[datetime]] = {}
    for cid, local, context, _source in rows:
        if _is_valid_at(context, min_valid_ms=min_valid_ms, ignore_skip=ignore_skip):
            out.setdefault(cid, []).append(local)
    return out


def _unfiltered_map(
    rows: list[tuple[str, datetime, PlayContext | None, str]],
) -> dict[str, list[datetime]]:
    out: dict[str, list[datetime]] = {}
    for cid, local, _context, _source in rows:
        out.setdefault(cid, []).append(local)
    return out


# --------------------------------------------------------------------------- #
# Epoch invariance guard (decision 77111ce0, AC5)
# --------------------------------------------------------------------------- #


def _epoch_signature(epochs) -> tuple:
    return tuple((e.range[0].isoformat(), e.range[1].isoformat(), e.n_events) for e in epochs)


class EpochInvarianceViolation(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Union-of-candidates verdict table (AC4)
# --------------------------------------------------------------------------- #

CandidateKey = tuple[str, tuple[str, str, str]]


def _bucket_key(b: Mapping[str, str]) -> tuple[str, str, str]:
    return (b["day_part"], b["weekday_kind"], b["season"])


def _all_bucket_counts(
    members: Mapping[str, list[str]],
    filtered: Mapping[str, list[datetime]],
    calendar: Mapping[str, Any],
) -> dict[CandidateKey, int]:
    """Per (root id, bucket) event counts irrespective of ``event_count_floor``
    — mirrors ``temporal.derive_temporal_roots``'s internal loop so the harness
    can attribute the *silent* count-floor drop (temporal.py:320) that no
    QualityLogEntry records."""
    counts: dict[CandidateKey, int] = {}
    for rid, member_ids in members.items():
        member_ids = list(dict.fromkeys(member_ids))
        r_times = [t for tid in member_ids for t in filtered.get(tid, [])]
        for t in r_times:
            key = (rid, _bucket_key(bucketize(t, calendar)))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _floor_candidate_table(
    members: Mapping[str, list[str]],
    filtered: Mapping[str, list[datetime]],
    result: TemporalDerivation,
    event_count_floor: int,
    calendar: Mapping[str, Any],
) -> dict[CandidateKey, dict[str, Any]]:
    table: dict[CandidateKey, dict[str, Any]] = {}
    for item in (*result.outcome.roots, *result.outcome.tendencies):
        d = item.structural_descriptor
        key = (d["conditioned_root_id"], _bucket_key(d["time_bucket"]))
        table[key] = {
            "verdict": item.classification,
            "lift": d["lift"],
            "evidence_count": item.evidence.evidence_count,
        }
    for q in result.outcome.quality_log:
        if q.failed_test != "lift_floor":
            continue
        d = q.details
        key = (d["conditioned_root_id"], _bucket_key(d["time_bucket"]))
        table[key] = {
            "verdict": "rejected:lift_floor",
            "lift": d["lift"],
            "evidence_count": d["events_in_bucket"],
        }
    for key, count in _all_bucket_counts(members, filtered, calendar).items():
        if count < event_count_floor and key not in table:
            table[key] = {
                "verdict": "dropped:event_count_floor",
                "lift": None,
                "evidence_count": count,
            }
    return table


def _agreement_rate(
    table_a: dict[CandidateKey, dict[str, Any]], table_b: dict[CandidateKey, dict[str, Any]]
) -> tuple[float, int]:
    keys = set(table_a) | set(table_b)
    if not keys:
        return 1.0, 0
    agree = 0
    for key in keys:
        va = table_a.get(key, {"verdict": "absent"})["verdict"]
        vb = table_b.get(key, {"verdict": "absent"})["verdict"]
        if va == vb:
            agree += 1
    return agree / len(keys), len(keys)


# --------------------------------------------------------------------------- #
# Quality criteria (AC6)
# --------------------------------------------------------------------------- #


def _hour_histogram(times: Sequence[datetime]) -> dict[str, int]:
    hist = Counter(t.hour for t in times)
    return {str(h): hist.get(h, 0) for h in range(24)}


def _evening_peak_survives(hour_hist: Mapping[str, int]) -> bool:
    """Pre-registered check (AC6.i), amended by #98 CRITIC loopback: does the
    largest 3-hour rolling total across the (cyclic) 24h day start within
    ``_EVENING_WINDOW_TOLERANCE_H`` hours of the 16-19h local evening window?
    Checked against the filtered histogram at every floor — a floor that
    erases the known evening peak is over-aggressive."""
    counts = [hour_hist[str(h)] for h in range(24)]
    if sum(counts) == 0:
        return False
    windows = [(sum(counts[(start + i) % 24] for i in range(3)), start) for start in range(24)]
    best_start = max(windows)[1]
    cyclic_distance = min(
        (best_start - _EVENING_WINDOW_START) % 24, (_EVENING_WINDOW_START - best_start) % 24
    )
    return cyclic_distance <= _EVENING_WINDOW_TOLERANCE_H


# --------------------------------------------------------------------------- #
# Pipeline wiring (mirrors analyzer.analyze / cli._cmd_analyze, AC2)
# --------------------------------------------------------------------------- #


def _reconstruct_members(
    events: Sequence[ListenEvent],
    shared_store: LocalSharedStore,
    audio_source: AcousticBrainzDump,
    resolver: IdentityResolver,
    now: datetime,
) -> tuple[dict[str, list[str]], dict[str, list[datetime]], DatasetContext, dict[str, TrackRef]]:
    """Run the audio + scene stages exactly once to get the surviving upstream
    ``members`` map temporal conditions on. Scene uses an empty in-memory tag
    source — offline by construction (AC2): only tags already cached in the
    shared store surface, never a live Last.fm call."""
    resolved_cache: dict[str, TrackRef] = {}
    plays, unique_ids = _group_and_seed(events, shared_store, now, resolver, resolved_cache)
    generated_from = _generated_from(events)
    ctx = _dataset_ctx(generated_from)
    params = MethodParams()

    enrich_audio_features(unique_ids, shared_store, audio_source, now=now)
    records = shared_store.get_tracks(unique_ids)
    audio = derive_audio_roots(
        records, plays, params=params.audio, validation_params=params.validation, dataset_ctx=ctx
    )

    enrich_tags(unique_ids, shared_store, InMemoryTagSource(), now=now)
    records = shared_store.get_tracks(unique_ids)
    scene = derive_scene_roots(
        records, plays, params=params.scene, validation_params=params.validation, dataset_ctx=ctx
    )

    surviving = {item.id for item in (*audio.outcome.roots, *audio.outcome.tendencies)}
    surviving |= {item.id for item in (*scene.outcome.roots, *scene.outcome.tendencies)}
    members = {**audio.members, **scene.members}
    members = {rid: ids for rid, ids in members.items() if rid in surviving}
    return members, plays, ctx, resolved_cache


def _run_floors(
    members: Mapping[str, list[str]],
    rows: list[tuple[str, datetime, PlayContext | None, str]],
    unfiltered: Mapping[str, list[datetime]],
    ctx: DatasetContext,
    params: MethodParams,
) -> dict[str, dict[str, Any]]:
    calendar = params.temporal.temporal_calendar
    results: dict[str, dict[str, Any]] = {}
    epoch_sig_ref: tuple | None = None
    for label, min_valid_ms, ignore_skip in FLOOR_CONFIGS:
        filtered = _filtered_map(rows, min_valid_ms=min_valid_ms, ignore_skip=ignore_skip)
        result = derive_temporal_roots(
            members,
            filtered,
            params=params.temporal,
            epoch_params=params.epochs,
            validation_params=params.validation,
            dataset_ctx=ctx,
            epoch_plays=unfiltered,
        )
        sig = _epoch_signature(result.epochs)
        if epoch_sig_ref is None:
            epoch_sig_ref = sig
        elif sig != epoch_sig_ref:
            raise EpochInvarianceViolation(
                f"epoch boundaries shifted at floor {label!r} — validity filtering must never "
                "move an epoch boundary (decision 77111ce0). Aborting the sweep."
            )
        results[label] = {
            "filtered": filtered,
            "result": result,
            "table": _floor_candidate_table(
                members, filtered, result, params.temporal.event_count_floor, calendar
            ),
        }
    return results


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def _cross_tab(
    rows: list[tuple[str, datetime, PlayContext | None, str]], min_valid_ms: int
) -> dict[str, int]:
    tab: Counter[str] = Counter()
    for _cid, _local, context, _source in rows:
        if context is None:
            tab["kept"] += 1
            continue
        short = context.ms_played is not None and context.ms_played < min_valid_ms
        skip = context.skipped is True
        if short and skip:
            tab["dropped_both"] += 1
        elif short:
            tab["dropped_short_only"] += 1
        elif skip:
            tab["dropped_skip_only"] += 1
        else:
            tab["kept"] += 1
    return dict(tab)


def _per_source_counts(
    rows: list[tuple[str, datetime, PlayContext | None, str]], min_valid_ms: int
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for _cid, _local, context, source in rows:
        bucket = out.setdefault(source, {"total": 0, "valid": 0})
        bucket["total"] += 1
        if _is_valid_at(context, min_valid_ms=min_valid_ms):
            bucket["valid"] += 1
    return out


def _split_half(events: Sequence[ListenEvent]) -> tuple[list[ListenEvent], list[ListenEvent]]:
    ordered = sorted(events, key=lambda e: e.played_at)
    mid = len(ordered) // 2
    return ordered[:mid], ordered[mid:]


def build_report(
    events: Sequence[ListenEvent],
    members: Mapping[str, list[str]],
    rows: list[tuple[str, datetime, PlayContext | None, str]],
    unfiltered: Mapping[str, list[datetime]],
    results: dict[str, dict[str, Any]],
    resolver: IdentityResolver,
    resolved_cache: dict[str, TrackRef],
    ctx: DatasetContext,
    params: MethodParams,
) -> dict[str, Any]:
    raw_hour_hist = _hour_histogram([local for _cid, local, _ctx, _src in rows])

    half1_events, half2_events = _split_half(events)
    half1_rows = _localized_rows(half1_events, resolver, resolved_cache)
    half2_rows = _localized_rows(half2_events, resolver, resolved_cache)
    half1_unfiltered = _unfiltered_map(half1_rows)
    half2_unfiltered = _unfiltered_map(half2_rows)

    floors: dict[str, Any] = {}
    for label, min_valid_ms, ignore_skip in FLOOR_CONFIGS:
        payload = results[label]
        filtered = payload["filtered"]
        all_times = [t for times in filtered.values() for t in times]
        hour_hist = _hour_histogram(all_times)

        f1 = _filtered_map(half1_rows, min_valid_ms=min_valid_ms, ignore_skip=ignore_skip)
        r1 = derive_temporal_roots(
            members,
            f1,
            params=params.temporal,
            epoch_params=params.epochs,
            validation_params=params.validation,
            dataset_ctx=ctx,
            epoch_plays=half1_unfiltered,
        )
        t1 = _floor_candidate_table(
            members, f1, r1, params.temporal.event_count_floor, params.temporal.temporal_calendar
        )

        f2 = _filtered_map(half2_rows, min_valid_ms=min_valid_ms, ignore_skip=ignore_skip)
        r2 = derive_temporal_roots(
            members,
            f2,
            params=params.temporal,
            epoch_params=params.epochs,
            validation_params=params.validation,
            dataset_ctx=ctx,
            epoch_plays=half2_unfiltered,
        )
        t2 = _floor_candidate_table(
            members, f2, r2, params.temporal.event_count_floor, params.temporal.temporal_calendar
        )

        agreement, n_candidates = _agreement_rate(t1, t2)

        floors[label] = {
            "min_valid_ms": min_valid_ms,
            "ignore_skip": ignore_skip,
            "n_valid_plays": len(all_times),
            "per_source": _per_source_counts(rows, min_valid_ms) if not ignore_skip else None,
            "validity_cross_tab": _cross_tab(rows, min_valid_ms),
            "day_part_hour_histogram": hour_hist,
            "evening_peak_survives": _evening_peak_survives(hour_hist),
            "history_midpoint": _history_midpoint_of(filtered),
            "candidate_table": {
                f"{rid}|{'|'.join(bucket)}": v for (rid, bucket), v in payload["table"].items()
            },
            "split_half_test_retest": {
                "agreement_rate": agreement,
                "n_candidates_compared": n_candidates,
            },
        }

    return {
        "generated_at": None,  # stamped by caller (Date.now unavailable in some contexts)
        "n_events_total": len(events),
        "n_surviving_upstream_roots": len(members),
        "raw_unfiltered_day_part_hour_histogram": raw_hour_hist,
        "note_event_count_floor_fixed_on_half_data": (
            "event_count_floor=30 is fixed while each split-half run sees ~half the events; "
            "this shrinks in-bucket counts uniformly across ALL floors in the split-half "
            "comparison and is not a floor-specific artifact (AC6)."
        ),
        "floors": floors,
    }


def _history_midpoint_of(track_plays: Mapping[str, list[datetime]]) -> str | None:
    from music_intel_mcp.audio import _history_midpoint

    mid = _history_midpoint(track_plays)
    return mid.isoformat() if mid else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=None, help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)"
    )
    parser.add_argument(
        "--user-id", default="sweep", help="only used for context; no profile is written"
    )
    parser.add_argument("--ab-index", default=None, help="AcousticBrainz features JSONL")
    parser.add_argument("--mb-index", default=None, help="MusicBrainz ISRC->MBID index TSV")
    parser.add_argument("--shared-store-path", default=None, help="local shared-cache JSONL")
    parser.add_argument(
        "--out",
        required=True,
        help="output report path (local file under the data dir — never commit this)",
    )
    args = parser.parse_args(argv)

    store = UserStore(root=args.data_dir)
    events = store.load_history()
    if not events:
        print("no history.jsonl events found — nothing to sweep")
        return 1

    shared_store = LocalSharedStore(args.shared_store_path)
    audio_source = AcousticBrainzDump(path=args.ab_index)
    mb_index = MusicBrainzIsrcIndex(path=args.mb_index)
    identity_cache = IdentityCache(root=args.data_dir)
    resolver = IdentityResolver(
        mb_index,
        cache=identity_cache,
        ab_covered=lambda mbid: audio_source.lookup(mbid) is not None,
    )

    now = datetime.now(UTC)
    members, plays, ctx, resolved_cache = _reconstruct_members(
        events, shared_store, audio_source, resolver, now
    )
    if not members:
        print("no surviving audio/scene roots to condition on — cannot sweep temporal floors")
        return 1

    rows = _localized_rows(events, resolver, resolved_cache)
    unfiltered = _unfiltered_map(rows)
    params = MethodParams()

    results = _run_floors(members, rows, unfiltered, ctx, params)
    report = build_report(
        events, members, rows, unfiltered, results, resolver, resolved_cache, ctx, params
    )
    report["generated_at"] = now.isoformat()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote sweep report to {out_path}")
    for label, payload in report["floors"].items():
        print(
            f"  {label}: n_valid_plays={payload['n_valid_plays']} "
            f"evening_peak_survives={payload['evening_peak_survives']} "
            f"split_half_agreement={payload['split_half_test_retest']['agreement_rate']:.3f} "
            f"(n={payload['split_half_test_retest']['n_candidates_compared']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
