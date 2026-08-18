"""Fixture tests for the offline validity-floor sweep harness (#95, AC7).

Synthetic events only — no real history, no network, no live API. Covers the
four seams the harness adds on top of production code:

1. the floor-seam predicate matches production ``temporal._is_valid`` exactly
   at the production floor (and the ``ignore_skip`` variant only relaxes skip);
2. the validity cross-tab counts short/skip/both correctly;
3. a bucket below ``event_count_floor`` is attributed ``dropped:event_count_floor``,
   never ``absent`` (the silent-drop the harness must reconstruct itself);
4. the epoch-invariance guard trips when the epoch mask is perturbed between
   runs (decision ``77111ce0`` — filtering must never move an epoch boundary).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sweep_validity_floor import (
    EpochInvarianceViolation,
    _agreement_rate,
    _all_bucket_counts,
    _cross_tab,
    _epoch_signature,
    _filtered_map,
    _floor_candidate_table,
    _is_valid_at,
    _unfiltered_map,
)

from music_intel_mcp.models import EpochParams, PlayContext, TemporalParams, ValidationParams
from music_intel_mcp.temporal import _is_valid, derive_temporal_roots
from music_intel_mcp.validation import DatasetContext

# --------------------------------------------------------------------------- #
# 1. floor-seam equivalence with production _is_valid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "context",
    [
        PlayContext(ms_played=29_999),
        PlayContext(ms_played=30_000),
        PlayContext(ms_played=None),
        PlayContext(skipped=True),
        PlayContext(skipped=None),
        PlayContext(skipped=False),
        None,
    ],
)
def test_is_valid_at_matches_production_at_default_floor(context):
    assert _is_valid_at(context, min_valid_ms=30_000) == _is_valid(context, min_valid_ms=30_000)


def test_is_valid_at_ignore_skip_only_relaxes_the_skip_predicate():
    # ms-floor test still applies under ignore_skip=True — only the skip test is bypassed.
    assert (
        _is_valid_at(PlayContext(ms_played=10_000), min_valid_ms=30_000, ignore_skip=True) is False
    )
    assert _is_valid_at(PlayContext(skipped=True), min_valid_ms=30_000, ignore_skip=True) is True
    assert (
        _is_valid_at(
            PlayContext(ms_played=10_000, skipped=True), min_valid_ms=30_000, ignore_skip=True
        )
        is False
    )


# --------------------------------------------------------------------------- #
# 2. validity cross-tab
# --------------------------------------------------------------------------- #


def _row(ms_played=None, skipped=None):
    context = (
        PlayContext(ms_played=ms_played, skipped=skipped)
        if (ms_played is not None or skipped is not None)
        else None
    )
    return ("cid", datetime(2025, 1, 1, tzinfo=UTC), context, "spotify_extended")


def test_cross_tab_separates_short_skip_and_both():
    rows = [
        _row(ms_played=50_000, skipped=False),  # kept
        _row(ms_played=10_000, skipped=False),  # short only
        _row(ms_played=50_000, skipped=True),  # skip only
        _row(ms_played=10_000, skipped=True),  # both
        _row(),  # no context -> kept
    ]
    tab = _cross_tab(rows, min_valid_ms=30_000)
    assert tab == {"kept": 2, "dropped_short_only": 1, "dropped_skip_only": 1, "dropped_both": 1}


def test_filtered_and_unfiltered_maps_agree_on_totals():
    rows = [
        _row(ms_played=50_000, skipped=False),
        _row(ms_played=10_000, skipped=False),
    ]
    filtered = _filtered_map(rows, min_valid_ms=30_000)
    unfiltered = _unfiltered_map(rows)
    assert sum(len(v) for v in filtered.values()) == 1
    assert sum(len(v) for v in unfiltered.values()) == 2


# --------------------------------------------------------------------------- #
# 3. count-floor attribution: dropped, not absent
# --------------------------------------------------------------------------- #

VP_OPEN = ValidationParams(
    N_THRESHOLD=2, T_THRESHOLD_DAYS=30, evidence_count_floor=1, confidence_floor=0.5
)
CTX = DatasetContext(n_unique_tracks=4, history_span_days=760)


def test_bucket_below_event_count_floor_is_attributed_dropped_not_absent():
    # Three plays in the same (weekday, winter, night) bucket for "root" — well
    # under a floor of 10, so derive_temporal_roots silently drops the bucket
    # (temporal.py's "silent precondition, not logged" continue) and the harness
    # must reconstruct the drop itself via _all_bucket_counts, not report "absent".
    times = [datetime(2025, 1, d, 2, tzinfo=UTC) for d in (6, 7, 8)]  # Mon-Wed
    members = {"root": ["track-a"]}
    filtered = {"track-a": times}
    params = TemporalParams(event_count_floor=10, lift_floor=1.3)
    result = derive_temporal_roots(
        members,
        filtered,
        params=params,
        epoch_params=EpochParams(),
        validation_params=VP_OPEN,
        dataset_ctx=CTX,
        epoch_plays=filtered,
    )
    table = _floor_candidate_table(
        members, filtered, result, params.event_count_floor, params.temporal_calendar
    )
    counts = _all_bucket_counts(members, filtered, params.temporal_calendar)
    assert counts  # the bucket exists...
    for key, count in counts.items():
        if count < params.event_count_floor:
            assert table[key]["verdict"] == "dropped:event_count_floor"
            assert table[key]["evidence_count"] == count


def test_agreement_rate_treats_missing_key_as_absent():
    key = ("root", ("night", "weekday", "winter"))
    table_a = {key: {"verdict": "root", "lift": 2.0, "evidence_count": 40}}
    table_b = {}
    rate, n = _agreement_rate(table_a, table_b)
    assert n == 1
    assert rate == 0.0  # "root" vs "absent" disagree

    rate2, n2 = _agreement_rate(table_a, table_a)
    assert n2 == 1
    assert rate2 == 1.0


# --------------------------------------------------------------------------- #
# 4. epoch-invariance guard
# --------------------------------------------------------------------------- #


def test_epoch_signature_stable_across_identical_epochs():
    from music_intel_mcp.models import Epoch

    e1 = Epoch(
        id="e1",
        range=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
        n_events=10,
    )
    e2 = Epoch(
        id="e2",
        range=(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 1, tzinfo=UTC)),
        n_events=10,
    )
    assert _epoch_signature([e1]) == _epoch_signature([e2])


def test_run_floors_aborts_when_epoch_mask_is_perturbed(monkeypatch):
    # Force derive_temporal_roots to return a perturbed epoch list on the second
    # floor config only, simulating filtering having shifted an epoch boundary
    # — the sweep must abort rather than silently accept the mismatch.
    import sweep_validity_floor as sweep

    members = {"root": ["track-a"]}
    times = [datetime(2025, 1, d, 2, tzinfo=UTC) for d in (6, 7, 8, 9, 10)]
    rows = [(t.isoformat(), t, None, "spotify_extended") for t in times]
    unfiltered = {"track-a": times}
    params_temporal = TemporalParams(event_count_floor=1, lift_floor=0.0)

    from music_intel_mcp.temporal import TemporalDerivation

    call_count = {"n": 0}
    real_derive = sweep.derive_temporal_roots

    def fake_derive(*args, **kwargs):
        result = real_derive(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 2:
            from music_intel_mcp.models import Epoch

            perturbed = Epoch(
                id="perturbed",
                range=(datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 2, 1, tzinfo=UTC)),
                n_events=999,
            )
            return TemporalDerivation(
                outcome=result.outcome,
                epochs=[perturbed],
                epoch_presence=result.epoch_presence,
            )
        return result

    monkeypatch.setattr(sweep, "derive_temporal_roots", fake_derive)
    monkeypatch.setattr(sweep, "FLOOR_CONFIGS", [("a", 0, False), ("b", 30_000, False)])

    from music_intel_mcp.models import MethodParams

    params = MethodParams(temporal=params_temporal, epochs=EpochParams(), validation=VP_OPEN)

    with pytest.raises(EpochInvarianceViolation):
        sweep._run_floors(members, rows, unfiltered, CTX, params)
