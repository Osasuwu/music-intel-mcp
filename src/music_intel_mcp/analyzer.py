"""The derivation engine's orchestration seam.

V0 skeleton: load events -> compute ``generated_from`` -> emit a schema-valid
honest-empty ``RootProfile``. Later slices plug their stages in here in the
forced order (audio -> scene -> temporal -> epochs); for now every derived
section is empty, which is a *correct* output under the honest-empty invariant
when no enrichment has run.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from .models import (
    GeneratedFrom,
    ListenEvent,
    MethodParams,
    RootProfile,
)
from .shared_store import canonical_track_id


def _generated_from(events: Sequence[ListenEvent]) -> GeneratedFrom:
    if not events:
        return GeneratedFrom(
            history_range=None,
            n_events=0,
            n_unique_tracks=0,
            history_span_days=0,
            data_sources=[],
            # temporal coverage is 1.0 even on empty input: every event that
            # *would* exist carries a timestamp. audio/scene need enrichers.
            coverage_per_category={"audio": 0.0, "scene": 0.0, "temporal": 1.0},
        )

    timestamps = [e.played_at for e in events]
    earliest, latest = min(timestamps), max(timestamps)
    unique = {canonical_track_id(e.track) for e in events}
    sources = sorted({e.source for e in events})

    return GeneratedFrom(
        history_range=(earliest, latest),
        n_events=len(events),
        n_unique_tracks=len(unique),
        history_span_days=(latest - earliest).days,
        data_sources=sources,
        coverage_per_category={"audio": 0.0, "scene": 0.0, "temporal": 1.0},
    )


def analyze(
    events: Sequence[ListenEvent],
    *,
    user_id: str,
    generated_at: datetime | None = None,
    method_params: MethodParams | None = None,
) -> RootProfile:
    """Run the (currently empty) derivation pipeline over ``events``.

    Returns a schema-valid honest-empty ``RootProfile``: correct
    ``generated_from`` counts, default ``method_params``, empty derived
    sections. Never fabricates roots.
    """
    generated_at = generated_at or datetime.now(UTC)
    return RootProfile(
        snapshot_id=f"{user_id}/{generated_at.isoformat()}",
        user_id=user_id,
        generated_from=_generated_from(events),
        method_params=method_params or MethodParams(),
        roots=[],
        tendencies=[],
        epochs=[],
        quality_log=[],
    )
