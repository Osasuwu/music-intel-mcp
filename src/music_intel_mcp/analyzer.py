"""The derivation engine's orchestration seam.

Loads events -> computes ``generated_from`` -> runs each derivation stage in the
forced order (audio -> scene -> temporal -> epochs) -> assembles a schema-valid
``RootProfile``. Stages are *opt-in by dependency*: the audio stage runs only
when a shared store **and** an audio feature source are supplied. With no
enrichment wired, every derived section is empty — a *correct* output under the
honest-empty invariant, not a failure. Scene/temporal stages land in #64/#65.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from .audio import AudioFeatureSource, derive_audio_roots, enrich_audio_features
from .models import (
    GeneratedFrom,
    ListenEvent,
    MethodParams,
    RootProfile,
    TrackRef,
)
from .shared_store import SharedStore, TrackMetadataRecord, canonical_track_id
from .validation import DatasetContext, ValidationOutcome


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


def _derive_audio(
    events: Sequence[ListenEvent],
    generated_from: GeneratedFrom,
    method_params: MethodParams,
    *,
    store: SharedStore,
    source: AudioFeatureSource,
    now: datetime,
):
    """Audio stage: group plays by canonical id, seed any unknown tracks into the
    shared store, enrich by MBID, then cluster the enriched population.

    Seeds carry only anonymous track facts (ids + name/artist) — never per-user
    fields — so writing them to the shared store is safe. Reads/writes are bulk
    only (the no-round-trip discipline)."""
    plays: dict[str, list[datetime]] = {}
    refs: dict[str, TrackRef] = {}
    for event in events:
        cid = canonical_track_id(event.track)
        played = event.played_at
        if played.tzinfo is None:
            played = played.replace(tzinfo=UTC)
        plays.setdefault(cid, []).append(played)
        refs.setdefault(cid, event.track)

    unique_ids = list(refs)
    existing = store.get_tracks(unique_ids)
    seeds = [
        TrackMetadataRecord(
            track_id=cid,
            spotify_id=ref.spotify_id,
            isrc=ref.isrc,
            mbid=ref.mbid,
            name=ref.name,
            artist=ref.artist,
            fetched_at=now,
        )
        for cid, ref in refs.items()
        if cid not in existing
    ]
    if seeds:
        store.upsert_tracks(seeds)

    enrich_audio_features(unique_ids, store, source, now=now)
    records = store.get_tracks(unique_ids)

    return derive_audio_roots(
        records,
        plays,
        params=method_params.audio,
        validation_params=method_params.validation,
        dataset_ctx=DatasetContext(
            n_unique_tracks=generated_from.n_unique_tracks,
            history_span_days=generated_from.history_span_days,
        ),
    )


def analyze(
    events: Sequence[ListenEvent],
    *,
    user_id: str,
    generated_at: datetime | None = None,
    method_params: MethodParams | None = None,
    shared_store: SharedStore | None = None,
    audio_source: AudioFeatureSource | None = None,
) -> RootProfile:
    """Run the derivation pipeline over ``events`` and assemble a ``RootProfile``.

    The audio stage (#63) runs when both ``shared_store`` and ``audio_source``
    are supplied; otherwise it is skipped and the audio sections stay empty
    (honest-empty, never fabricated). ``generated_from`` counts are always
    correct. Scene/temporal stages plug in here in #64/#65.
    """
    generated_at = generated_at or datetime.now(UTC)
    generated_from = _generated_from(events)
    params = method_params or MethodParams()

    outcome = ValidationOutcome()
    if events and shared_store is not None and audio_source is not None:
        derivation = _derive_audio(
            events,
            generated_from,
            params,
            store=shared_store,
            source=audio_source,
            now=generated_at,
        )
        outcome = derivation.outcome
        generated_from.coverage_per_category["audio"] = derivation.coverage

    return RootProfile(
        snapshot_id=f"{user_id}/{generated_at.isoformat()}",
        user_id=user_id,
        generated_from=generated_from,
        method_params=params,
        roots=outcome.roots,
        tendencies=outcome.tendencies,
        epochs=[],
        quality_log=outcome.quality_log,
    )
