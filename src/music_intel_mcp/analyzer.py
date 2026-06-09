"""The derivation engine's orchestration seam.

Loads events -> computes ``generated_from`` -> runs each derivation stage in the
forced order (audio -> scene -> temporal -> epochs) -> assembles a schema-valid
``RootProfile``. Stages are *opt-in by dependency*: the audio stage runs only
when a shared store **and** an audio feature source are supplied; the scene
stage when a shared store **and** a tag source are supplied. The temporal stage
(#65) runs whenever either upstream stage produced at least one root/tendency to
condition on — it qualifies those roots by calendar bucket and detects epochs;
with no upstream members its ordering guard skips it. With no enrichment wired,
every derived section is empty — a *correct* output under the honest-empty
invariant, not a failure.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from .audio import AudioFeatureSource, derive_audio_roots, enrich_audio_features
from .models import (
    Epoch,
    GeneratedFrom,
    ListenEvent,
    MethodParams,
    RootProfile,
    TrackRef,
)
from .scene import TagSource, derive_scene_roots, enrich_tags
from .shared_store import SharedStore, TrackMetadataRecord, canonical_track_id
from .temporal import derive_temporal_roots
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


def _group_and_seed(
    events: Sequence[ListenEvent],
    store: SharedStore,
    now: datetime,
) -> tuple[dict[str, list[datetime]], list[str]]:
    """Group plays by canonical id and seed any unknown tracks into the shared
    store, returning the play map and the unique id list both stages share.

    Seeds carry only anonymous track facts (ids + name/artist) — never per-user
    fields — so writing them to the shared store is safe. Naive timestamps are
    coerced to UTC. Done once so audio and scene reuse the same seeded store."""
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
    return plays, unique_ids


def _dataset_ctx(generated_from: GeneratedFrom) -> DatasetContext:
    return DatasetContext(
        n_unique_tracks=generated_from.n_unique_tracks,
        history_span_days=generated_from.history_span_days,
    )


def analyze(
    events: Sequence[ListenEvent],
    *,
    user_id: str,
    generated_at: datetime | None = None,
    method_params: MethodParams | None = None,
    shared_store: SharedStore | None = None,
    audio_source: AudioFeatureSource | None = None,
    tag_source: TagSource | None = None,
) -> RootProfile:
    """Run the derivation pipeline over ``events`` and assemble a ``RootProfile``.

    The audio stage (#63) runs when a ``shared_store`` and an ``audio_source``
    are supplied; the scene stage (#64) when a ``shared_store`` and a
    ``tag_source`` are supplied. Each is independent — either, both, or neither
    may run; unrun categories stay honest-empty (never fabricated). The temporal
    stage (#65) then conditions the surviving roots/tendencies on calendar
    buckets and detects epochs. ``generated_from`` counts are always correct.
    """
    generated_at = generated_at or datetime.now(UTC)
    generated_from = _generated_from(events)
    # Deep copy so recording chosen params (e.g. scene.K_selected) never mutates
    # the caller's MethodParams.
    params = (method_params or MethodParams()).model_copy(deep=True)

    outcome = ValidationOutcome()
    epochs: list[Epoch] = []
    # candidate id (r-audio-N / r-scene-N) -> member track ids, for the temporal
    # stage to condition lift on root membership.
    members: dict[str, list[str]] = {}
    if events and shared_store is not None and (audio_source is not None or tag_source is not None):
        plays, unique_ids = _group_and_seed(events, shared_store, generated_at)
        ctx = _dataset_ctx(generated_from)

        if audio_source is not None:
            enrich_audio_features(unique_ids, shared_store, audio_source, now=generated_at)
            records = shared_store.get_tracks(unique_ids)
            audio = derive_audio_roots(
                records,
                plays,
                params=params.audio,
                validation_params=params.validation,
                dataset_ctx=ctx,
            )
            _merge(outcome, audio.outcome)
            members.update(audio.members)
            generated_from.coverage_per_category["audio"] = audio.coverage

        if tag_source is not None:
            enrich_tags(unique_ids, shared_store, tag_source, now=generated_at)
            records = shared_store.get_tracks(unique_ids)
            scene = derive_scene_roots(
                records,
                plays,
                params=params.scene,
                validation_params=params.validation,
                dataset_ctx=ctx,
            )
            _merge(outcome, scene.outcome)
            members.update(scene.members)
            generated_from.coverage_per_category["scene"] = scene.coverage
            params.scene.K_selected = scene.k_selected

        # Temporal qualifies only roots/tendencies that actually survived
        # validation — artifact_suspects are not real patterns to condition on.
        surviving = {item.id for item in (*outcome.roots, *outcome.tendencies)}
        members = {rid: ids for rid, ids in members.items() if rid in surviving}
        temporal = derive_temporal_roots(
            members,
            plays,
            params=params.temporal,
            epoch_params=params.epochs,
            validation_params=params.validation,
            dataset_ctx=ctx,
        )
        if not temporal.skipped:
            _merge(outcome, temporal.outcome)
            epochs = temporal.epochs
            for item in (*outcome.roots, *outcome.tendencies):
                if item.id in temporal.epoch_presence:
                    item.epoch_presence = temporal.epoch_presence[item.id]

    return RootProfile(
        snapshot_id=f"{user_id}/{generated_at.isoformat()}",
        user_id=user_id,
        generated_from=generated_from,
        method_params=params,
        roots=outcome.roots,
        tendencies=outcome.tendencies,
        epochs=epochs,
        quality_log=outcome.quality_log,
    )


def _merge(into: ValidationOutcome, other: ValidationOutcome) -> None:
    """Accumulate one stage's outcome into the running profile outcome."""
    into.roots.extend(other.roots)
    into.tendencies.extend(other.tendencies)
    into.quality_log.extend(other.quality_log)
