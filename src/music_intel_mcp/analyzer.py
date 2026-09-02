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

import logging
import os
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from .account_history import SOURCE as ACCOUNT_HISTORY_SOURCE
from .audio import AudioFeatureSource, derive_audio_roots, enrich_audio_features
from .automated_playback import AUTOMATED_PLAYBACK_SOURCE
from .identity import IdentityResolver
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
from .spotify_extended import SOURCE as SPOTIFY_EXTENDED_SOURCE
from .temporal import _is_valid, derive_temporal_roots
from .timezones import zone_for
from .validation import DatasetContext, ValidationOutcome

logger = logging.getLogger(__name__)

# Sources whose ``played_at`` is honestly UTC-stamped and therefore needs
# shifting to local wall-clock before bucketizing. Any source not in this set
# is assumed already-local wall-clock (e.g. ifttt, #76) and is only stripped
# of tzinfo, never shifted.
_UTC_STAMPED_SOURCES = frozenset({SPOTIFY_EXTENDED_SOURCE, ACCOUNT_HISTORY_SOURCE})

# Sources that are coverage plays, not genuine listening signal (#129, decision
# ``1c0f37f0``): the backfill playlist (#127) has no source of its own -- per
# its own module docstring, a backfill-queue track is only ever actually
# *played* via automated playback (#128) -- so this single origin tag already
# covers both origins named in the issue, with no separate tag/schema change
# (AC4: driven by the existing tag, not a heuristic).
_ANALYTICS_EXCLUDED_SOURCES = frozenset({AUTOMATED_PLAYBACK_SOURCE})
_INCLUDE_AGENT_PLAYS_ENV = "MUSIC_INTEL_INCLUDE_AGENT_PLAYS_IN_ANALYTICS"
_TRUTHY = {"1", "true", "yes", "on"}


def include_agent_originated_plays_by_default(env: dict[str, str] | None = None) -> bool:
    """AC2's opt-in override: off by default, so backfill/automated-origin
    plays stay excluded from taste analytics unless the user explicitly opts
    in via ``MUSIC_INTEL_INCLUDE_AGENT_PLAYS_IN_ANALYTICS`` (mirrors
    :func:`~music_intel_mcp.backfill_playlist.is_backfill_enabled`)."""
    source = env if env is not None else os.environ
    return source.get(_INCLUDE_AGENT_PLAYS_ENV, "").strip().lower() in _TRUTHY


def _analytics_events(
    events: Sequence[ListenEvent], *, include_agent_originated: bool
) -> list[ListenEvent]:
    """AC1's exclusion half: drop agent-originated (backfill/automated) plays
    from taste analytics unless ``include_agent_originated`` opts them back in.

    This runs independently of, and before, the play-validity filter
    (:func:`~music_intel_mcp.temporal._is_valid`, decision ``c69c6f17``) --
    the two are orthogonal predicates over different fields (origin vs.
    duration/skip), neither shortcircuits the other (AC3)."""
    if include_agent_originated:
        return list(events)
    return [e for e in events if e.source not in _ANALYTICS_EXCLUDED_SOURCES]


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


def _resolve_ref(
    track: TrackRef,
    resolver: IdentityResolver | None,
    cache: dict[str, TrackRef],
) -> TrackRef:
    """Walk ``track`` up the identity waterfall, memoised per raw canonical id so
    duplicate plays never re-walk the dump. Returns the raw ref unchanged when no
    resolver is wired (the honest-empty V0 path). The ``cache`` is shared between
    :func:`_group_and_seed` and :func:`_temporal_plays` so the two seed maps agree
    on the resolved key-space and the resolver runs once per distinct raw id."""
    if resolver is None:
        return track
    raw_cid = canonical_track_id(track)
    ref = cache.get(raw_cid)
    if ref is None:
        ref = resolver.resolve(track).to_track_ref()
        cache[raw_cid] = ref
    return ref


def _group_and_seed(
    events: Sequence[ListenEvent],
    store: SharedStore,
    now: datetime,
    resolver: IdentityResolver | None = None,
    resolved_cache: dict[str, TrackRef] | None = None,
) -> tuple[dict[str, list[datetime]], list[str]]:
    """Group plays by canonical id and seed any unknown tracks into the shared
    store, returning the play map and the unique id list both stages share.

    When a ``resolver`` is supplied (the P9 bridge, #87), each event's track is
    walked up the identity waterfall and grouped by its *resolved* canonical id
    — so a ``spotify:``/``isrc:`` event that reaches an MBID is rebuilt onto its
    ``mbid:`` key and its plays merge with any other event resolving to the same
    recording. This is the only path by which the enricher — which joins on the
    record's MBID — sees anything on the real history (whose events carry
    spotify_ids, not MBIDs). Resolution is memoised per raw id so duplicate
    plays don't re-walk the dump. Without a resolver the raw ingest identity is
    used unchanged (the honest-empty V0 path).

    Seeds carry only anonymous track facts (ids + name/artist) — never per-user
    fields — so writing them to the shared store is safe. Naive timestamps are
    coerced to UTC. Done once so audio and scene reuse the same seeded store."""
    if resolved_cache is None:
        resolved_cache = {}
    plays: dict[str, list[datetime]] = {}
    refs: dict[str, TrackRef] = {}
    for event in events:
        track = _resolve_ref(event.track, resolver, resolved_cache)
        cid = canonical_track_id(track)
        played = event.played_at
        if played.tzinfo is None:
            played = played.replace(tzinfo=UTC)
        plays.setdefault(cid, []).append(played)
        refs.setdefault(cid, track)

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


def _local_naive(played: datetime, zone: str) -> datetime:
    """Shift a (UTC-coerced) timestamp into ``zone`` and drop the tzinfo, yielding
    a naive *local wall-clock* datetime. zoneinfo picks the offset in force on the
    play's own date, so a DST/historical transition (e.g. KZ 2024-03-01 UTC+6→+5)
    lands the play in the hour the listener actually experienced."""
    if played.tzinfo is None:
        played = played.replace(tzinfo=UTC)
    return played.astimezone(ZoneInfo(zone)).replace(tzinfo=None)


def _modal_zone(events: Sequence[ListenEvent]) -> str | None:
    """The user's most common *mapped* zone across their own spotify_extended
    plays — the fallback for a play whose conn_country is null/unmapped. Computed
    per user from their own data; never a hardcoded owner default. ``None`` when
    the user has zero mapped rows (then the caller leaves such plays in raw UTC).
    Ties break by zone name so the choice is deterministic across runs."""
    counts: Counter[str] = Counter()
    for event in events:
        if event.source != SPOTIFY_EXTENDED_SOURCE:
            continue
        ctx = event.context
        zone = zone_for(ctx.conn_country) if ctx else None
        if zone is not None:
            counts[zone] += 1
    if not counts:
        return None
    return min(counts, key=lambda z: (-counts[z], z))


def _localize(event: ListenEvent, modal_zone: str | None) -> tuple[datetime, bool]:
    """Map one event to its naive local wall-clock datetime and a ``shifted`` flag.

    - UTC-stamped sources (``_UTC_STAMPED_SOURCES``: ``spotify_extended``,
      ``spotify_account_history``): shift by ``conn_country``'s zone, falling
      back to the user's ``modal_zone``; ``shifted=True``. With no zone at all
      (unmapped country AND no modal) leave it in raw UTC, ``shifted=False``.
    - Any other source (``ifttt`` etc.): the timestamp is already local wall-clock
      (#76), so only strip tzinfo — never shift; ``shifted=False``.
    """
    played = event.played_at
    if event.source not in _UTC_STAMPED_SOURCES:
        return (played.replace(tzinfo=None) if played.tzinfo else played), False
    ctx = event.context
    zone = zone_for(ctx.conn_country) if ctx else None
    if zone is None:
        zone = modal_zone
    if zone is None:
        return (played.replace(tzinfo=None) if played.tzinfo else played), False
    return _local_naive(played, zone), True


def _temporal_plays(
    events: Sequence[ListenEvent],
    resolver: IdentityResolver | None,
    resolved_cache: dict[str, TrackRef],
) -> tuple[dict[str, list[datetime]], dict[str, list[datetime]]]:
    """Build the temporal-seed play maps straight off the events (#90, AC2/AC4).

    Reads ``played_at``/``context``/``source`` from each :class:`ListenEvent` —
    NOT the audio/scene ``plays`` map, which is already UTC-coerced (double-shift
    risk). Localizes each play to naive local wall-clock (:func:`_localize`) and
    groups by *resolved* canonical id (shared ``resolved_cache``). Returns two
    maps: a validity-**filtered** one (feeds lift/buckets/_balance) and an
    **unfiltered** one (feeds epoch detection only), so the filter never moves an
    epoch boundary (decision ``77111ce0``).

    AC4: logs the count of *unshifted* plays split by source — the IFTTT
    traveler-smear residual that local-tz bucketize cannot correct.
    """
    modal_zone = _modal_zone(events)
    filtered: dict[str, list[datetime]] = {}
    unfiltered: dict[str, list[datetime]] = {}
    unshifted: Counter[str] = Counter()
    for event in events:
        cid = canonical_track_id(_resolve_ref(event.track, resolver, resolved_cache))
        local, shifted = _localize(event, modal_zone)
        if not shifted:
            unshifted[event.source] += 1
        unfiltered.setdefault(cid, []).append(local)
        if _is_valid(event.context):
            filtered.setdefault(cid, []).append(local)
    if unshifted:
        logger.info(
            "temporal: %d plays left unshifted (local wall-clock as-is) by source: %s",
            sum(unshifted.values()),
            dict(unshifted),
        )
    return filtered, unfiltered


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
    resolver: IdentityResolver | None = None,
    include_agent_originated_plays: bool | None = None,
) -> RootProfile:
    """Run the derivation pipeline over ``events`` and assemble a ``RootProfile``.

    The audio stage (#63) runs when a ``shared_store`` and an ``audio_source``
    are supplied; the scene stage (#64) when a ``shared_store`` and a
    ``tag_source`` are supplied. Each is independent — either, both, or neither
    may run; unrun categories stay honest-empty (never fabricated). The temporal
    stage (#65) then conditions the surviving roots/tendencies on calendar
    buckets and detects epochs. ``generated_from`` counts are always correct.

    When a ``resolver`` is supplied (the P9 bridge, #87), the seed/group step
    walks each event up the spotify_id -> ISRC -> MBID waterfall first, so the
    enrichers see the resolved MBID rather than the raw ingest-time identity.
    Without it, spotify-only history resolves to no MBID and audio stays
    honest-empty regardless of the dump.

    ``include_agent_originated_plays`` (#129): backfill/automated-origin plays
    (``AUTOMATED_PLAYBACK_SOURCE``) are excluded from every downstream stage by
    default -- they are coverage plays, not genuine listening signal. ``None``
    (the default) resolves via :func:`include_agent_originated_plays_by_default`'s
    env-var opt-in; pass ``True``/``False`` explicitly to override.
    """
    generated_at = generated_at or datetime.now(UTC)
    include_agent = (
        include_agent_originated_plays
        if include_agent_originated_plays is not None
        else include_agent_originated_plays_by_default()
    )
    events = _analytics_events(events, include_agent_originated=include_agent)
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
        resolved_cache: dict[str, TrackRef] = {}
        plays, unique_ids = _group_and_seed(
            events, shared_store, generated_at, resolver, resolved_cache
        )
        ctx = _dataset_ctx(generated_from)

        if audio_source is not None:
            report = enrich_audio_features(unique_ids, shared_store, audio_source, now=generated_at)
            generated_from.enrichment_diagnostics["audio"] = report.counts()
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
        # Temporal seeds off the events directly, localized to the listener's wall
        # clock (#90): the filtered map drives lift/buckets, the unfiltered map
        # drives epoch detection only. Shares the resolve cache with the seed step.
        temporal_filtered, temporal_unfiltered = _temporal_plays(events, resolver, resolved_cache)
        temporal = derive_temporal_roots(
            members,
            temporal_filtered,
            params=params.temporal,
            epoch_params=params.epochs,
            validation_params=params.validation,
            dataset_ctx=ctx,
            epoch_plays=temporal_unfiltered,
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
