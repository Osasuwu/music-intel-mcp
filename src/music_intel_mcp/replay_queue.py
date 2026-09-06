"""Replay queue selector (#163) — pilot slice 1.

``select_backfill_tracks`` (``backfill_playlist.py``) builds a queue from the
saved library *minus played tracks* — the inverse of this module's need. The
pilot's replay queue is the participant's own listening history filtered to
tracks with taste **signal**: a canonical key needs >=``min_valid_plays``
**valid** plays (``_is_valid``, the >=30s rule already used by the temporal
seed path, decision ``c69c6f17``) before it is worth spending an audio
analysis on, and a track already analyzed (pool or root, decision ``2e17aafa``)
is dropped since paying for a duplicate analysis wastes the pilot's bounded
processing budget.

Both dimensions the AC stratifies on are read straight off the listening
history: ``artist`` (``TrackRef.artist``) and a **release-year proxy** — the
year of the canonical key's *earliest* valid play. No release-year field
exists anywhere in the schema (``TrackRef``/``PlayContext``/``ListenEvent``/
``TrackMetadataRecord``), and fetching one is out of scope for a pure
selection-logic issue; per CONTEXT.md's "proxy outputs are labelled"
invariant this is an explicitly-documented proxy, not real metadata.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .models import ListenEvent, TrackRef
from .shared_store import canonical_track_id
from .temporal import _is_valid

MIN_VALID_PLAYS = 3
DEFAULT_REPLAY_QUEUE_CAP = 200


def valid_play_counts(events: Iterable[ListenEvent]) -> dict[str, int]:
    """Count of *valid* (``_is_valid``) plays per canonical track key."""
    counts: dict[str, int] = {}
    for event in events:
        if not _is_valid(event.context):
            continue
        cid = canonical_track_id(event.track)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


@dataclass(frozen=True)
class _Candidate:
    cid: str
    track: TrackRef
    valid_plays: int
    year: int


def _candidates(
    events: list[ListenEvent],
    *,
    min_valid_plays: int,
    has_audio_analysis: Callable[[str], bool],
) -> list[_Candidate]:
    counts: dict[str, int] = {}
    reps: dict[str, TrackRef] = {}
    earliest_year: dict[str, int] = {}
    for event in events:
        cid = canonical_track_id(event.track)
        reps.setdefault(cid, event.track)
        if not _is_valid(event.context):
            continue
        counts[cid] = counts.get(cid, 0) + 1
        year = event.played_at.year
        if cid not in earliest_year or year < earliest_year[cid]:
            earliest_year[cid] = year

    candidates: list[_Candidate] = []
    for cid, count in counts.items():
        if count < min_valid_plays:
            continue
        if has_audio_analysis(cid):
            continue
        candidates.append(
            _Candidate(cid=cid, track=reps[cid], valid_plays=count, year=earliest_year[cid])
        )
    return candidates


def _stratify_and_cap(candidates: list[_Candidate], *, cap: int) -> list[_Candidate]:
    """Round-robin across (artist, year-proxy) strata so no single artist or
    year can exhaust the cap before the others get a turn (AC3). Each stratum
    is visited once per round in a deterministic order (largest stratum
    first, tie-broken by artist/year); a stratum that runs dry simply drops
    out of the rotation, letting the remaining budget flow to the others."""
    buckets: dict[tuple[str, int], list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate.track.artist, candidate.year)].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda c: (-c.valid_plays, c.cid))
    bucket_keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k[0], k[1]))

    selected: list[_Candidate] = []
    while len(selected) < cap and any(buckets[key] for key in bucket_keys):
        for key in bucket_keys:
            if len(selected) >= cap:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
    return selected


def select_replay_queue(
    events: Iterable[ListenEvent],
    *,
    has_audio_analysis: Callable[[str], bool],
    min_valid_plays: int = MIN_VALID_PLAYS,
    cap: int = DEFAULT_REPLAY_QUEUE_CAP,
) -> list[TrackRef]:
    """The pilot replay queue: >=``min_valid_plays`` valid plays on the
    canonical key, minus tracks already analyzed (pool or root), stratified
    by artist/year-proxy and capped at ``cap`` (#163)."""
    candidates = _candidates(
        list(events), min_valid_plays=min_valid_plays, has_audio_analysis=has_audio_analysis
    )
    selected = _stratify_and_cap(candidates, cap=cap)
    return [candidate.track for candidate in selected]


@dataclass(frozen=True)
class ReplayQueueStats:
    """The AC5.1 coverage stat (decision ``87277764``): share of **valid
    plays** covered by already-analysed + newly-queued tracks, over the
    full history -- not share of unique/eligible tracks."""

    eligible_track_count: int
    already_analyzed_count: int
    queued_count: int
    valid_play_coverage: float


def replay_queue_coverage(
    events: Iterable[ListenEvent],
    *,
    has_audio_analysis: Callable[[str], bool],
    min_valid_plays: int = MIN_VALID_PLAYS,
    cap: int = DEFAULT_REPLAY_QUEUE_CAP,
) -> ReplayQueueStats:
    events = list(events)
    counts = valid_play_counts(events)
    total_valid_plays = sum(counts.values())

    eligible_cids = {cid for cid, count in counts.items() if count >= min_valid_plays}
    analyzed_cids = {cid for cid in eligible_cids if has_audio_analysis(cid)}
    queue = select_replay_queue(
        events, has_audio_analysis=has_audio_analysis, min_valid_plays=min_valid_plays, cap=cap
    )
    queued_cids = {canonical_track_id(track) for track in queue}

    covered_plays = sum(counts[cid] for cid in analyzed_cids | queued_cids)
    coverage = covered_plays / total_valid_plays if total_valid_plays else 0.0

    return ReplayQueueStats(
        eligible_track_count=len(eligible_cids),
        already_analyzed_count=len(analyzed_cids),
        queued_count=len(queued_cids),
        valid_play_coverage=coverage,
    )
