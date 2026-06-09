"""Scene root pipeline (#64) — the deep module behind the scene category.

One pipeline, two halves (mirrors :mod:`audio`, decision ``91229a71``):

1. **Enrichment.** :func:`enrich_tags` populates ``tags`` on shared-store
   records from a :class:`TagSource` (Last.fm in production, an in-memory map in
   tests). Tags are *universal* — every track carries a name+artist, so there is
   no ``no_mbid`` bucket the way audio has; a track is either already tagged,
   freshly tagged, or the source had nothing for it.

2. **Derivation.** :func:`derive_scene_roots` builds the user's track×tag matrix,
   fits **NMF** for each K in the configured grid, and auto-selects the K whose
   topics are, on mean, the most *coherent* (NPMI over each topic's top tags)
   above a floor. Each accepted topic becomes one scene root carrying its top
   tags, top artists, and coherence; the validation core (#62) then decides
   root / tendency / artifact_suspect. Topics below the coherence floor are
   rejected transparently to ``quality_log[]``. If **no** K clears the floor the
   stage is honest-empty — ``scene_roots = []``, ``K_selected = None`` — never a
   fabricated topic.

Three validation scalars, three distinct meanings (matches the canonical
example): **confidence** = mean topic dominance over a topic's docs (soft
membership strength); **coverage** = fraction of those docs that are *majority*
this topic (≥0.5 mass); **coherence_score** = the NPMI tag coherence that gates
the topic in the first place.

Determinism: docs are sorted by canonical id and the vocab is sorted before the
matrix is built; NMF uses ``init="nndsvd"`` (SVD-seeded, no randomness) with a
pinned ``random_state`` — same input ⇒ identical roots.

Tag canonicalization and Last.fm artist-similarity are V1+ (out of scope here).
The Last.fm source is network-only and never exercised in CI; tests use
synthetic tags exclusively.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.decomposition import NMF

from .audio import _history_midpoint, _stability
from .models import SceneParams, ValidationParams
from .shared_store import SharedStore, TrackMetadataRecord, TrackTag
from .validation import Candidate, DatasetContext, QualityLogEntry, ValidationOutcome, Validator

_LASTFM_API_KEY_ENV = "LASTFM_API_KEY"

# Stop tags (v1): library-management, sentiment, and structural noise that are
# never a *scene*. Kept deliberately small — genre/mood/era tags (incl.
# "female vocalists", "90s") stay in, since they are real scene signal. Tag
# canonicalization (merging synonyms) is a separate V1 concern.
_STOP_TAGS_V1 = frozenset(
    {
        "seen live",
        "favorite",
        "favorites",
        "favourite",
        "favourites",
        "favorite songs",
        "good",
        "awesome",
        "love",
        "loved",
        "beautiful",
        "cool",
        "nice",
        "best",
        "amazing",
        "great",
        "epic",
        "masterpiece",
        "various",
        "various artists",
        "misc",
        "miscellaneous",
        "music",
        "songs i like",
        "albums i own",
        "owned",
        "spotify",
        "itunes",
    }
)


def _stop_tags(version: str) -> frozenset[str]:
    """Resolve a ``stop_tags_version`` to its set. Unknown versions degrade to
    the empty set (forward-compatible: a future caller naming ``v2`` gets no
    silent v1 filtering)."""
    return _STOP_TAGS_V1 if version == "v1" else frozenset()


# --------------------------------------------------------------------------- #
# Enrichment — tag sources
# --------------------------------------------------------------------------- #


@runtime_checkable
class TagSource(Protocol):
    """Looks scene tags up for a track. Last.fm in production, a map in tests."""

    def lookup(self, artist: str, track: str, mbid: str | None) -> list[TrackTag] | None:
        """Return tags for the track or ``None`` when the source has none."""
        ...


class InMemoryTagSource:
    """Dict-backed :class:`TagSource` for tests, keyed by
    ``(artist.casefold(), track.casefold())``. ``lookups`` records each query so
    tests can assert no wasted lookups (already-tagged tracks must not reach
    the source)."""

    def __init__(self, mapping: Mapping[tuple[str, str], list[TrackTag]] | None = None) -> None:
        self._mapping = {
            (a.casefold(), t.casefold()): tags for (a, t), tags in (mapping or {}).items()
        }
        self.lookups: list[tuple[str, str]] = []

    def lookup(self, artist: str, track: str, mbid: str | None) -> list[TrackTag] | None:
        self.lookups.append((artist, track))
        return self._mapping.get((artist.casefold(), track.casefold()))


class LastfmTagSource:  # pragma: no cover - network-only, never run in CI
    """Production :class:`TagSource` over the Last.fm ``track.getTopTags`` API.

    Network-only: ``httpx`` is imported lazily and the API key is read from
    ``LASTFM_API_KEY``. Never exercised in CI — tests use
    :class:`InMemoryTagSource`. Tag ``weight`` is Last.fm's 0–100 ``count``.
    """

    _ENDPOINT = "https://ws.audioscrobbler.com/2.0/"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get(_LASTFM_API_KEY_ENV)

    def lookup(self, artist: str, track: str, mbid: str | None) -> list[TrackTag] | None:
        if not self._api_key:
            raise RuntimeError(f"{_LASTFM_API_KEY_ENV} must be set to use LastfmTagSource.")
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "LastfmTagSource needs httpx: pip install 'music-intel-mcp[lastfm]'"
            ) from exc
        params = {
            "method": "track.gettoptags",
            "artist": artist,
            "track": track,
            "api_key": self._api_key,
            "format": "json",
            "autocorrect": "1",
        }
        if mbid:
            params["mbid"] = mbid
        resp = httpx.get(self._ENDPOINT, params=params, timeout=15.0)
        resp.raise_for_status()
        raw = resp.json().get("toptags", {}).get("tag", [])
        tags = [
            TrackTag(tag=t["name"], weight=float(t.get("count", 0)), source="lastfm")
            for t in raw
            if t.get("name")
        ]
        return tags or None


# --------------------------------------------------------------------------- #
# Enrichment — orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class TagEnrichmentReport:
    """Outcome of an :func:`enrich_tags` run. Every considered track lands in
    exactly one bucket — transparent, nothing silently lost.

    - ``enriched`` — gained tags from the source this run (written back).
    - ``already_present`` — carried tags already; source not queried.
    - ``missing_tags`` — the source had nothing for the track.
    """

    enriched: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    missing_tags: list[str] = field(default_factory=list)

    @property
    def total_considered(self) -> int:
        return len(self.enriched) + len(self.already_present) + len(self.missing_tags)

    @property
    def coverage(self) -> float:
        """Fraction of considered tracks that carry tags afterwards."""
        n = self.total_considered
        if n == 0:
            return 0.0
        return (len(self.enriched) + len(self.already_present)) / n


def enrich_tags(
    track_ids: Sequence[str],
    store: SharedStore,
    source: TagSource,
    *,
    now: datetime,
) -> TagEnrichmentReport:
    """Populate ``tags`` on the store's records for ``track_ids``.

    One bulk read, one bulk write-back (the shared-store round-trip discipline).
    Tracks already carrying tags are skipped (no source lookup). Records absent
    from the store are not considered (the caller seeds them first)."""
    records = store.get_tracks(list(dict.fromkeys(track_ids)))
    report = TagEnrichmentReport()
    to_write: list[TrackMetadataRecord] = []

    for tid, record in records.items():
        if record.tags:
            report.already_present.append(tid)
            continue
        tags = source.lookup(record.artist, record.name, record.mbid)
        if not tags:
            report.missing_tags.append(tid)
            continue
        record.tags = tags
        to_write.append(record)
        report.enriched.append(tid)

    if to_write:
        store.upsert_tracks(to_write)
    return report


# --------------------------------------------------------------------------- #
# Derivation — NMF topics → coherence gate → structural descriptor → validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SceneDerivation:
    """Result of :func:`derive_scene_roots`.

    - ``outcome`` — the validated roots / tendencies / quality_log (the
      quality_log already includes any coherence-floor rejections).
    - ``coverage`` — fraction of the user's records carrying tags (the scene
      entry of ``generated_from.coverage_per_category``).
    - ``n_tagged`` — docs that entered NMF (had ≥1 non-stop tag).
    - ``k_selected`` — the chosen K, or ``None`` on honest-empty.
    - ``k_coherences`` — mean coherence per evaluated K (transparency).
    """

    outcome: ValidationOutcome
    coverage: float
    n_tagged: int
    k_selected: int | None
    k_coherences: dict[int, float]


def derive_scene_roots(
    records: Mapping[str, TrackMetadataRecord],
    track_plays: Mapping[str, list[datetime]],
    *,
    params: SceneParams,
    validation_params: ValidationParams,
    dataset_ctx: DatasetContext,
) -> SceneDerivation:
    """Factor the user's track×tag matrix into scene topics, then validate.

    ``records`` is the user's track population keyed by canonical id;
    ``track_plays`` maps the same ids to play timestamps (drives the temporal
    split). Only records carrying ≥1 non-stop tag enter the matrix. NMF is fit
    for each K in ``params.K_grid_explored`` that fits the data; the K with the
    highest mean topic coherence above ``params.coherence_floor`` is selected.
    No K clears the floor ⇒ honest-empty.
    """
    stop = _stop_tags(params.stop_tags_version)
    floor = params.coherence_floor
    total = len(records)

    docs = sorted(
        ((tid, rec, dt) for tid, rec in records.items() if (dt := _doc_tags(rec, stop))),
        key=lambda d: d[0],
    )
    n_tagged = len(docs)
    coverage = n_tagged / total if total else 0.0

    vocab = sorted({tag for _, _, dt in docs for tag in dt})
    n_docs, n_tags = n_tagged, len(vocab)

    valid_ks = [k for k in sorted(set(params.K_grid_explored)) if 1 <= k <= min(n_docs, n_tags)]
    if not valid_ks:
        return SceneDerivation(ValidationOutcome(), coverage, n_tagged, None, {})

    tag_index = {t: j for j, t in enumerate(vocab)}
    x = np.zeros((n_docs, n_tags), dtype=float)
    b = np.zeros((n_docs, n_tags), dtype=float)
    for i, (_, _, dt) in enumerate(docs):
        for tag, weight in dt.items():
            j = tag_index[tag]
            x[i, j] = float(weight) if weight is not None and weight > 0 else 1.0
            b[i, j] = 1.0

    cooccur = b.T @ b  # n_tags × n_tags: docs carrying both a and b
    doc_freq = b.sum(axis=0)  # n_tags: docs carrying each tag

    # Fit each candidate K and score it by mean topic coherence.
    fits: dict[int, _Fit] = {}
    k_coherences: dict[int, float] = {}
    for k in valid_ks:
        model = NMF(n_components=k, init="nndsvd", random_state=0, max_iter=1000)
        w = model.fit_transform(x)
        h = model.components_
        topics = [
            _topic(t, h[t], vocab, cooccur, doc_freq, n_docs, params.coherence_top_n)
            for t in range(k)
        ]
        mean_coh = sum(tp.coherence for tp in topics) / k
        fits[k] = _Fit(w=w, topics=topics, mean_coherence=mean_coh)
        k_coherences[k] = round(mean_coh, 4)

    passing = [k for k in valid_ks if fits[k].mean_coherence >= floor]
    if not passing:
        return SceneDerivation(ValidationOutcome(), coverage, n_tagged, None, k_coherences)

    # Highest mean coherence wins; ties resolve to the smaller (simpler) K.
    k_selected = max(passing, key=lambda k: (fits[k].mean_coherence, -k))
    fit = fits[k_selected]

    midpoint = _history_midpoint(track_plays)
    rejections: list[QualityLogEntry] = []
    built: list[dict] = []
    for topic in fit.topics:
        if topic.coherence < floor:
            rejections.append(
                QualityLogEntry(
                    candidate_id=f"q-scene-{topic.index}",
                    category="scene",
                    failed_test="coherence_floor",
                    details={
                        "coherence": round(topic.coherence, 4),
                        "floor": floor,
                        "K": k_selected,
                        "top_tags": [vocab[j] for j in topic.tag_idx],
                    },
                )
            )
            continue
        built.append(
            _build_scene_cluster(topic, fit.w, docs, vocab, track_plays, midpoint, n_docs, params)
        )

    # Rank by member count (desc), tie-broken by smallest member id, so r-scene-1
    # is the most prominent topic and ids are deterministic.
    built.sort(key=lambda c: (-c["cluster_size"], c["min_id"]))
    candidates = [
        Candidate(
            candidate_id=f"r-scene-{rank}",
            category="scene",
            cluster_size=c["cluster_size"],
            cluster_share=c["cluster_share"],
            evidence_count=c["cluster_size"],
            coverage=c["coverage"],
            confidence=c["confidence"],
            structural_descriptor=c["descriptor"],
            temporal_stability_score=c["stability"],
            sample_tracks=c["samples"],
            actionability_hint=c["hint"],
        )
        for rank, c in enumerate(built, start=1)
    ]

    outcome = Validator(validation_params).validate(candidates, dataset_ctx)
    # Coherence rejections precede the validator's own rejections in the log.
    outcome.quality_log = rejections + outcome.quality_log
    return SceneDerivation(outcome, coverage, n_tagged, k_selected, k_coherences)


# --------------------------------------------------------------------------- #
# Derivation helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Topic:
    """One NMF component, scored. ``tag_idx`` are the indices into ``vocab`` of
    the topic's meaningful top tags (those with non-zero rounded weight)."""

    index: int
    tag_idx: list[int]
    tag_weights: dict[int, float]  # vocab index → normalised topic weight
    coherence: float


@dataclass(frozen=True)
class _Fit:
    w: np.ndarray
    topics: list[_Topic]
    mean_coherence: float


def _doc_tags(record: TrackMetadataRecord, stop: frozenset[str]) -> dict[str, float | None]:
    """A record's non-stop tags as ``{tag: weight}`` (casefolded keys, last
    weight wins on duplicates). Empty ⇒ the doc carries no scene signal."""
    out: dict[str, float | None] = {}
    for t in record.tags:
        key = t.tag.casefold()
        if key in stop:
            continue
        out[key] = t.weight
    return out


def _topic(
    index: int,
    h_row: np.ndarray,
    vocab: list[str],
    cooccur: np.ndarray,
    doc_freq: np.ndarray,
    n_docs: int,
    top_n: int,
) -> _Topic:
    """Score one NMF component: its top-``top_n`` tags by weight (dropping any
    that round to zero), normalised weights, and mean pairwise NPMI coherence."""
    hsum = float(h_row.sum())
    if hsum <= 0.0:
        return _Topic(index=index, tag_idx=[], tag_weights={}, coherence=0.0)
    order = sorted(range(len(vocab)), key=lambda j: (-h_row[j], vocab[j]))[:top_n]
    tag_idx = [j for j in order if round(float(h_row[j]) / hsum, 4) > 0.0]
    tag_weights = {j: round(float(h_row[j]) / hsum, 4) for j in tag_idx}
    coherence = _coherence(tag_idx, cooccur, doc_freq, n_docs)
    return _Topic(index=index, tag_idx=tag_idx, tag_weights=tag_weights, coherence=coherence)


def _coherence(
    tag_idx: Sequence[int], cooccur: np.ndarray, doc_freq: np.ndarray, n_docs: int
) -> float:
    """Mean pairwise NPMI over a topic's top tags. < 2 tags ⇒ 0.0 (a single tag
    is not a *scene*)."""
    if len(tag_idx) < 2:
        return 0.0
    pairs = list(combinations(tag_idx, 2))
    return sum(_npmi(doc_freq[a], doc_freq[b], cooccur[a, b], n_docs) for a, b in pairs) / len(
        pairs
    )


def _npmi(da: float, db: float, dab: float, n: int) -> float:
    """Normalised pointwise mutual information of two tags over the doc corpus,
    in ``[-1, 1]``: +1 = always co-occur, 0 = independent, -1 = never co-occur."""
    if dab <= 0.0:
        return -1.0
    p_ab = dab / n
    if p_ab >= 1.0:
        return 1.0
    pmi = math.log(p_ab / ((da / n) * (db / n)))
    return max(-1.0, min(1.0, pmi / (-math.log(p_ab))))


def _build_scene_cluster(
    topic: _Topic,
    w: np.ndarray,
    docs: list[tuple[str, TrackMetadataRecord, dict]],
    vocab: list[str],
    track_plays: Mapping[str, list[datetime]],
    midpoint: datetime | None,
    n_docs: int,
    params: SceneParams,
) -> dict:
    t = topic.index
    row_sums = w.sum(axis=1)
    # A doc belongs to the topic that dominates its NMF loading (argmax).
    assigned = [i for i in range(n_docs) if row_sums[i] > 0 and int(w[i].argmax()) == t]
    member_ids = [docs[i][0] for i in assigned]
    member_recs = [docs[i][1] for i in assigned]
    size = len(assigned)

    dominance = [float(w[i, t] / row_sums[i]) for i in assigned]
    confidence = round(sum(dominance) / size, 6) if size else 0.0
    # coverage = fraction of member docs that are *majority* this topic.
    coverage = round(sum(1 for d in dominance if d >= 0.5) / size, 4) if size else 0.0

    top_tags = [{"tag": vocab[j], "weight": topic.tag_weights[j]} for j in topic.tag_idx]

    artist_counts = Counter(r.artist for r in member_recs)
    top_artists = [
        {"name": name, "count_in_topic": count}
        for name, count in sorted(artist_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ][: params.top_artists_count]

    sample_order = sorted(range(size), key=lambda s: (-dominance[s], member_ids[s]))
    samples = [
        {
            "track_id": member_ids[s],
            "name": member_recs[s].name,
            "artist": member_recs[s].artist,
            "topic_weight": round(dominance[s], 4),
        }
        for s in sample_order[: params.sample_track_count]
    ]

    tag_names = ", ".join(vocab[j] for j in topic.tag_idx[:3]) or "this scene"
    return {
        "cluster_size": size,
        "min_id": min(member_ids) if member_ids else "",
        "cluster_share": size / n_docs,
        "coverage": coverage,
        "confidence": confidence,
        "stability": _stability(member_ids, track_plays, midpoint),
        "descriptor": {
            "top_tags": top_tags,
            "top_artists": top_artists,
            "coherence_score": round(topic.coherence, 4),
            "topic_index_in_K": t,
        },
        "samples": samples,
        "hint": f"amplify: seek unheard artists in the {tag_names} scene outside the listened set",
    }
