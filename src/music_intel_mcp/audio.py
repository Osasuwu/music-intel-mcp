"""Audio root pipeline (#63) — the deep module behind the audio category.

One pipeline, two halves (decision ``01ba9bb7``):

1. **Enrichment.** :func:`enrich_audio_features` populates ``audio_features`` on
   shared-store records by MBID, pulling from an :class:`AudioFeatureSource`
   (the env-pointed :class:`AcousticBrainzDump` in production, an in-memory map
   in tests). It writes facts back to the store and reports coverage plus a
   transparent breakdown of what it could *not* enrich (no MBID, no dump entry).

2. **Derivation.** :func:`derive_audio_roots` z-scores the configured
   ``cluster_features``, fits HDBSCAN once over the whole user population, and
   turns each cluster 1:1 into a :class:`~music_intel_mcp.validation.Candidate`
   carrying a complete structural descriptor (four bands + a raw centroid),
   evidence (size/share/coverage + sample tracks), a chronological-split
   temporal-stability score, and a confidence = mean HDBSCAN membership
   probability. The batch is handed to the #62 validator, which decides
   root / tendency / artifact_suspect against ``method_params``.

Determinism: feature rows are sorted by canonical id before clustering, and
sklearn's HDBSCAN has no random initialisation — same input ⇒ identical roots.
The optional LLM ``curator_prose`` is left ``None`` in V0.

The AcousticBrainz dump lives **outside the repo** (env-pointed); tests use
synthetic features only and never touch the real dump or any live API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.cluster import HDBSCAN

from .models import AudioParams, ValidationParams
from .shared_store import AudioFeatures, SharedStore, TrackMetadataRecord
from .validation import Candidate, DatasetContext, ValidationOutcome, Validator

# Env metadata for the production AcousticBrainz dump (the dump itself lives
# outside the repo — see CLAUDE.md). Values resolve at lookup time; missing →
# an empty index, so an uninstalled dump degrades to honest low coverage.
_AB_DUMP_PATH_ENV = "ACOUSTICBRAINZ_FEATURES_INDEX"
_AB_DUMP_DIR_ENV = "ACOUSTICBRAINZ_DUMP_DIR"
_AB_DEFAULT_FILENAME = "acousticbrainz_features.jsonl"


# --------------------------------------------------------------------------- #
# Enrichment — audio feature sources
# --------------------------------------------------------------------------- #


@runtime_checkable
class AudioFeatureSource(Protocol):
    """Looks audio features up by MBID. The dump in production, a map in tests."""

    def lookup(self, mbid: str) -> AudioFeatures | None:
        """Return features for ``mbid`` or ``None`` when the source has none."""
        ...


class InMemoryAudioFeatureSource:
    """Dict-backed :class:`AudioFeatureSource` for tests. ``lookups`` records each
    MBID queried so tests can assert no wasted lookups (no-MBID / already-enriched
    tracks must never reach the source)."""

    def __init__(self, mapping: Mapping[str, AudioFeatures] | None = None) -> None:
        self._mapping = dict(mapping or {})
        self.lookups: list[str] = []

    def lookup(self, mbid: str) -> AudioFeatures | None:
        self.lookups.append(mbid)
        return self._mapping.get(mbid)


class AcousticBrainzDump:
    """Production :class:`AudioFeatureSource` over an env-pointed JSONL dump.

    One JSON object per line, keyed by ``mbid``; the scalar columns map straight
    onto :class:`AudioFeatures`. The path resolves from an explicit argument,
    then ``ACOUSTICBRAINZ_FEATURES_INDEX``, then
    ``$ACOUSTICBRAINZ_DUMP_DIR/acousticbrainz_features.jsonl``. A missing file is
    an empty index (every lookup ``None``) — never an error, so the pipeline
    runs with honest low coverage when the dump is not installed. The index is
    loaded lazily on first lookup and cached.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = _resolve_dump_path(path)
        self._index: dict[str, AudioFeatures] | None = None

    def _load(self) -> dict[str, AudioFeatures]:
        index: dict[str, AudioFeatures] = {}
        if self._path is None or not self._path.exists():
            return index
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mbid = row.get("mbid")
                if mbid:
                    index[mbid] = _features_from_row(row)
        return index

    def lookup(self, mbid: str) -> AudioFeatures | None:
        if self._index is None:
            self._index = self._load()
        return self._index.get(mbid)


def _resolve_dump_path(path: str | Path | None) -> Path | None:
    if path is not None:
        return Path(path)
    explicit = os.environ.get(_AB_DUMP_PATH_ENV)
    if explicit:
        return Path(explicit)
    dump_dir = os.environ.get(_AB_DUMP_DIR_ENV)
    if dump_dir:
        return Path(dump_dir) / _AB_DEFAULT_FILENAME
    return None


def _features_from_row(row: dict) -> AudioFeatures:
    return AudioFeatures(
        bpm=row.get("bpm"),
        energy=row.get("energy"),
        valence=row.get("valence"),
        danceability=row.get("danceability"),
        acousticness=row.get("acousticness"),
        instrumentalness=row.get("instrumentalness"),
        source="acousticbrainz_dump",
    )


# --------------------------------------------------------------------------- #
# Enrichment — orchestrator
# --------------------------------------------------------------------------- #


@dataclass
class AudioEnrichmentReport:
    """Outcome of an :func:`enrich_audio_features` run. Every considered track
    lands in exactly one bucket — transparent rejection, nothing silently lost.

    - ``enriched`` — gained features from the source this run (written back).
    - ``already_present`` — carried features already; source not queried.
    - ``missing_features`` — has an MBID but the source had no entry.
    - ``no_mbid`` — cannot be looked up (no MBID); never queried.
    """

    enriched: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    no_mbid: list[str] = field(default_factory=list)

    @property
    def total_considered(self) -> int:
        return (
            len(self.enriched)
            + len(self.already_present)
            + len(self.missing_features)
            + len(self.no_mbid)
        )

    @property
    def coverage(self) -> float:
        """Fraction of considered tracks that carry audio features afterwards."""
        n = self.total_considered
        if n == 0:
            return 0.0
        return (len(self.enriched) + len(self.already_present)) / n


def enrich_audio_features(
    track_ids: Sequence[str],
    store: SharedStore,
    source: AudioFeatureSource,
    *,
    now: datetime,
) -> AudioEnrichmentReport:
    """Populate ``audio_features`` on the store's records for ``track_ids``.

    One bulk read, one bulk write-back (the shared-store round-trip discipline).
    Tracks already carrying features are skipped (no source lookup); tracks
    without an MBID cannot be looked up and are reported, not queried. Records
    absent from the store are not considered (the caller seeds them first).
    """
    records = store.get_tracks(list(dict.fromkeys(track_ids)))
    report = AudioEnrichmentReport()
    to_write: list[TrackMetadataRecord] = []

    for tid, record in records.items():
        if record.audio_features is not None:
            report.already_present.append(tid)
            continue
        if not record.mbid:
            report.no_mbid.append(tid)
            continue
        features = source.lookup(record.mbid)
        if features is None:
            report.missing_features.append(tid)
            continue
        record.audio_features = features
        to_write.append(record)
        report.enriched.append(tid)

    if to_write:
        store.upsert_tracks(to_write)
    return report


# --------------------------------------------------------------------------- #
# Derivation — clustering → structural descriptor → validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AudioDerivation:
    """Result of :func:`derive_audio_roots`.

    - ``outcome`` — the validated roots / tendencies / quality_log.
    - ``coverage`` — fraction of the user's records with usable cluster features
      (the audio entry of ``generated_from.coverage_per_category``).
    - ``n_clustered`` — feature vectors fed to HDBSCAN.
    - ``n_noise`` — points HDBSCAN left unclustered (label -1).
    - ``n_clusters`` — clusters found (1:1 with candidates before validation).
    - ``members`` — candidate id (``r-audio-N``) → its member canonical track
      ids, for the temporal stage (#65) to condition lift on root membership.
    """

    outcome: ValidationOutcome
    coverage: float
    n_clustered: int
    n_noise: int
    n_clusters: int
    members: dict[str, list[str]]


def derive_audio_roots(
    records: Mapping[str, TrackMetadataRecord],
    track_plays: Mapping[str, list[datetime]],
    *,
    params: AudioParams,
    validation_params: ValidationParams,
    dataset_ctx: DatasetContext,
) -> AudioDerivation:
    """Cluster enriched tracks into audio roots, then validate.

    ``records`` is the user's track population keyed by canonical id;
    ``track_plays`` maps the same ids to play timestamps (drives the temporal
    split). Tracks missing any ``cluster_features`` dim cannot enter clustering.
    A population smaller than ``min_cluster_size`` cannot form a cluster, so it
    short-circuits to an honest-empty outcome.
    """
    cluster_features = params.cluster_features
    total = len(records)

    clusterable = sorted(
        (r for r in records.values() if _has_features(r, cluster_features)),
        key=lambda r: r.track_id,
    )
    n_clustered = len(clusterable)
    coverage = n_clustered / total if total else 0.0

    if n_clustered < params.min_cluster_size:
        # Too few points to form even one cluster — everything is noise.
        return AudioDerivation(ValidationOutcome(), coverage, n_clustered, n_clustered, 0, {})

    x_raw = np.array(
        [[_feature(r, dim) for dim in cluster_features] for r in clusterable],
        dtype=float,
    )
    std = x_raw.std(axis=0)
    x_z = (x_raw - x_raw.mean(axis=0)) / np.where(std == 0.0, 1.0, std)

    # copy=True: we reuse x_z below for centroids/sample distances, so HDBSCAN
    # must not mutate it in place (the pre-1.10 default would).
    clusterer = HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        copy=True,
    )
    labels = clusterer.fit_predict(x_z)
    probabilities = clusterer.probabilities_

    midpoint = _history_midpoint(track_plays)
    cluster_labels = sorted({int(label) for label in labels if label >= 0})

    built: list[dict] = []
    for label in cluster_labels:
        members = [i for i, lab in enumerate(labels) if lab == label]
        built.append(
            _build_cluster(
                members,
                clusterable,
                x_raw,
                x_z,
                probabilities,
                track_plays,
                midpoint,
                params,
                n_clustered,
            )
        )

    # Rank clusters by size (desc), tie-broken by smallest member id, so ids are
    # deterministic and the prominent root is r-audio-1.
    built.sort(key=lambda c: (-c["cluster_size"], c["min_id"]))
    candidates = [
        Candidate(
            candidate_id=f"r-audio-{rank}",
            category="audio",
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
    members = {f"r-audio-{rank}": c["member_ids"] for rank, c in enumerate(built, start=1)}

    outcome = Validator(validation_params).validate(candidates, dataset_ctx)
    n_noise = int(np.sum(labels == -1))
    return AudioDerivation(outcome, coverage, n_clustered, n_noise, len(cluster_labels), members)


# --------------------------------------------------------------------------- #
# Derivation helpers
# --------------------------------------------------------------------------- #


def _has_features(record: TrackMetadataRecord, dims: Sequence[str]) -> bool:
    af = record.audio_features
    return af is not None and all(getattr(af, dim, None) is not None for dim in dims)


def _feature(record: TrackMetadataRecord, dim: str) -> float:
    return float(getattr(record.audio_features, dim))


def _band(value: float, cutoffs: tuple[float, float]) -> str:
    lo, hi = cutoffs
    if value < lo:
        return "low"
    if value < hi:
        return "mid"
    return "high"


def _history_midpoint(track_plays: Mapping[str, list[datetime]]) -> datetime | None:
    times = [t for plays in track_plays.values() for t in plays]
    if not times:
        return None
    earliest, latest = min(times), max(times)
    return earliest + (latest - earliest) / 2


def _build_cluster(
    members: list[int],
    clusterable: list[TrackMetadataRecord],
    x_raw: np.ndarray,
    x_z: np.ndarray,
    probabilities: np.ndarray,
    track_plays: Mapping[str, list[datetime]],
    midpoint: datetime | None,
    params: AudioParams,
    n_clustered: int,
) -> dict:
    cluster_features = params.cluster_features
    member_records = [clusterable[i] for i in members]
    size = len(members)

    centroid_raw = x_raw[members].mean(axis=0)
    descriptor: dict = {}
    for dim, value in zip(cluster_features, centroid_raw, strict=True):
        descriptor[f"{dim}_band"] = _band(float(value), params.band_cutoffs[dim])
    descriptor["centroid_raw"] = {
        dim: round(float(value), 4)
        for dim, value in zip(cluster_features, centroid_raw, strict=True)
    }

    # sample tracks: closest members to the cluster centroid in z-space
    z_centroid = x_z[members].mean(axis=0)
    distances = np.linalg.norm(x_z[members] - z_centroid, axis=1)
    order = sorted(range(size), key=lambda k: (float(distances[k]), member_records[k].track_id))
    samples = [
        {
            "track_id": member_records[k].track_id,
            "name": member_records[k].name,
            "artist": member_records[k].artist,
            "distance_to_centroid": round(float(distances[k]), 4),
        }
        for k in order[: params.sample_track_count]
    ]

    covered = sum(1 for r in member_records if _has_features(r, params.feature_set))
    band_summary = " / ".join(f"{descriptor[f'{dim}_band']} {dim}" for dim in cluster_features)

    return {
        "cluster_size": size,
        "min_id": min(r.track_id for r in member_records),
        "member_ids": [r.track_id for r in member_records],
        "cluster_share": size / n_clustered,
        "coverage": covered / size,
        "confidence": round(float(probabilities[members].mean()), 6),
        "stability": _stability([r.track_id for r in member_records], track_plays, midpoint),
        "descriptor": descriptor,
        "samples": samples,
        "hint": (
            f"amplify: seek unheard tracks near {band_summary}; "
            "dampen: cap weekly plays of this cluster"
        ),
    }


def _stability(
    member_ids: list[str],
    track_plays: Mapping[str, list[datetime]],
    midpoint: datetime | None,
) -> float | None:
    """Chronological-split stability: 1 - |first_half_frac - second_half_frac|
    over the cluster's plays. ``None`` (→ not_evaluated) when the cluster has no
    play timestamps or the history has no midpoint."""
    if midpoint is None:
        return None
    times = [t for tid in member_ids for t in track_plays.get(tid, [])]
    if not times:
        return None
    before = sum(1 for t in times if t < midpoint)
    total = len(times)
    after = total - before
    return round(1 - abs(before / total - after / total), 4)
