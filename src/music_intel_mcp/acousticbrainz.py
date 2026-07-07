"""Build the AcousticBrainz features JSONL (#87 AC-E, the ``MBID -> audio`` leg).

The audio pipeline reads a compact ``acousticbrainz_features.jsonl`` via
:class:`~music_intel_mcp.audio.AcousticBrainzDump`. This module *builds* that file
from AcousticBrainz for the MBIDs the identity chain already resolved.

**Mechanism substitution (decision ``d8b488ba``).** AcousticBrainz shut down Feb
2022; the data is frozen (CC0) but the query API is still live. The only bulk dump
carrying bpm is the ~590 GB low-level export — impractical. So instead of "download
the dump", we fetch our ~27k resolved MBIDs *targeted* from the live API, which
serves the same "ready-made dataset" intent feasibly. Two AB documents per MBID:

- **high-level** — mood/genre classifier probabilities in ``[0,1]``:
  ``doc["highlevel"][classifier]["all"][label]``.
- **low-level** — raw descriptors: ``doc["rhythm"]["bpm"]`` and
  ``doc["lowlevel"]["average_loudness"]``.

Both endpoints return ``{<mbid>: {"<submission-offset>": <doc>}}`` and can carry a
special ``mbid_mapping`` bookkeeping key; we read submission offset ``"0"`` and
ignore the rest.

The AB → :class:`AudioFeatures` field mapping (decision ``d8b488ba``):

===============  ================================================
AudioFeatures    AcousticBrainz source
===============  ================================================
bpm              low-level  ``rhythm.bpm``
energy           low-level  ``average_loudness`` (loudness drives Spotify-energy)
valence          high-level ``mood_happy.all.happy``
danceability     high-level ``danceability.all.danceable`` (bounded 0..1)
acousticness     high-level ``mood_acoustic.all.acoustic``
instrumentalness high-level ``voice_instrumental.all.instrumental``
===============  ================================================

Design notes:

- **Two layers for reversibility.** :func:`extract_ab_scalars` captures AB-native
  scalars (including ``aggressive``/``relaxed``, unused today) into the raw cache;
  :func:`features_from_scalars` maps them onto :class:`AudioFeatures`. Because the
  raw scalars are cached, re-mapping later is free — no re-fetch (the decision is
  ``reversible`` precisely because of this split).
- **Resumable raw cache.** Every queried MBID — hit or miss — is appended to a
  local JSONL sidecar, so a run interrupted by the harness (background tasks are
  killed at ~15 min) resumes by skipping cached MBIDs instead of re-querying from
  zero. The features JSONL is a pure projection of that cache, rewritten each run.
- **High-level-first gate.** Low-level is fetched only for MBIDs that hit
  high-level: a high-level miss has no valence/danceability, so it can never enter
  the four-axis cluster — its (larger) low-level call would be wasted.

Network-only: ``httpx`` is imported lazily and tests mock it with ``respx`` — the
live endpoint is never touched in CI. The raw cache is per-user-local, never cloud
(history-never-leaves-the-machine); AB features themselves are shared, anonymous,
CC0 track-level metadata.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .shared_store import AudioFeatures

AB_HIGHLEVEL_URL = "https://acousticbrainz.org/api/v1/high-level"
AB_LOWLEVEL_URL = "https://acousticbrainz.org/api/v1/low-level"

_BATCH = 25  # AB accepts ~25 recording_ids per bulk call.
_MAX_RETRIES = 5
_USER_AGENT = "music-intel-mcp/0.0.1 ( https://github.com/Osasuwu/music-intel-mcp )"
_SOURCE = "acousticbrainz_api"

# The AB-native scalar keys carried in the raw cache. The first six back the
# current :class:`AudioFeatures` mapping; ``aggressive``/``relaxed`` are captured
# for a future re-map without a re-fetch (decision d8b488ba).
_SCALAR_KEYS = (
    "bpm",
    "average_loudness",
    "danceable",
    "happy",
    "acoustic",
    "instrumental",
    "aggressive",
    "relaxed",
)


# --------------------------------------------------------------------------- #
# Layer 1 — pure mapping (domain-correctness critical, /grill-gated)
# --------------------------------------------------------------------------- #


def _prob(hl_doc: dict | None, classifier: str, label: str) -> float | None:
    """A high-level classifier probability ``highlevel[classifier].all[label]``,
    or ``None`` when the doc/path/value is absent or non-numeric."""
    if not hl_doc:
        return None
    all_probs = hl_doc.get("highlevel", {}).get(classifier, {}).get("all", {})
    value = all_probs.get(label) if isinstance(all_probs, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _ll_scalar(ll_doc: dict | None, section: str, key: str) -> float | None:
    """A low-level scalar ``<section>[key]``, or ``None`` when absent/non-numeric."""
    if not ll_doc:
        return None
    section_obj = ll_doc.get(section, {})
    value = section_obj.get(key) if isinstance(section_obj, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def extract_ab_scalars(hl_doc: dict | None, ll_doc: dict | None) -> dict | None:
    """Pull the AB-native scalars from the offset-0 high/low-level docs.

    Returns a dict keyed by :data:`_SCALAR_KEYS`, or ``None`` when neither doc
    yields a single value (a total miss). A partial hit (e.g. high-level only) is
    kept — the absent scalars are ``None`` and the track simply won't cluster."""
    scalars = {
        "bpm": _ll_scalar(ll_doc, "rhythm", "bpm"),
        "average_loudness": _ll_scalar(ll_doc, "lowlevel", "average_loudness"),
        "danceable": _prob(hl_doc, "danceability", "danceable"),
        "happy": _prob(hl_doc, "mood_happy", "happy"),
        "acoustic": _prob(hl_doc, "mood_acoustic", "acoustic"),
        "instrumental": _prob(hl_doc, "voice_instrumental", "instrumental"),
        "aggressive": _prob(hl_doc, "mood_aggressive", "aggressive"),
        "relaxed": _prob(hl_doc, "mood_relaxed", "relaxed"),
    }
    if all(v is None for v in scalars.values()):
        return None
    return scalars


def features_from_scalars(scalars: dict | None) -> AudioFeatures | None:
    """Map AB-native scalars onto :class:`AudioFeatures` (decision ``d8b488ba``).

    Returns ``None`` when every mapped field is absent — nothing worth writing."""
    if not scalars:
        return None
    feats = AudioFeatures(
        bpm=scalars.get("bpm"),
        energy=scalars.get("average_loudness"),
        valence=scalars.get("happy"),
        danceability=scalars.get("danceable"),
        acousticness=scalars.get("acoustic"),
        instrumentalness=scalars.get("instrumental"),
        source=_SOURCE,
    )
    mapped = (
        feats.bpm,
        feats.energy,
        feats.valence,
        feats.danceability,
        feats.acousticness,
        feats.instrumentalness,
    )
    if all(v is None for v in mapped):
        return None
    return feats


# --------------------------------------------------------------------------- #
# Layer 2 — live AcousticBrainz API client
# --------------------------------------------------------------------------- #


class AcousticBrainzApiClient:
    """Bulk reader over the (frozen but live) AcousticBrainz query API.

    :meth:`fetch_highlevel` / :meth:`fetch_lowlevel` take any number of MBIDs,
    chunk them ``;``-joined at :data:`_BATCH`, and return ``{mbid: offset-0 doc}``
    for the hits only (a missing MBID is simply absent, like a cache miss). ``sleep``
    is injectable so tests exercise the 429/503 backoff without real delays;
    ``api_calls`` counts logical batch fetches for telemetry/tests."""

    def __init__(
        self,
        *,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
    ) -> None:
        self._sleep = sleep
        self._timeout = timeout
        self.api_calls = 0

    def _request(self, httpx, url: str, params: dict):
        """One GET with retry on 429/503 (``Retry-After`` honored, else exponential),
        capped at :data:`_MAX_RETRIES`. Mirrors the Spotify client's backoff."""
        resp = None
        for attempt in range(_MAX_RETRIES):
            resp = httpx.request(
                "GET",
                url,
                params=params,
                headers={"User-Agent": _USER_AGENT},
                timeout=self._timeout,
            )
            if resp.status_code in (429, 503):
                retry_after = resp.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.strip().isdigit()
                    else 2.0**attempt
                )
                self._sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def _fetch(self, url: str, mbids: Sequence[str]) -> dict[str, dict]:
        import httpx

        ids = [m for m in mbids if m]
        out: dict[str, dict] = {}
        for i in range(0, len(ids), _BATCH):
            chunk = ids[i : i + _BATCH]
            resp = self._request(httpx, url, {"recording_ids": ";".join(chunk)})
            self.api_calls += 1
            out.update(_parse_ab_response(resp.json()))
        return out

    def fetch_highlevel(self, mbids: Sequence[str]) -> dict[str, dict]:
        return self._fetch(AB_HIGHLEVEL_URL, mbids)

    def fetch_lowlevel(self, mbids: Sequence[str]) -> dict[str, dict]:
        return self._fetch(AB_LOWLEVEL_URL, mbids)


def _parse_ab_response(payload: dict) -> dict[str, dict]:
    """``{mbid: {"0": doc, ...}}`` -> ``{mbid: offset-0 doc}``. The API's own
    ``mbid_mapping`` bookkeeping key and any non-dict submission are skipped."""
    out: dict[str, dict] = {}
    for mbid, submissions in payload.items():
        if mbid == "mbid_mapping" or not isinstance(submissions, dict):
            continue
        doc = submissions.get("0")
        if isinstance(doc, dict):
            out[mbid] = doc
    return out


# --------------------------------------------------------------------------- #
# Layer 3 — resumable builder
# --------------------------------------------------------------------------- #


@dataclass
class AcousticBrainzBuildReport:
    """Outcome of a :func:`build_acousticbrainz_index` run — the AC-E honest
    measurement of the feature leg.

    - ``fetched`` — MBIDs newly queried this run (0 on a fully-cached re-run).
    - ``hits`` — of those, ones AB had any scalars for.
    - ``misses`` — of those, ones AB had nothing for (negative-cached).
    - ``written`` — rows in the projected features JSONL (MBIDs whose scalars map
      to a non-empty :class:`AudioFeatures`).
    - ``total_cached`` — MBIDs in the raw cache after the run.
    - ``cached_hits`` / ``cached_misses`` — hit/miss split over the *whole* cache
      (cumulative across resumed runs), so ``cached_hits / total_cached`` is the
      honest cumulative P(features|MBID), not just this run's slice.
    """

    fetched: int
    hits: int
    misses: int
    written: int
    total_cached: int
    cached_hits: int
    cached_misses: int


def _load_raw_cache(path: Path) -> dict[str, dict | None]:
    """``mbid -> scalars`` (``None`` encodes a known miss). A torn final line from
    an interrupted run is tolerated (skipped), like the Spotify cache."""
    cache: dict[str, dict | None] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mbid = row.get("mbid")
            if mbid:
                cache[mbid] = row.get("scalars")
    return cache


def _write_features(out_path: Path, cache: dict[str, dict | None]) -> int:
    """Project the raw scalar cache into the features JSONL the audio pipeline
    reads. Rewritten whole (not appended) so it stays a clean, dedup-free
    projection. Returns rows written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for mbid, scalars in cache.items():
            feats = features_from_scalars(scalars)
            if feats is None:
                continue
            out.write(
                json.dumps(
                    {
                        "mbid": mbid,
                        "bpm": feats.bpm,
                        "energy": feats.energy,
                        "valence": feats.valence,
                        "danceability": feats.danceability,
                        "acousticness": feats.acousticness,
                        "instrumentalness": feats.instrumentalness,
                    }
                )
                + "\n"
            )
            written += 1
    return written


def build_acousticbrainz_index(
    mbids: Sequence[str],
    client: AcousticBrainzApiClient,
    *,
    out_path: str | Path,
    raw_cache_path: str | Path,
) -> AcousticBrainzBuildReport:
    """Fetch AB features for ``mbids`` and build the features JSONL at ``out_path``.

    Resumable: MBIDs already in ``raw_cache_path`` (hit or miss) are skipped, so a
    run cut short by the harness re-runs cheaply. Per batch: fetch high-level, then
    low-level only for the high-level hits (the gate), extract scalars, append the
    raw cache. The features JSONL is (re)projected from the full cache at the end,
    so even a fully-cached re-run refreshes the output.
    """
    out_path, raw_cache_path = Path(out_path), Path(raw_cache_path)
    cache = _load_raw_cache(raw_cache_path)

    todo: list[str] = []
    seen = set(cache)
    for mbid in mbids:
        if mbid and mbid not in seen:
            seen.add(mbid)
            todo.append(mbid)

    fetched = hits = misses = 0
    if todo:
        raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            for i in range(0, len(todo), _BATCH):
                batch = todo[i : i + _BATCH]
                highlevel = client.fetch_highlevel(batch)
                # High-level-first gate: only hl-hits can cluster, so only they
                # are worth a low-level call.
                lowlevel = client.fetch_lowlevel(list(highlevel)) if highlevel else {}
                for mbid in batch:
                    scalars = extract_ab_scalars(highlevel.get(mbid), lowlevel.get(mbid))
                    cache[mbid] = scalars
                    fh.write(json.dumps({"mbid": mbid, "scalars": scalars}) + "\n")
                    fetched += 1
                    if scalars is None:
                        misses += 1
                    else:
                        hits += 1
                fh.flush()

    written = _write_features(out_path, cache)
    cached_misses = sum(1 for scalars in cache.values() if scalars is None)
    return AcousticBrainzBuildReport(
        fetched=fetched,
        hits=hits,
        misses=misses,
        written=written,
        total_cached=len(cache),
        cached_hits=len(cache) - cached_misses,
        cached_misses=cached_misses,
    )
