"""Per-user store — local plain files, no DB (decision f7a9fcbd).

Layout under the data root (default ``data/``, overridable via the
``MUSIC_INTEL_DATA_DIR`` env var or the ``UserStore(root=...)`` argument):

- ``history.jsonl`` — append-only listening events, one JSON object per line.
- ``profiles/<snapshot>.json`` — RootProfile time-series snapshots.

Personal data lives here and *only* here — never to the shared metadata store
(history-never-leaves-the-machine invariant).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Library, ListenEvent, RootProfile


class ConsentFormatError(Exception):
    """Raised by :meth:`UserStore.has_automated_playback_consent` when the
    on-disk consent file predates #165's grantor+timestamp+scope schema."""


@dataclass(frozen=True)
class AudioAnalysisRecord:
    """One persisted live-capture inference result, read back from
    ``audio_analysis/*.json`` (#125 AC1). ``embedding`` is the raw
    Discogs-EffNet vector (~1280-dim in production); ``tags`` are the
    MTG-Jamendo classifier scores — descriptive only, never clustering input
    (decision 0c762eec)."""

    track_id: str
    embedding: list[float]
    tags: dict[str, float]
    provenance: dict[str, Any] | None = None
    model_version: str | None = None
    input_rms: float | None = None


DEFAULT_DATA_DIR = "data"
_DATA_DIR_ENV = "MUSIC_INTEL_DATA_DIR"

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

# #158 AC4: pre-#158 audio-analysis files carry a *bare* key -- whichever of
# mbid/isrc/spotify_id/name_key the live waterfall picked first, written
# without our current ``mbid:``/``isrc:``/``spotify:``/``name:`` prefix. The
# bare string itself carries no type tag, so the one-shot migration below
# classifies it back by format (lengths/alphabets don't overlap between the
# three id kinds) and falls back to rebuilding the ``name:`` form from the
# provenance sidecar's raw title/artist when nothing matches.
_CANONICAL_PREFIXES = ("mbid:", "isrc:", "spotify:", "name:")
_MBID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_ISRC_RE = re.compile(r"^[A-Za-z]{2}[A-Za-z0-9]{3}\d{7}$")
_SPOTIFY_ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")


@dataclass(frozen=True)
class KeyMigrationReport:
    """Outcome of one :func:`migrate_audio_analysis_keys` run. ``migrated``
    and ``conflicts`` are ``(old_bare_id, new_prefixed_id)`` pairs -- a
    conflict means the prefixed target already existed (first-write-wins:
    the pre-existing file is kept, the bare file is left untouched)."""

    migrated: list[tuple[str, str]]
    conflicts: list[tuple[str, str]]


def _infer_canonical_key(bare_id: str, provenance: dict[str, Any] | None) -> str | None:
    if _MBID_RE.match(bare_id):
        return f"mbid:{bare_id}"
    if _ISRC_RE.match(bare_id):
        return f"isrc:{bare_id}"
    if _SPOTIFY_ID_RE.match(bare_id):
        return f"spotify:{bare_id}"
    if provenance and provenance.get("raw_title") and provenance.get("raw_artist"):
        # Local import: live_identity imports from this module (resolve_data_root),
        # so a module-level import here would be circular.
        from .live_identity import normalize_track_name

        name_key = normalize_track_name(provenance["raw_title"], provenance["raw_artist"])
        return f"name:{name_key}"
    return None


def migrate_audio_analysis_keys(store: UserStore) -> KeyMigrationReport:
    """One-shot #158 AC4 migration: rename bare-key ``audio_analysis/*.json``
    files to the canonical prefixed ``canonical_track_id`` form so file
    names, dedup lookups and clustering all key off the same identity waterfall.
    Idempotent -- an already-prefixed ``track_id`` is left alone, so a second
    run over already-migrated data is a no-op."""
    migrated: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []
    if not store.audio_analysis_dir.exists():
        return KeyMigrationReport(migrated=migrated, conflicts=conflicts)

    for path in sorted(store.audio_analysis_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        old_id = payload["track_id"]
        if old_id.startswith(_CANONICAL_PREFIXES):
            continue
        new_id = _infer_canonical_key(old_id, payload.get("provenance"))
        if new_id is None:
            conflicts.append((old_id, "<unclassifiable>"))
            continue
        new_path = store.audio_analysis_path(new_id)
        if new_path.exists() and new_path != path:
            conflicts.append((old_id, new_id))
            continue
        payload["track_id"] = new_id
        new_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if new_path != path:
            path.unlink()
        migrated.append((old_id, new_id))

    return KeyMigrationReport(migrated=migrated, conflicts=conflicts)


def load_aliases(path: Path) -> dict[str, str]:
    """#140 AC4/AC5: read an ``aliases.jsonl`` sidecar into a ``loser ->
    winner`` map. Honest-empty when the file has never been written --
    ``apply`` may never have run yet."""
    if not path.exists():
        return {}
    aliases: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            aliases[record["loser"]] = record["winner"]
    return aliases


def resolve_key(
    key: str,
    *,
    pool_aliases: dict[str, str] | None = None,
    root_aliases: dict[str, str] | None = None,
    max_chain: int = 64,
) -> str:
    """#140 AC5, precedence inverted by #170 AC7 (decision 7a40049d): follow
    an alias chain to its winner, checking the participant-root map before
    the pool map at every hop (participant-root-then-pool precedence) -- a
    participant's own accepted near-dup/crosswalk merge should not be
    silently overridden by a pool-wide alias. A cycle guard -- bounded hop
    count plus a seen-set -- stops the walk and returns the last key reached
    rather than looping forever; a genuine cycle should never occur
    (``apply`` only ever points loser -> winner by rank), but this function
    does not trust that invariant blindly."""
    pool_aliases = pool_aliases or {}
    root_aliases = root_aliases or {}
    current = key
    seen = {current}
    for _ in range(max_chain):
        if current in root_aliases:
            next_key = root_aliases[current]
        elif current in pool_aliases:
            next_key = pool_aliases[current]
        else:
            break
        if next_key in seen:
            break
        seen.add(next_key)
        current = next_key
    return current


def resolve_data_root(root: str | Path | None) -> Path:
    """Resolve the data root: explicit arg > ``MUSIC_INTEL_DATA_DIR`` > default.
    Shared by the per-user store and the (local) shared-metadata cache."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(_DATA_DIR_ENV, DEFAULT_DATA_DIR))


class UserStore:
    """Read history, read/write RootProfile snapshots for one user."""

    def __init__(
        self, root: str | Path | None = None, *, pool_root: str | Path | None = None
    ) -> None:
        self.root = resolve_data_root(root)
        self.pool_root = Path(pool_root) if pool_root is not None else None

    @property
    def history_path(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def library_path(self) -> Path:
        return self.root / "library.json"

    # --- history --------------------------------------------------------- #

    def load_history(self) -> list[ListenEvent]:
        """Parse every line of ``history.jsonl`` into a ``ListenEvent``.
        Missing file -> empty history (a valid honest-empty input)."""
        if not self.history_path.exists():
            return []
        events: list[ListenEvent] = []
        with self.history_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                events.append(ListenEvent.model_validate_json(line))
        return events

    def append_events(self, events: list[ListenEvent]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.model_dump_json() + "\n")

    def replace_history(self, events: list[ListenEvent]) -> None:
        """Rewrite ``history.jsonl`` from scratch (overwrite, not append).
        Used by idempotent importers that merge+dedup, then write the full
        history back so re-running the same source yields the same file."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(event.model_dump_json() + "\n")

    # --- library (explicit-preference layer, #97) ------------------------ #

    def load_library(self) -> Library | None:
        """Parse ``library.json`` into a typed ``Library``. Missing file ->
        ``None`` (honest-empty: no explicit-preference import has run yet),
        mirroring ``latest_profile``'s None-on-absent idiom."""
        if not self.library_path.exists():
            return None
        return Library.model_validate_json(self.library_path.read_text(encoding="utf-8"))

    def write_library(self, library: Library) -> Path:
        """Write the current-state ``library.json`` (overwrite, not append).
        Re-importing the same export yields byte-identical output — the file is
        the single current-state document, so a re-import replaces it."""
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        self.library_path.write_text(
            library.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return self.library_path

    # --- profiles -------------------------------------------------------- #

    def write_profile(self, profile: RootProfile) -> Path:
        """Serialize a snapshot to ``profiles/<sanitized snapshot_id>.json``."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        path = self.profiles_dir / f"{self._safe_name(profile.snapshot_id)}.json"
        path.write_text(
            profile.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def read_profile(self, path: str | Path) -> RootProfile:
        return RootProfile.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def latest_profile(self) -> RootProfile | None:
        """Most recent snapshot by filename (snapshot ids are timestamp-led)."""
        if not self.profiles_dir.exists():
            return None
        snapshots = sorted(self.profiles_dir.glob("*.json"))
        if not snapshots:
            return None
        return self.read_profile(snapshots[-1])

    @staticmethod
    def _safe_name(snapshot_id: str) -> str:
        return _UNSAFE_FILENAME.sub("_", snapshot_id)

    # --- live-capture audio analysis (#124 AC5) --------------------------- #

    @property
    def audio_analysis_dir(self) -> Path:
        return self.root / "audio_analysis"

    def audio_analysis_path(self, track_id: str) -> Path:
        return self.audio_analysis_dir / f"{self._safe_name(track_id)}.json"

    # --- node-level anonymous pool (#161) ---------------------------------- #
    #
    # Pilot topology (decision 1fe8e95f): the participant root is transient
    # (deleted after delivery), the pool is the only thing that persists
    # (decision 92241497). A pool record is deliberately a narrower shape than
    # a root record -- key + embedding + tags + model_version, no provenance
    # (decision 446d7d0a) -- because provenance (raw title/artist, source app
    # id, capture timestamp) would make a shared multi-participant pool a
    # re-identifiable copy of one participant's play list.

    @property
    def pool_audio_analysis_dir(self) -> Path | None:
        return self.pool_root / "audio_analysis" if self.pool_root is not None else None

    def pool_audio_analysis_path(self, track_id: str) -> Path | None:
        pool_dir = self.pool_audio_analysis_dir
        return pool_dir / f"{self._safe_name(track_id)}.json" if pool_dir is not None else None

    def has_audio_analysis(self, track_id: str) -> bool:
        """#126 AC1/AC4: dedup check the live pipeline runs via the identity
        waterfall's resolved ``track_id`` before paying for inference.

        #161 AC2: when a pool is configured, the pool is consulted first --
        an already-analyzed track (from any participant) dedupes even if this
        participant's own transient root never saw it -- falling back to the
        participant root for pre-pool or root-only data.

        #140 AC5: the key is resolved through recorded aliases first -- the
        loser side of an accepted near-dup merge counts as analysed once its
        winner has been. Scope is deliberately narrow to this call site
        (decision ce94e03c-acf1-4533-99cf-2fbbca36b3c4); the replay queue
        selector and pool-history intersection are not wired here."""
        track_id = self.resolve_track_key(track_id)
        pool_path = self.pool_audio_analysis_path(track_id)
        if pool_path is not None and pool_path.exists():
            return True
        return self.audio_analysis_path(track_id).exists()

    def write_audio_analysis(
        self,
        *,
        track_id: str,
        embedding: Any,
        tags: dict[str, float],
        provenance: Any | None = None,
        model_version: str | None = None,
        input_rms: float | None = None,
    ) -> Path:
        """Write one live-capture inference result under the LOCAL store only
        (#124 AC5). MTG-Jamendo outputs are licensing-gated local-only
        (decision 29852699); ``UserStore`` never talks to Supabase/SharedStore,
        so this path is structurally local-only, not just conventionally so.

        ``provenance`` (#139 AC5) is the live-capture sidecar — raw title/
        artist, source app id, capture timestamp, chromaprint fingerprint —
        stored alongside the embedding so a future identity-strategy change is
        a re-mapping job over sidecars, never a re-listen. Accepts anything
        with a ``model_dump()`` (a :class:`~music_intel_mcp.live_identity.
        ProvenanceSidecar`) or a plain dict; ``None`` for the pre-#139 batch
        path, which has no live capture metadata to attach.

        #126 AC2: first-write-wins. Two near-simultaneous writers for the same
        ``track_id`` (e.g. two overlapping capture sessions) must not average
        or clobber each other's embedding — the file is claimed atomically via
        ``O_CREAT | O_EXCL`` at the OS level; a losing writer's payload is
        silently discarded and the winner's path is returned either way.

        #161 AC2: when a pool is configured, the analysis is written
        exclusively to the pool, never to the participant root -- and the
        pool record excludes ``provenance`` entirely (not even as a ``null``
        key) in favor of ``model_version``. Without a pool this is the
        original #124/#139 root write, unchanged.

        #194 AC6/AC7/AC9: ``model_version`` (the embedding-space version that
        produced this record) and ``input_rms`` (the pre-normalization RMS
        applied-gain scalar) round-trip through BOTH branches now -- previously
        ``model_version`` was pool-only, leaving root records with no way to
        tell which embedding-space version produced them."""
        pool_path = self.pool_audio_analysis_path(track_id)
        if pool_path is not None:
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._audio_analysis_payload(
                track_id, embedding, tags, model_version=model_version, input_rms=input_rms
            )
            return self._atomic_write(pool_path, payload)

        self.audio_analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_analysis_path(track_id)
        if provenance is None:
            provenance_payload = None
        elif hasattr(provenance, "model_dump"):
            provenance_payload = provenance.model_dump()
        else:
            provenance_payload = dict(provenance)
        payload = self._audio_analysis_payload(
            track_id, embedding, tags, model_version=model_version, input_rms=input_rms
        )
        payload["provenance"] = provenance_payload
        return self._atomic_write(path, payload)

    @staticmethod
    def _audio_analysis_payload(
        track_id: str,
        embedding: Any,
        tags: dict[str, float],
        *,
        model_version: str | None = None,
        input_rms: float | None = None,
    ) -> dict[str, Any]:
        """Fields common to both the root and pool record shapes -- the two
        writers diverge only on ``provenance`` (root-only, #161)."""
        return {
            "track_id": track_id,
            "embedding": [float(x) for x in embedding],
            "tags": {label: float(score) for label, score in tags.items()},
            "model_version": model_version,
            "input_rms": input_rms,
        }

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> Path:
        """First-write-wins: claim ``path`` atomically via ``O_CREAT |
        O_EXCL``; a losing writer's payload is discarded, winner's path
        returned either way (#126 AC2, shared by root and pool writes)."""
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return path
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2))
        return path

    def list_audio_analyses(self) -> list[AudioAnalysisRecord]:
        """Read every persisted per-track embedding+tags record back (#125
        AC1) — the population :func:`music_intel_mcp.timbre.derive_timbre_clusters`
        clusters over. Missing dir -> empty list (honest-empty: no capture run
        yet), mirroring ``load_history``'s missing-file idiom. Sorted by
        ``track_id`` (not filename — sanitization can reorder them) for
        deterministic downstream clustering input."""
        if not self.audio_analysis_dir.exists():
            return []
        records: list[AudioAnalysisRecord] = []
        for path in self.audio_analysis_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                AudioAnalysisRecord(
                    track_id=payload["track_id"],
                    embedding=[float(x) for x in payload["embedding"]],
                    tags={k: float(v) for k, v in payload.get("tags", {}).items()},
                    provenance=payload.get("provenance"),
                    model_version=payload.get("model_version"),
                    input_rms=payload.get("input_rms"),
                )
            )
        records.sort(key=lambda r: r.track_id)
        return records

    def list_pool_audio_analyses(self) -> list[AudioAnalysisRecord]:
        """Read every persisted record from the node-level pool (#161), for
        #162's pool ∩ participant-history timbre derivation. Mirrors
        ``list_audio_analyses`` but reads ``pool_audio_analysis_dir`` instead
        of the participant root. No pool configured, or pool dir not yet
        created -> empty list (honest-empty), same idiom as the root reader."""
        pool_dir = self.pool_audio_analysis_dir
        if pool_dir is None or not pool_dir.exists():
            return []
        records: list[AudioAnalysisRecord] = []
        for path in pool_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                AudioAnalysisRecord(
                    track_id=payload["track_id"],
                    embedding=[float(x) for x in payload["embedding"]],
                    tags={k: float(v) for k, v in payload.get("tags", {}).items()},
                    provenance=payload.get("provenance"),
                    model_version=payload.get("model_version"),
                    input_rms=payload.get("input_rms"),
                )
            )
        records.sort(key=lambda r: r.track_id)
        return records

    # --- raw fingerprint sidecar (#140 AC1) -------------------------------- #
    #
    # Raw uint32 chromaprint arrays (evidence for offline near-duplicate
    # comparison, never an identity key) live in their own sidecar directory
    # beside wherever the corresponding audio_analysis record actually landed
    # (root or pool) -- the #161 AudioAnalysisRecord/pool-record schema is
    # deliberately left untouched (CONTEXT.md "Post-CRITIC refinements").

    @property
    def fingerprints_dir(self) -> Path:
        return self.root / "fingerprints"

    def fingerprint_path(self, track_id: str) -> Path:
        return self.fingerprints_dir / f"{self._safe_name(track_id)}.json"

    @property
    def pool_fingerprints_dir(self) -> Path | None:
        return self.pool_root / "fingerprints" if self.pool_root is not None else None

    def pool_fingerprint_path(self, track_id: str) -> Path | None:
        pool_dir = self.pool_fingerprints_dir
        return pool_dir / f"{self._safe_name(track_id)}.json" if pool_dir is not None else None

    def write_fingerprint(
        self, *, track_id: str, fingerprint: list[int], duration_s: float
    ) -> Path:
        """Write one raw chromaprint array sidecar. Mirrors
        ``write_audio_analysis``'s pool-exclusive-write-when-configured and
        first-write-wins (``_atomic_write``) semantics."""
        payload = {
            "track_id": track_id,
            "fingerprint": [int(x) for x in fingerprint],
            "duration_s": float(duration_s),
        }
        pool_path = self.pool_fingerprint_path(track_id)
        if pool_path is not None:
            pool_path.parent.mkdir(parents=True, exist_ok=True)
            return self._atomic_write(pool_path, payload)

        self.fingerprints_dir.mkdir(parents=True, exist_ok=True)
        return self._atomic_write(self.fingerprint_path(track_id), payload)

    # --- #140 AC4/AC5: near-dup aliases ------------------------------------ #

    @property
    def aliases_path(self) -> Path:
        return self.root / "aliases.jsonl"

    @property
    def pool_aliases_path(self) -> Path | None:
        return self.pool_root / "aliases.jsonl" if self.pool_root is not None else None

    def resolve_track_key(self, track_id: str) -> str:
        """#140 AC5, precedence inverted by #170 AC7: follow recorded aliases
        to the winner key, participant-root aliases taking precedence over
        pool aliases at each hop."""
        pool_aliases = load_aliases(self.pool_aliases_path) if self.pool_aliases_path else {}
        root_aliases = load_aliases(self.aliases_path)
        return resolve_key(track_id, pool_aliases=pool_aliases, root_aliases=root_aliases)

    def read_fingerprint(self, track_id: str) -> list[int] | None:
        """Pool-first-then-root read, mirroring ``has_audio_analysis``.
        Missing sidecar -> ``None`` (honest-empty: fingerprinting is
        best-effort and may have been skipped, e.g. ``fpcalc`` unavailable)."""
        pool_path = self.pool_fingerprint_path(track_id)
        if pool_path is not None and pool_path.exists():
            return json.loads(pool_path.read_text(encoding="utf-8"))["fingerprint"]
        path = self.fingerprint_path(track_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["fingerprint"]
        return None

    # --- automated playback consent (#128 AC1/AC3) ------------------------ #

    @property
    def automated_playback_consent_path(self) -> Path:
        return self.root / "automated_playback_consent.json"

    def has_automated_playback_consent(self) -> bool:
        """#128 AC1: off by default. Deliberately a separate file from any
        env-var opt-in (e.g. #127's ``MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED``)
        — this drives a real playback session, not just a queue, so it needs
        its own, separately-recorded consent action.

        #165 AC3: a pre-#165 file recorded only ``granted_at`` -- no grantor,
        no scope. Reading that as valid consent would grant an authorization
        nobody actually recorded, so it is rejected outright rather than
        silently treated as either granted or not-granted."""
        if not self.automated_playback_consent_path.exists():
            return False
        payload = json.loads(self.automated_playback_consent_path.read_text(encoding="utf-8"))
        if "grantor" not in payload or "scope" not in payload:
            raise ConsentFormatError(
                f"{self.automated_playback_consent_path} is in the old consent "
                "format (missing grantor/scope) -- revoke it and re-grant with "
                "`automated-playback-consent --grant --grantor ... --scope ...`"
            )
        return True

    def grant_automated_playback_consent(
        self, *, grantor: str, granted_at: str, scope: str
    ) -> Path:
        self.automated_playback_consent_path.parent.mkdir(parents=True, exist_ok=True)
        self.automated_playback_consent_path.write_text(
            json.dumps({"grantor": grantor, "timestamp": granted_at, "scope": scope}),
            encoding="utf-8",
        )
        return self.automated_playback_consent_path

    def revoke_automated_playback_consent(self) -> None:
        """#128 AC3: revocable at any time. The automated-playback loop polls
        this every few seconds, so deleting the file stops an in-progress
        session within one poll interval, mid-track if needed."""
        self.automated_playback_consent_path.unlink(missing_ok=True)
