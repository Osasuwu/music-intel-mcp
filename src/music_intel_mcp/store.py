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


DEFAULT_DATA_DIR = "data"
_DATA_DIR_ENV = "MUSIC_INTEL_DATA_DIR"

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def resolve_data_root(root: str | Path | None) -> Path:
    """Resolve the data root: explicit arg > ``MUSIC_INTEL_DATA_DIR`` > default.
    Shared by the per-user store and the (local) shared-metadata cache."""
    if root is not None:
        return Path(root)
    return Path(os.environ.get(_DATA_DIR_ENV, DEFAULT_DATA_DIR))


class UserStore:
    """Read history, read/write RootProfile snapshots for one user."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = resolve_data_root(root)

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

    def has_audio_analysis(self, track_id: str) -> bool:
        """#126 AC1/AC4: dedup check the live pipeline runs via the identity
        waterfall's resolved ``track_id`` before paying for inference."""
        return self.audio_analysis_path(track_id).exists()

    def write_audio_analysis(
        self,
        *,
        track_id: str,
        embedding: Any,
        tags: dict[str, float],
        provenance: Any | None = None,
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
        silently discarded and the winner's path is returned either way."""
        self.audio_analysis_dir.mkdir(parents=True, exist_ok=True)
        path = self.audio_analysis_path(track_id)
        if provenance is None:
            provenance_payload = None
        elif hasattr(provenance, "model_dump"):
            provenance_payload = provenance.model_dump()
        else:
            provenance_payload = dict(provenance)
        payload = {
            "track_id": track_id,
            "embedding": [float(x) for x in embedding],
            "tags": {label: float(score) for label, score in tags.items()},
            "provenance": provenance_payload,
        }
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
                )
            )
        records.sort(key=lambda r: r.track_id)
        return records

    # --- automated playback consent (#128 AC1/AC3) ------------------------ #

    @property
    def automated_playback_consent_path(self) -> Path:
        return self.root / "automated_playback_consent.json"

    def has_automated_playback_consent(self) -> bool:
        """#128 AC1: off by default. Deliberately a separate file from any
        env-var opt-in (e.g. #127's ``MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED``)
        — this drives a real playback session, not just a queue, so it needs
        its own, separately-recorded consent action."""
        return self.automated_playback_consent_path.exists()

    def grant_automated_playback_consent(self, *, granted_at: str) -> Path:
        self.automated_playback_consent_path.parent.mkdir(parents=True, exist_ok=True)
        self.automated_playback_consent_path.write_text(
            json.dumps({"granted_at": granted_at}), encoding="utf-8"
        )
        return self.automated_playback_consent_path

    def revoke_automated_playback_consent(self) -> None:
        """#128 AC3: revocable at any time. The automated-playback loop polls
        this every few seconds, so deleting the file stops an in-progress
        session within one poll interval, mid-track if needed."""
        self.automated_playback_consent_path.unlink(missing_ok=True)
