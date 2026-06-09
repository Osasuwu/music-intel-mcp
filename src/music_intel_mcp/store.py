"""Per-user store — local plain files, no DB (decision f7a9fcbd).

Layout under the data root (default ``data/``, overridable via the
``MUSIC_INTEL_DATA_DIR`` env var or the ``UserStore(root=...)`` argument):

- ``history.jsonl`` — append-only listening events, one JSON object per line.
- ``profiles/<snapshot>.json`` — RootProfile time-series snapshots.

Personal data lives here and *only* here — never to the shared metadata store
(history-never-leaves-the-machine invariant).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .models import ListenEvent, RootProfile

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
