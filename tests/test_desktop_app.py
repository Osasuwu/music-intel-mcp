"""``music-intel-desktop`` --data-dir (#165 AC1): the tray app must open a
per-participant UserStore rooted at the flag's value instead of the
process-cwd default, so a phone-only participant's transient per-participant
root gets its own history/analyses/token/consent.
"""

from __future__ import annotations

import threading

import pytest

import music_intel_mcp.desktop_app as desktop_app
from music_intel_mcp.store import UserStore

# desktop_app.main() imports pystray internally; it ships in the Windows-only
# `desktop` extra (pyproject.toml), deliberately excluded from CI's `dev`
# install the same way live-capture's winsdk/onnxruntime are. Skip rather than
# fail where it isn't installed.
pytest.importorskip("pystray")


class _FakeIcon:
    """Stands in for ``pystray.Icon``: ``run()`` returns immediately instead
    of blocking on a real system-tray event loop."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self) -> None:
        return


def test_data_dir_flag_threads_to_userstore_root(tmp_path, monkeypatch):
    captured: dict[str, UserStore] = {}

    def fake_run_loop(stop_event: threading.Event, status, store: UserStore) -> None:
        captured["store"] = store

    monkeypatch.setattr(desktop_app, "_run_loop", fake_run_loop)
    monkeypatch.setattr("pystray.Icon", _FakeIcon)

    rc = desktop_app.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert captured["store"].root == tmp_path


def test_no_data_dir_flag_defaults_to_resolve_data_root(monkeypatch):
    captured: dict[str, UserStore] = {}

    def fake_run_loop(stop_event: threading.Event, status, store: UserStore) -> None:
        captured["store"] = store

    monkeypatch.setattr(desktop_app, "_run_loop", fake_run_loop)
    monkeypatch.setattr("pystray.Icon", _FakeIcon)
    monkeypatch.delenv("MUSIC_INTEL_DATA_DIR", raising=False)

    rc = desktop_app.main([])

    assert rc == 0
    assert captured["store"].root == UserStore().root
