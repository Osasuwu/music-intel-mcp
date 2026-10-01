"""``music-intel-desktop`` --data-dir (#165 AC1): the tray app must open a
per-participant UserStore rooted at the flag's value instead of the
process-cwd default, so a phone-only participant's transient per-participant
root gets its own history/analyses/token/consent.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import music_intel_mcp.desktop_app as desktop_app
from music_intel_mcp.store import UserStore


class _FakeIcon:
    """Stands in for ``pystray.Icon``: ``run()`` returns immediately instead
    of blocking on a real system-tray event loop."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def run(self) -> None:
        return


@pytest.fixture
def fake_tray(monkeypatch):
    """desktop_app.main() imports pystray and draws the icon with Pillow, both
    from the Windows-only `desktop` extra that CI's `dev` install leaves out.
    Installing pystray would not help on a Linux runner either: with no display
    its import raises Xlib's DisplayNameError (#229). The tests only need main()
    to reach the icon, so a fake module stands in for pystray and the icon
    image is stubbed."""
    fake_pystray = types.ModuleType("pystray")
    fake_pystray.Icon = _FakeIcon
    fake_pystray.Menu = lambda *items: items
    fake_pystray.MenuItem = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(sys.modules, "pystray", fake_pystray)
    monkeypatch.setattr(desktop_app, "_build_icon_image", lambda color: None)


def test_data_dir_flag_threads_to_userstore_root(tmp_path, monkeypatch, fake_tray):
    captured: dict[str, UserStore] = {}

    def fake_run_loop(stop_event: threading.Event, status, store: UserStore) -> None:
        captured["store"] = store

    monkeypatch.setattr(desktop_app, "_run_loop", fake_run_loop)

    rc = desktop_app.main(["--data-dir", str(tmp_path)])

    assert rc == 0
    assert captured["store"].root == tmp_path


def test_no_data_dir_flag_defaults_to_resolve_data_root(monkeypatch, fake_tray):
    captured: dict[str, UserStore] = {}

    def fake_run_loop(stop_event: threading.Event, status, store: UserStore) -> None:
        captured["store"] = store

    monkeypatch.setattr(desktop_app, "_run_loop", fake_run_loop)
    monkeypatch.delenv("MUSIC_INTEL_DATA_DIR", raising=False)

    rc = desktop_app.main([])

    assert rc == 0
    assert captured["store"].root == UserStore().root
