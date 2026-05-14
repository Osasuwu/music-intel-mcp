"""Smoke test — package imports and version is exposed."""

import music_intel_mcp


def test_package_imports():
    assert music_intel_mcp.__version__


def test_version_is_semver_ish():
    parts = music_intel_mcp.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
