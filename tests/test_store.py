"""Per-user store: history load + profile snapshot round-trip."""

from __future__ import annotations

import shutil

from music_intel_mcp.analyzer import analyze
from music_intel_mcp.store import UserStore


def test_load_history_parses_fixture(tmp_path, history_sample_path):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    store = UserStore(root=tmp_path)
    events = store.load_history()
    assert len(events) == 5
    assert events[0].track.spotify_id == "spotify:track:AAA"
    assert events[0].context.skipped is False
    assert events[2].context is None  # lastfm scrobble, no play-context


def test_load_history_missing_file_is_empty(tmp_path):
    assert UserStore(root=tmp_path).load_history() == []


def test_profile_snapshot_round_trips(tmp_path, history_sample_path):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    store = UserStore(root=tmp_path)
    events = store.load_history()
    profile = analyze(events, user_id="petr")

    path = store.write_profile(profile)
    assert path.exists()
    # write -> read -> re-validate is lossless
    reloaded = store.read_profile(path)
    assert reloaded == profile
    # latest_profile finds it
    assert store.latest_profile() == profile


def test_snapshot_filename_is_filesystem_safe(tmp_path):
    store = UserStore(root=tmp_path)
    profile = analyze([], user_id="petr")
    path = store.write_profile(profile)
    # snapshot_id has '/' and ':' which are illegal on Windows; filename is sanitized
    assert "/" not in path.name
    assert ":" not in path.name
    assert path.suffix == ".json"


def test_env_var_sets_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_INTEL_DATA_DIR", str(tmp_path / "envdata"))
    store = UserStore()
    assert store.root == tmp_path / "envdata"
