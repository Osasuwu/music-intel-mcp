"""Per-user store: history load + profile snapshot round-trip."""

from __future__ import annotations

import json
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


def test_replace_history_overwrites_not_appends(tmp_path):
    from datetime import UTC, datetime

    from music_intel_mcp.models import ListenEvent, TrackRef

    store = UserStore(root=tmp_path)
    e1 = ListenEvent(
        track=TrackRef(spotify_id="A", name="a", artist="x"),
        played_at=datetime(2025, 1, 1, tzinfo=UTC),
        source="ifttt",
    )
    e2 = ListenEvent(
        track=TrackRef(spotify_id="B", name="b", artist="y"),
        played_at=datetime(2025, 2, 1, tzinfo=UTC),
        source="ifttt",
    )
    store.append_events([e1])
    store.replace_history([e2])  # replaces, does not append
    reloaded = store.load_history()
    assert reloaded == [e2]


# AC5 (#124): live-capture inference output is written to the LOCAL store
# only. UserStore never talks to Supabase/SharedStore, so this write path is
# structurally local-only, not just conventionally so.
def test_write_audio_analysis_writes_under_local_root(tmp_path):
    import numpy as np

    store = UserStore(root=tmp_path)
    path = store.write_audio_analysis(
        track_id="mbid-around-the-world",
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        tags={"genre---electronic": 0.9},
    )

    assert path.exists()
    assert path.is_relative_to(tmp_path)
    payload = path.read_text(encoding="utf-8")
    assert "genre---electronic" in payload
    assert "0.1" in payload


def test_write_audio_analysis_sanitizes_track_id(tmp_path):
    store = UserStore(root=tmp_path)
    path = store.write_audio_analysis(track_id="spotify:track:AAA/BBB", embedding=[0.1], tags={})
    assert path.parent == store.root / "audio_analysis"
    assert "/" not in path.name and "\\" not in path.name


# AC5 (#139): every live-capture embedding is paired with a provenance sidecar
# (raw title/artist, source app id, capture timestamp, chromaprint fingerprint)
# so a future identity-strategy change is a re-mapping job over sidecars, never
# a re-listen.
def test_write_audio_analysis_persists_provenance_sidecar(tmp_path):
    from music_intel_mcp.live_identity import ProvenanceSidecar

    store = UserStore(root=tmp_path)
    provenance = ProvenanceSidecar(
        raw_title="Song (Official Video)",
        raw_artist="Artist",
        app_id="Spotify.exe",
        captured_at="2026-01-01T00:00:00+00:00",
        chromaprint_fingerprint="fp1",
    )
    path = store.write_audio_analysis(
        track_id="mbid-1", embedding=[0.1], tags={}, provenance=provenance
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provenance"]["raw_title"] == "Song (Official Video)"
    assert payload["provenance"]["chromaprint_fingerprint"] == "fp1"


def test_write_audio_analysis_provenance_optional(tmp_path):
    store = UserStore(root=tmp_path)
    path = store.write_audio_analysis(track_id="mbid-1", embedding=[0.1], tags={})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provenance"] is None
