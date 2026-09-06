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


# #125 AC1: a reader for the local store so per-user clustering can consume
# every persisted per-track embedding, not just write them.
def test_list_audio_analyses_reads_back_every_persisted_record(tmp_path):
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="b", embedding=[0.4, 0.5], tags={"genre---rock": 0.8})
    store.write_audio_analysis(track_id="a", embedding=[0.1, 0.2], tags={"genre---jazz": 0.6})

    records = store.list_audio_analyses()

    assert [r.track_id for r in records] == ["a", "b"]  # sorted, deterministic
    assert records[0].embedding == [0.1, 0.2]
    assert records[0].tags == {"genre---jazz": 0.6}


def test_list_audio_analyses_missing_dir_is_empty(tmp_path):
    assert UserStore(root=tmp_path).list_audio_analyses() == []


def test_list_audio_analyses_sorts_by_track_id_not_filename(tmp_path):
    # "spotify:track:Z" sanitizes to "spotify_track_Z" (filename order: Z after A),
    # but "b" (raw track_id) sorts before "spotify:track:Z" only by track_id,
    # not by the sanitized filename — glob's filename order would put "b.json"
    # before "spotify_track_Z.json" too, so instead pick ids where sanitization
    # flips relative order: "A/Z" -> "A_Z" vs "A0" stays "A0" (raw '/' < '0' in
    # ASCII, but sanitized '_' > '0'), so filename-order and track_id-order disagree.
    store = UserStore(root=tmp_path)
    store.write_audio_analysis(track_id="A0", embedding=[0.1], tags={})
    store.write_audio_analysis(track_id="A/Z", embedding=[0.2], tags={})

    records = store.list_audio_analyses()

    assert [r.track_id for r in records] == ["A/Z", "A0"]  # raw track_id order


# #126 AC2: concurrent-duplicate race — pinned first-write-wins, no averaging,
# no data loss. A second writer for the same track_id must not clobber the
# first writer's payload, and must not raise.
def test_write_audio_analysis_first_write_wins_on_duplicate_track_id(tmp_path):
    store = UserStore(root=tmp_path)
    first_path = store.write_audio_analysis(
        track_id="mbid-dup", embedding=[0.1], tags={"genre---rock": 1.0}
    )
    second_path = store.write_audio_analysis(
        track_id="mbid-dup", embedding=[0.9], tags={"genre---jazz": 1.0}
    )

    assert first_path == second_path
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["tags"] == {"genre---rock": 1.0}
    assert payload["embedding"] == [0.1]


def test_write_audio_analysis_first_write_wins_under_thread_concurrency(tmp_path):
    import threading

    store = UserStore(root=tmp_path)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _write(tag_value: float) -> None:
        try:
            barrier.wait()
            store.write_audio_analysis(
                track_id="mbid-race", embedding=[tag_value], tags={"genre---x": tag_value}
            )
        except Exception as exc:  # pragma: no cover - failure path asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(v,)) for v in (1.0, 2.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    path = store.audio_analysis_path("mbid-race")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["embedding"] in ([1.0], [2.0])  # exactly one writer's payload, never a mix


# #158 AC4: a one-shot migration renames bare-key audio-analysis files
# (written before the canonical_track_id prefix contract existed) to their
# prefixed form. Bare keys carry no type tag of their own, so the migration
# classifies each by format (MBID uuid / ISRC / 22-char Spotify id / else a
# name-key rebuilt from the provenance sidecar's raw title+artist).
def test_migrate_audio_analysis_keys_renames_bare_mbid_key(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    bare = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    old_path = store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})

    report = migrate_audio_analysis_keys(store)

    assert report.migrated == [(bare, f"mbid:{bare}")]
    assert not old_path.exists()
    new_path = store.audio_analysis_path(f"mbid:{bare}")
    assert new_path.exists()
    assert json.loads(new_path.read_text(encoding="utf-8"))["track_id"] == f"mbid:{bare}"


def test_migrate_audio_analysis_keys_renames_bare_isrc_key(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    bare = "USRC17607839"
    store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})

    report = migrate_audio_analysis_keys(store)

    assert report.migrated == [(bare, f"isrc:{bare}")]
    assert store.audio_analysis_path(f"isrc:{bare}").exists()


def test_migrate_audio_analysis_keys_renames_bare_spotify_id_key(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    bare = "AbCdEfGhIj1234567890AB"  # 22-char base62 spotify id shape
    assert len(bare) == 22
    store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})

    report = migrate_audio_analysis_keys(store)

    assert report.migrated == [(bare, f"spotify:{bare}")]
    assert store.audio_analysis_path(f"spotify:{bare}").exists()


def test_migrate_audio_analysis_keys_renames_bare_name_key_using_provenance(tmp_path):
    from music_intel_mcp.live_identity import ProvenanceSidecar
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    old_name_key = "song title\x1fthe artist"  # pre-#158 name_key normalization
    store.write_audio_analysis(
        track_id=old_name_key,
        embedding=[0.1],
        tags={},
        provenance=ProvenanceSidecar(
            raw_title="Song Title (Official Video)",
            raw_artist="The Artist",
            app_id="Spotify.exe",
            captured_at="2026-01-01T00:00:00+00:00",
        ),
    )

    report = migrate_audio_analysis_keys(store)

    # normalize_track_name() strips "(Official Video)" noise -- the migrated
    # key must match what a fresh live capture of the same track would
    # compute today (live_pipeline.py's #158 AC1 fix), not a plain casefold
    # of the raw provenance title (which would reintroduce the noise).
    expected_new_id = "name:song title\x1fthe artist"
    assert report.migrated == [(old_name_key, expected_new_id)]
    assert store.audio_analysis_path(expected_new_id).exists()


def test_migrate_audio_analysis_keys_is_idempotent(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    bare = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={})

    first = migrate_audio_analysis_keys(store)
    second = migrate_audio_analysis_keys(store)

    assert first.migrated == [(bare, f"mbid:{bare}")]
    assert second.migrated == []
    assert second.conflicts == []


def test_migrate_audio_analysis_keys_reports_conflicts_without_overwriting(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    bare = "5b11f4ce-a62d-471e-81fc-a69a8278c7da"
    old_path = store.write_audio_analysis(track_id=bare, embedding=[0.1], tags={"old": 1.0})
    # A post-#158 write already claimed the canonical prefixed key.
    store.write_audio_analysis(track_id=f"mbid:{bare}", embedding=[0.9], tags={"new": 1.0})

    report = migrate_audio_analysis_keys(store)

    assert report.migrated == []
    assert report.conflicts == [(bare, f"mbid:{bare}")]
    # Neither file was touched -- old bare file preserved, new file untouched.
    assert old_path.exists()
    assert json.loads(old_path.read_text(encoding="utf-8"))["tags"] == {"old": 1.0}
    new_path = store.audio_analysis_path(f"mbid:{bare}")
    assert json.loads(new_path.read_text(encoding="utf-8"))["tags"] == {"new": 1.0}


def test_migrate_audio_analysis_keys_missing_dir_returns_empty_report(tmp_path):
    from music_intel_mcp.store import migrate_audio_analysis_keys

    store = UserStore(root=tmp_path)
    report = migrate_audio_analysis_keys(store)
    assert report.migrated == []
    assert report.conflicts == []


# #161 AC1: pool records hold exactly key + embedding + tags + model_version --
# no provenance field at all (not even null), since a pool is shared across
# participants and provenance (raw_title/app_id/captured_at) would make it a
# re-identifiable copy of one participant's play list.
def test_write_audio_analysis_with_pool_writes_exact_schema_no_provenance(tmp_path):
    store = UserStore(root=tmp_path / "participant", pool_root=tmp_path / "pool")
    path = store.write_audio_analysis(
        track_id="mbid-1",
        embedding=[0.1, 0.2],
        tags={"genre---rock": 0.9},
        model_version="discogs-effnet-bsdynamic-1",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"track_id", "embedding", "tags", "model_version"}
    assert payload["model_version"] == "discogs-effnet-bsdynamic-1"


# #161 AC2: with a pool configured, new analyses land in the pool -- not the
# participant root -- and has_audio_analysis reports a hit off the pool write.
def test_write_audio_analysis_with_pool_never_touches_participant_root(tmp_path):
    participant_root = tmp_path / "participant"
    pool_root = tmp_path / "pool"
    store = UserStore(root=participant_root, pool_root=pool_root)

    assert store.has_audio_analysis("mbid-1") is False
    store.write_audio_analysis(track_id="mbid-1", embedding=[0.1], tags={})

    assert store.has_audio_analysis("mbid-1") is True
    assert (pool_root / "audio_analysis" / "mbid-1.json").exists()
    assert not (participant_root / "audio_analysis").exists()


# #161 AC2: pool-first lookup -- a track analyzed via one participant's store
# dedupes for a second participant's store sharing the same pool, even though
# it never touched that second participant's own (transient) root.
def test_has_audio_analysis_consults_pool_first_across_participants(tmp_path):
    pool_root = tmp_path / "pool"
    store_a = UserStore(root=tmp_path / "participant-a", pool_root=pool_root)
    store_b = UserStore(root=tmp_path / "participant-b", pool_root=pool_root)

    store_a.write_audio_analysis(track_id="mbid-shared", embedding=[0.1], tags={})

    assert store_b.has_audio_analysis("mbid-shared") is True


# #161 AC4: pool writes keep first-write-wins semantics -- same guarantee as
# root writes (#126 AC2), now shared via UserStore._atomic_write.
def test_write_audio_analysis_pool_first_write_wins_on_duplicate_track_id(tmp_path):
    store = UserStore(root=tmp_path / "participant", pool_root=tmp_path / "pool")
    first_path = store.write_audio_analysis(
        track_id="mbid-dup", embedding=[0.1], tags={"genre---rock": 1.0}
    )
    second_path = store.write_audio_analysis(
        track_id="mbid-dup", embedding=[0.9], tags={"genre---jazz": 1.0}
    )

    assert first_path == second_path
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["tags"] == {"genre---rock": 1.0}
    assert payload["embedding"] == [0.1]


# #161 AC4: pool file mtimes must not be used by any reader -- a pool record
# written long ago (simulated via os.utime) must still be reported as present.
def test_has_audio_analysis_pool_hit_ignores_file_mtime(tmp_path):
    import os
    import time

    store = UserStore(root=tmp_path / "participant", pool_root=tmp_path / "pool")
    path = store.write_audio_analysis(track_id="mbid-old", embedding=[0.1], tags={})
    ancient = time.time() - 60 * 60 * 24 * 365 * 5  # 5 years ago
    os.utime(path, (ancient, ancient))

    assert store.has_audio_analysis("mbid-old") is True


# #128 AC1: automated-playback mode is off by default; enabling it requires an
# explicit, separately-recorded consent action distinct from #127's
# MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED env-var opt-in.
def test_automated_playback_consent_is_off_by_default(tmp_path):
    assert UserStore(root=tmp_path).has_automated_playback_consent() is False


def test_grant_automated_playback_consent_persists_it(tmp_path):
    store = UserStore(root=tmp_path)
    path = store.grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    assert path.exists()
    assert store.has_automated_playback_consent() is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["grantor"] == "alice"
    assert payload["timestamp"] == "2026-01-01T00:00:00Z"
    assert payload["scope"] == "automated-playback"


# #128 AC3: consent is revocable at any time.
def test_revoke_automated_playback_consent_removes_it(tmp_path):
    store = UserStore(root=tmp_path)
    store.grant_automated_playback_consent(
        grantor="alice", granted_at="2026-01-01T00:00:00Z", scope="automated-playback"
    )

    store.revoke_automated_playback_consent()

    assert store.has_automated_playback_consent() is False


def test_revoke_automated_playback_consent_is_a_noop_when_never_granted(tmp_path):
    store = UserStore(root=tmp_path)
    store.revoke_automated_playback_consent()  # must not raise
    assert store.has_automated_playback_consent() is False


# #165 AC3: the pre-#165 bare {"granted_at": ...} shape carries no grantor or
# scope -- silently treating it as valid consent would let an old file grant
# a broader authorization than anyone actually recorded, so it must be
# rejected with a clear message rather than read as if still valid.
def test_old_format_consent_file_is_rejected_with_clear_message(tmp_path):
    from music_intel_mcp.store import ConsentFormatError

    store = UserStore(root=tmp_path)
    store.automated_playback_consent_path.parent.mkdir(parents=True, exist_ok=True)
    store.automated_playback_consent_path.write_text(
        json.dumps({"granted_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )

    try:
        store.has_automated_playback_consent()
        raise AssertionError("expected ConsentFormatError")
    except ConsentFormatError as exc:
        assert "old" in str(exc).lower() or "format" in str(exc).lower()


def test_automated_playback_consent_is_independent_of_backfill_playlist_opt_in(
    tmp_path, monkeypatch
):
    # "distinct from S6's playlist opt-in" (#128 issue body) -- setting the
    # unrelated #127 env flag must not itself grant automated-playback consent.
    monkeypatch.setenv("MUSIC_INTEL_BACKFILL_PLAYLIST_ENABLED", "true")
    assert UserStore(root=tmp_path).has_automated_playback_consent() is False
