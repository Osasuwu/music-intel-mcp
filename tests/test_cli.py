"""CLI `analyze` + `resolve` + enrichment-wiring entrypoints."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import music_intel_mcp.cli as cli
from music_intel_mcp.audio import AcousticBrainzDump
from music_intel_mcp.cli import build_parser, main, plan_enrichment
from music_intel_mcp.models import ListenEvent, TrackRef
from music_intel_mcp.scene import InMemoryTagSource, LastfmTagSource
from music_intel_mcp.shared_store import InMemorySharedStore, SupabaseSharedStore, TrackTag
from music_intel_mcp.store import UserStore

FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index.tsv"


def _analyze_args(*extra: str):
    """Parse an ``analyze`` argv through the real parser (so the flags are
    exercised too) and return the resulting namespace."""
    return build_parser().parse_args(["analyze", "--user-id", "petr", *extra])


def test_cli_analyze_writes_snapshot(tmp_path, history_sample_path, capsys):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "snapshot:" in out
    assert "events=5" in out
    assert "unique_tracks=3" in out
    assert "roots=0" in out  # honest-empty

    # a valid snapshot landed in the store and re-validates
    profile = UserStore(root=tmp_path).latest_profile()
    assert profile is not None
    assert profile.user_id == "petr"
    assert profile.roots == []


def test_cli_analyze_empty_history(tmp_path, capsys):
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "events=0" in out
    assert "roots=0" in out


def test_cli_analyze_prints_per_category_coverage(tmp_path, history_sample_path, capsys):
    """The honest-empty default prints coverage with no enrichment run: audio and
    scene are 0 (no source), temporal is 1.0 (every event carries a timestamp)."""
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path)])
    assert rc == 0
    assert "coverage: audio=0.00 scene=0.00 temporal=1.00" in capsys.readouterr().out


def test_cli_resolve_reports_coverage(tmp_path, history_sample_path, capsys):
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["resolve", "--data-dir", str(tmp_path), "--mb-index", str(FIXTURE_INDEX)])
    assert rc == 0

    out = capsys.readouterr().out
    # 3 unique tracks: spotify AAA (no source -> spotify), isrc USABC (-> mbid
    # via fixture), mbid 1111 (passthrough). 2 reach MBID.
    assert "resolved 2/3 unique tracks to MBID" in out
    assert "unresolved (flagged, not dropped): 1" in out

    # identities are cached for re-runs
    assert (tmp_path / "identity").is_dir()


# --------------------------------------------------------------------------- #
# plan_enrichment — flag/env matrix (pure, fully offline)
# --------------------------------------------------------------------------- #


def test_plan_enrichment_no_flags_is_honest_empty():
    store, audio, tag, errors = plan_enrichment(_analyze_args(), {})
    assert (store, audio, tag, errors) == (None, None, None, [])


def test_plan_enrichment_scene_requires_lastfm_key():
    args = _analyze_args("--with-scene", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {})  # no key present
    assert store is None and audio is None and tag is None
    assert any("LASTFM_API_KEY" in e for e in errors)


def test_plan_enrichment_scene_memory_store_ok():
    args = _analyze_args("--with-scene", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {"LASTFM_API_KEY": "present"})
    assert errors == []
    assert isinstance(store, InMemorySharedStore)
    assert audio is None
    assert isinstance(tag, LastfmTagSource)


def test_plan_enrichment_audio_constructs_dump_with_ab_index(tmp_path):
    args = _analyze_args(
        "--with-audio", "--shared-store", "memory", "--ab-index", str(tmp_path / "x.jsonl")
    )
    store, audio, tag, errors = plan_enrichment(args, {})
    assert errors == []
    assert isinstance(store, InMemorySharedStore)
    assert isinstance(audio, AcousticBrainzDump)
    assert tag is None


def test_plan_enrichment_supabase_requires_creds():
    # default --shared-store is supabase
    store, audio, tag, errors = plan_enrichment(_analyze_args("--with-audio"), {})
    assert store is None
    assert any("SUPABASE_URL" in e for e in errors)


def test_plan_enrichment_supabase_with_creds_ok():
    args = _analyze_args("--with-audio")
    env = {"SUPABASE_URL": "u", "SUPABASE_KEY": "k"}
    store, audio, tag, errors = plan_enrichment(args, env)
    assert errors == []
    assert isinstance(store, SupabaseSharedStore)
    assert isinstance(audio, AcousticBrainzDump)


def test_plan_enrichment_reports_every_missing_credential():
    # scene + default supabase, empty env -> both the supabase and lastfm checks fire
    store, audio, tag, errors = plan_enrichment(_analyze_args("--with-scene"), {})
    assert store is None
    assert len(errors) == 2


# --------------------------------------------------------------------------- #
# analyze command — fail-fast credential gates + end-to-end enrichment wiring
# --------------------------------------------------------------------------- #


def test_cli_analyze_scene_without_key_fails_fast(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(
        [
            "analyze",
            "--user-id",
            "petr",
            "--data-dir",
            str(tmp_path),
            "--with-scene",
            "--shared-store",
            "memory",
        ]
    )
    assert rc == 2
    assert "LASTFM_API_KEY" in capsys.readouterr().out
    # fail-fast: nothing was analysed or written
    assert UserStore(root=tmp_path).latest_profile() is None


def test_cli_analyze_supabase_without_creds_fails_fast(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(["analyze", "--user-id", "petr", "--data-dir", str(tmp_path), "--with-audio"])
    assert rc == 2
    assert "SUPABASE_URL" in capsys.readouterr().out
    assert UserStore(root=tmp_path).latest_profile() is None


def test_cli_analyze_with_scene_runs_enrichment(tmp_path, capsys, monkeypatch):
    """End-to-end through the CLI with the production Last.fm source swapped for an
    offline in-memory one: enrichment runs, tags land, scene coverage reflects it.
    No root validates under default (uncalibrated) thresholds — that is correct
    until #66 calibration — so the proof of wiring is the coverage, not a root."""
    tracks = [("metal-band", "Track A"), ("jazz-band", "Track B")]
    mapping = {(a, n): [TrackTag(tag="scene-tag", weight=1.0, source="lastfm")] for a, n in tracks}
    events = [
        ListenEvent(
            track=TrackRef(name=n, artist=a, mbid=f"mbid-{n}"),
            played_at=datetime(2025, 1, 1, tzinfo=UTC),
            source="lastfm",
        )
        for a, n in tracks
    ]
    (tmp_path / "history.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in events), encoding="utf-8"
    )
    monkeypatch.setenv("LASTFM_API_KEY", "present")
    monkeypatch.setattr(cli, "LastfmTagSource", lambda: InMemoryTagSource(mapping))

    rc = main(
        [
            "analyze",
            "--user-id",
            "petr",
            "--data-dir",
            str(tmp_path),
            "--with-scene",
            "--shared-store",
            "memory",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "scene=1.00" in out  # every track tagged -> full scene coverage
    profile = UserStore(root=tmp_path).latest_profile()
    assert profile is not None
    assert profile.roots == []  # uncalibrated thresholds: honest-empty roots


def test_cli_analyze_with_audio_no_dump_notes_zero_coverage(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    """`--with-audio` with no dump installed is honest low coverage, not an error:
    rc 0, a snapshot, audio=0.00, and a diagnostic note pointing at the dump env."""
    monkeypatch.delenv("ACOUSTICBRAINZ_FEATURES_INDEX", raising=False)
    monkeypatch.delenv("ACOUSTICBRAINZ_DUMP_DIR", raising=False)
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")
    rc = main(
        [
            "analyze",
            "--user-id",
            "petr",
            "--data-dir",
            str(tmp_path),
            "--with-audio",
            "--shared-store",
            "memory",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "audio=0.00" in out
    assert "note: audio coverage 0" in out
