"""CLI `analyze` + `resolve` + enrichment-wiring entrypoints."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

import music_intel_mcp.cli as cli
from music_intel_mcp.audio import AcousticBrainzDump
from music_intel_mcp.cli import (
    audio_identity_metrics,
    build_parser,
    main,
    plan_enrichment,
    plan_spotify_source,
)
from music_intel_mcp.models import ListenEvent, TrackRef
from music_intel_mcp.scene import (
    CompositeTagSource,
    DiscogsStyleSource,
    InMemoryTagSource,
    LastfmTagSource,
    MusicBrainzGenreSource,
)
from music_intel_mcp.shared_store import (
    InMemorySharedStore,
    LocalSharedStore,
    SupabaseSharedStore,
    TrackTag,
)
from music_intel_mcp.spotify_api import (
    SPOTIFY_TOKEN_URL,
    SPOTIFY_TRACKS_URL,
    SpotifyApiIsrcSource,
)
from music_intel_mcp.store import UserStore

FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index.tsv"
FIXTURE_AB = Path(__file__).parent / "fixtures" / "acousticbrainz_features_sample.jsonl"


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
# plan_spotify_source — flag/env gate (pure, fully offline)
# --------------------------------------------------------------------------- #


def test_plan_spotify_source_no_flag_is_none():
    source, errors = plan_spotify_source(False, {}, data_dir=None)
    assert (source, errors) == (None, [])


def test_plan_spotify_source_requires_both_credentials():
    # only one of the two present -> the missing one is reported, no source built
    source, errors = plan_spotify_source(True, {"SPOTIFY_CLIENT_ID": "x"}, data_dir=None)
    assert source is None
    assert any("SPOTIFY_CLIENT_SECRET" in e for e in errors)


def test_plan_spotify_source_with_creds_builds_source(tmp_path):
    env = {"SPOTIFY_CLIENT_ID": "x", "SPOTIFY_CLIENT_SECRET": "y"}
    source, errors = plan_spotify_source(True, env, data_dir=str(tmp_path))
    assert errors == []
    assert isinstance(source, SpotifyApiIsrcSource)


# --------------------------------------------------------------------------- #
# resolve --with-spotify — end-to-end bridge (network mocked with respx)
# --------------------------------------------------------------------------- #


def test_cli_resolve_with_spotify_without_creds_fails_fast(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")

    rc = main(
        ["resolve", "--data-dir", str(tmp_path), "--mb-index", str(FIXTURE_INDEX), "--with-spotify"]
    )
    assert rc == 2
    assert "SPOTIFY_CLIENT_ID" in capsys.readouterr().out


def test_cli_resolve_with_spotify_bridges_spotify_only_track(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    """The spotify-only track AAA stalls at the ``spotify`` rung without the live
    leg (2/3). With ``--with-spotify`` mocked to return an ISRC the fixture index
    maps to an MBID, AAA reaches MBID and coverage rises to 3/3."""
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "csecret")
    shutil.copy(history_sample_path, tmp_path / "history.jsonl")

    def _tracks(request: httpx.Request) -> httpx.Response:
        ids = request.url.params.get("ids", "").split(",")
        # AAA -> USABC1234567, which isrc_mbid_index.tsv maps to a real MBID.
        tracks = [
            {"id": tid, "external_ids": {"isrc": "USABC1234567"}} if tid == "AAA" else None
            for tid in ids
        ]
        return httpx.Response(200, json={"tracks": tracks})

    with respx.mock(assert_all_called=False) as router:
        router.post(SPOTIFY_TOKEN_URL).mock(
            return_value=httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        )
        router.get(SPOTIFY_TRACKS_URL).mock(side_effect=_tracks)
        rc = main(
            [
                "resolve",
                "--data-dir",
                str(tmp_path),
                "--mb-index",
                str(FIXTURE_INDEX),
                "--with-spotify",
            ]
        )

    assert rc == 0
    out = capsys.readouterr().out
    assert "spotify: prefetched 1 new spotify_id->ISRC lookups" in out
    assert "resolved 3/3 unique tracks to MBID" in out
    # the spotify_id->ISRC lookups are cached locally for re-runs (never re-queried)
    assert (tmp_path / "spotify_isrc_cache.jsonl").exists()


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


def test_plan_enrichment_default_store_is_local(tmp_path):
    # V0 default (decision 813de040): file-backed local store, no creds required.
    args = _analyze_args("--with-audio", "--shared-store-path", str(tmp_path / "sc.jsonl"))
    assert args.shared_store == "local"
    store, audio, tag, errors = plan_enrichment(args, {})  # empty env — no creds needed
    assert errors == []
    assert isinstance(store, LocalSharedStore)
    assert store.path == tmp_path / "sc.jsonl"
    assert isinstance(audio, AcousticBrainzDump)


def test_plan_enrichment_supabase_requires_creds():
    args = _analyze_args("--with-audio", "--shared-store", "supabase")
    store, audio, tag, errors = plan_enrichment(args, {})
    assert store is None
    assert any("SUPABASE_URL" in e for e in errors)


def test_plan_enrichment_supabase_with_creds_ok():
    args = _analyze_args("--with-audio", "--shared-store", "supabase")
    env = {"SUPABASE_URL": "u", "SUPABASE_KEY": "k"}
    store, audio, tag, errors = plan_enrichment(args, env)
    assert errors == []
    assert isinstance(store, SupabaseSharedStore)
    assert isinstance(audio, AcousticBrainzDump)


def test_plan_enrichment_reports_every_missing_credential():
    # scene + supabase, empty env -> both the supabase and lastfm checks fire
    args = _analyze_args("--with-scene", "--shared-store", "supabase")
    store, audio, tag, errors = plan_enrichment(args, {})
    assert store is None
    assert len(errors) == 2


def test_plan_enrichment_musicbrainz_genre_requires_app_contact():
    args = _analyze_args("--with-musicbrainz-genre", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {})
    assert store is None and audio is None and tag is None
    assert any("MUSICBRAINZ_APP_CONTACT" in e for e in errors)


def test_plan_enrichment_musicbrainz_genre_memory_store_ok():
    args = _analyze_args("--with-musicbrainz-genre", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {"MUSICBRAINZ_APP_CONTACT": "me@example.com"})
    assert errors == []
    assert isinstance(store, InMemorySharedStore)
    assert isinstance(tag, MusicBrainzGenreSource)


def test_plan_enrichment_discogs_requires_token():
    args = _analyze_args("--with-discogs", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {})
    assert store is None and audio is None and tag is None
    assert any("DISCOGS_TOKEN" in e for e in errors)


def test_plan_enrichment_discogs_memory_store_ok():
    args = _analyze_args("--with-discogs", "--shared-store", "memory")
    store, audio, tag, errors = plan_enrichment(args, {"DISCOGS_TOKEN": "tok"})
    assert errors == []
    assert isinstance(store, InMemorySharedStore)
    assert isinstance(tag, DiscogsStyleSource)


def test_plan_enrichment_combines_multiple_tag_sources_into_composite():
    args = _analyze_args(
        "--with-scene", "--with-musicbrainz-genre", "--with-discogs", "--shared-store", "memory"
    )
    store, audio, tag, errors = plan_enrichment(
        args,
        {
            "LASTFM_API_KEY": "present",
            "MUSICBRAINZ_APP_CONTACT": "me@example.com",
            "DISCOGS_TOKEN": "tok",
        },
    )
    assert errors == []
    assert isinstance(tag, CompositeTagSource)
    # order: Last.fm -> MusicBrainz genre -> Discogs, per the module docstring
    assert [type(s) for s in tag._sources] == [
        LastfmTagSource,
        MusicBrainzGenreSource,
        DiscogsStyleSource,
    ]


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
    rc = main(
        [
            "analyze",
            "--user-id",
            "petr",
            "--data-dir",
            str(tmp_path),
            "--with-audio",
            "--shared-store",
            "supabase",
        ]
    )
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


def test_cli_analyze_with_audio_bridges_isrc_to_mbid_and_enriches(
    tmp_path, history_sample_path, capsys, monkeypatch
):
    """End-to-end P9 bridge (#87 AC-C) through the CLI: `--with-audio --mb-index`
    resolves the ISRC-only history track to its MBID, and the AB-features fixture
    keyed by that MBID enriches it — so audio coverage is non-zero where the raw
    ingest identity carried no MBID. The AB fixture holds only the *resolved*
    MBID (not the history's raw ``mbid:1111`` track), so a passing coverage is
    proof the resolver ran, not the passthrough."""
    monkeypatch.delenv("MUSICBRAINZ_ISRC_INDEX", raising=False)
    monkeypatch.delenv("MUSICBRAINZ_DUMP_DIR", raising=False)
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
            "--mb-index",
            str(FIXTURE_INDEX),
            "--ab-index",
            str(FIXTURE_AB),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # 3 unique tracks; only the ISRC track resolves to an AB-covered MBID -> 1/3.
    assert "audio=0.33" in out
    assert "note: audio coverage 0" not in out
    # #87 AC-E honest measurement, visible on the run: 2 of 3 tracks carry an MBID
    # (the spotify-only AAA does not), and of those 2 the AB fixture covers 1.
    assert "mbid_coverage=0.67 (2/3)" in out
    assert "p(features|mbid)=0.50" in out
    assert "enriched=1 already_present=0 missing_features=1 no_mbid=1" in out
    # the resolver walked the tracks -> identities cached for re-runs
    assert (tmp_path / "identity").is_dir()


def test_audio_identity_metrics_derives_mbid_coverage_and_p_features():
    """The two #87 AC-E rates fall out of the enrichment buckets: MBID coverage =
    share carrying an MBID; P(features|MBID) = share of MBID-bearing tracks the
    features source covered (enriched or already carried)."""
    diag = {"enriched": 3, "already_present": 1, "missing_features": 2, "no_mbid": 4}
    m = audio_identity_metrics(diag)
    assert m["mbid_coverage"] == 6 / 10  # 4 of 10 have no MBID
    assert m["features_given_mbid"] == 4 / 6  # 4 of the 6 MBID-bearing have features


def test_audio_identity_metrics_empty_is_honest_zero():
    """No tracks / no MBIDs -> 0.0, never a divide-by-zero."""
    assert audio_identity_metrics({}) == {"mbid_coverage": 0.0, "features_given_mbid": 0.0}
    only_no_mbid = {"enriched": 0, "already_present": 0, "missing_features": 0, "no_mbid": 5}
    assert audio_identity_metrics(only_no_mbid)["features_given_mbid"] == 0.0


def test_cli_main_loads_dotenv_credentials(tmp_path, capsys, monkeypatch):
    """``main`` loads a gitignored ``.env`` so the documented unblock mechanism
    works: a ``LASTFM_API_KEY`` present only in ``.env`` (not the host env)
    clears the ``--with-scene`` fail-fast gate instead of exiting 2.

    This test exercises the *real* loader, re-pointing ``cli.load_dotenv`` back
    over the hermetic autouse stub."""
    from dotenv import load_dotenv as real_load_dotenv

    monkeypatch.setattr(cli, "load_dotenv", real_load_dotenv)
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)  # only .env provides it
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LASTFM_API_KEY=x\n", encoding="utf-8")
    # offline scene source so the gate-pass doesn't reach live Last.fm
    monkeypatch.setattr(cli, "LastfmTagSource", lambda: InMemoryTagSource({}))

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
    assert rc == 0  # gate cleared: the key crossed from .env into os.environ
    assert "LASTFM_API_KEY" not in capsys.readouterr().out  # no fail-fast error


# --- build-mb-index (#87 AC-B): raw dump -> ISRC->MBID TSV ----------------- #

FIXTURE_MB_ISRC = Path(__file__).parent / "fixtures" / "mbdump" / "isrc"
FIXTURE_MB_RECORDING = Path(__file__).parent / "fixtures" / "mbdump" / "recording"


def test_cli_build_mb_index_writes_consumable_tsv(tmp_path, capsys):
    from music_intel_mcp.identity import MusicBrainzIsrcIndex

    out = tmp_path / "isrc_to_mbid.tsv"
    rc = main(
        [
            "build-mb-index",
            "--isrc-dump",
            str(FIXTURE_MB_ISRC),
            "--recording-dump",
            str(FIXTURE_MB_RECORDING),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "wrote 3 ISRC->MBID pairs" in capsys.readouterr().out
    # the freshly built TSV is exactly what the resolver's index reads
    index = MusicBrainzIsrcIndex(path=out)
    assert index.lookup("USONE00000001") == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert len(index.lookup_all("USMULTI00001")) == 2


# --- build-artist-index (#102): raw dump -> artist-URI->MBID TSV ----------- #

FIXTURE_MB_URL = Path(__file__).parent / "fixtures" / "mbdump" / "url"
FIXTURE_MB_L_ARTIST_URL = Path(__file__).parent / "fixtures" / "mbdump" / "l_artist_url"
FIXTURE_MB_ARTIST = Path(__file__).parent / "fixtures" / "mbdump" / "artist"


def test_cli_build_artist_index_writes_consumable_tsv(tmp_path, capsys):
    from music_intel_mcp.artist_identity import MusicBrainzArtistUrlIndex

    out = tmp_path / "artist_uri_to_mbid.tsv"
    rc = main(
        [
            "build-artist-index",
            "--url-dump",
            str(FIXTURE_MB_URL),
            "--l-artist-url-dump",
            str(FIXTURE_MB_L_ARTIST_URL),
            "--artist-dump",
            str(FIXTURE_MB_ARTIST),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert "wrote 2 artist URI->MBID pairs" in capsys.readouterr().out
    # the freshly built TSV is exactly what the artist resolver's index reads
    index = MusicBrainzArtistUrlIndex(path=out)
    assert index.lookup("spotify:artist:spotArtist1") == "bbbbbbbb-1111-1111-1111-111111111111"
