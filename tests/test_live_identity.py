"""Live-capture identity waterfall (#139): AcoustID -> Spotify search -> ISRC
-> MBID -> MusicBrainz name search -> normalized name key.

All sources are in-memory test doubles or respx-mocked HTTP — no live AcoustID/
Spotify/MusicBrainz calls (AC9). ``identity.py``'s existing batch waterfall is
untouched by this module; these tests only exercise the new live-capture path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import respx

from music_intel_mcp.identity import InMemoryIsrcMbidIndex, InMemorySpotifyIsrcSource
from music_intel_mcp.live_identity import (
    ACOUSTID_LOOKUP_URL,
    AcoustIdApiSource,
    AcoustIdMatch,
    InMemoryAcoustIdSource,
    InMemoryMusicBrainzNameSearchSource,
    InMemorySpotifySearchSource,
    LiveIdentityResolver,
    LiveNegativeCache,
    ProvenanceSidecar,
    normalize_track_name,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


# --- AC4: normalization ----------------------------------------------------- #


@pytest.mark.parametrize(
    "title,artist,expected",
    [
        ("Song", "Artist", "song\x1fartist"),
        ("Song (feat. Other Artist)", "Artist", "song\x1fartist"),
        ("Song ft. Other", "Artist", "song\x1fartist"),
        ("Song featuring Other", "Artist", "song\x1fartist"),
        ("Song (Official Video)", "Artist", "song\x1fartist"),
        ("Song [Official Music Video]", "Artist", "song\x1fartist"),
        ("Song [Lyrics]", "Artist", "song\x1fartist"),
        ("Song (2011 Remaster)", "Artist", "song\x1fartist"),
        ("Song - 2011 Remastered", "Artist", "song\x1fartist"),
        ("Song – Title", "Artist", "song - title\x1fartist"),
        ("Song ‘quoted’", "Artist", "song 'quoted'\x1fartist"),
        ("Song   with   spaces", "Artist", "song with spaces\x1fartist"),
    ],
)
def test_normalize_track_name_strips_noise(title, artist, expected):
    assert normalize_track_name(title, artist) == expected


def test_normalize_track_name_keeps_live_remix_radio_edit():
    """Live/Remix/Radio Edit suffixes denote a different recording — must survive
    normalization (fragmentation over false merge)."""
    assert normalize_track_name("Song (Live)", "Artist") == "song (live)\x1fartist"
    assert normalize_track_name("Song (Remix)", "Artist") == "song (remix)\x1fartist"
    assert normalize_track_name("Song (Radio Edit)", "Artist") == "song (radio edit)\x1fartist"


def test_normalize_track_name_casefolds():
    assert normalize_track_name("SONG", "ARTIST") == normalize_track_name("song", "artist")


# --- AC2/AC3: waterfall order + AcoustID score gating ----------------------- #


def test_resolve_high_score_acoustid_wins_first_rung():
    acoustid = InMemoryAcoustIdSource({"fp1": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    spotify_search = InMemorySpotifySearchSource({("Song", "Artist"): "SP-1"})
    resolver = LiveIdentityResolver(acoustid_source=acoustid, spotify_search=spotify_search)

    ident = resolver.resolve(title="Song", artist="Artist", fingerprint="fp1", duration_s=180)

    assert ident.level == "acoustid"
    assert ident.mbid == "M-1"
    assert spotify_search.calls == []  # short-circuited, later rungs untouched


def test_resolve_low_score_acoustid_falls_through_to_spotify_search():
    """AC3: fragmentation over false merge — a low-confidence AcoustID match is
    discarded, not trusted."""
    acoustid = InMemoryAcoustIdSource({"fp1": [AcoustIdMatch(score=0.2, mbid="M-wrong")]})
    spotify_search = InMemorySpotifySearchSource({("Song", "Artist"): "SP-1"})
    resolver = LiveIdentityResolver(acoustid_source=acoustid, spotify_search=spotify_search)

    ident = resolver.resolve(title="Song", artist="Artist", fingerprint="fp1", duration_s=180)

    assert ident.level == "spotify_search"
    assert ident.spotify_id == "SP-1"
    assert ident.mbid is None


def test_resolve_spotify_search_then_isrc_then_mbid():
    spotify_search = InMemorySpotifySearchSource({("Song", "Artist"): "SP-1"})
    spotify_isrc = InMemorySpotifyIsrcSource({"SP-1": "ISRC-1"})
    isrc_index = InMemoryIsrcMbidIndex({"ISRC-1": "M-1"})
    resolver = LiveIdentityResolver(
        spotify_search=spotify_search, spotify_isrc=spotify_isrc, isrc_index=isrc_index
    )

    ident = resolver.resolve(title="Song", artist="Artist")

    assert ident.level == "mbid"
    assert ident.mbid == "M-1"
    assert ident.isrc == "ISRC-1"
    assert ident.spotify_id == "SP-1"


def test_resolve_falls_through_to_mb_name_search():
    spotify_search = InMemorySpotifySearchSource()  # no match
    mb_name_search = InMemoryMusicBrainzNameSearchSource({("Song", "Artist"): "M-2"})
    resolver = LiveIdentityResolver(spotify_search=spotify_search, mb_name_search=mb_name_search)

    ident = resolver.resolve(title="Song", artist="Artist")

    assert ident.level == "mb_name"
    assert ident.mbid == "M-2"


def test_resolve_bottoms_out_at_normalized_name_key():
    resolver = LiveIdentityResolver()  # no sources configured at all

    ident = resolver.resolve(title="Song (feat. X)", artist="Artist")

    assert ident.level == "name"
    assert ident.mbid is None
    assert ident.name_key == normalize_track_name("Song (feat. X)", "Artist")


def test_resolve_without_fingerprint_skips_acoustid_rung():
    acoustid = InMemoryAcoustIdSource({"fp1": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    resolver = LiveIdentityResolver(acoustid_source=acoustid)

    ident = resolver.resolve(title="Song", artist="Artist", fingerprint=None)

    assert acoustid.calls == []
    assert ident.level == "name"


# --- AC6: negative cache ----------------------------------------------------- #


def test_negative_cache_put_then_get_true_within_ttl(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    cache.put("mbname:song\x1fartist", reason="mb_name_search_miss")

    assert cache.get("mbname:song\x1fartist") is True


def test_negative_cache_expires_after_ttl(tmp_path: Path):
    clock = {"now": T0}
    cache = LiveNegativeCache(tmp_path, ttl_days=30, now=lambda: clock["now"])
    cache.put("mbname:song\x1fartist", reason="mb_name_search_miss")

    clock["now"] = T0 + timedelta(days=31)

    assert cache.get("mbname:song\x1fartist") is False


def test_negative_cache_miss_for_unknown_key(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    assert cache.get("mbname:never\x1fseen") is False


def test_resolver_skips_mb_name_search_on_cached_negative(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    cache.put("mbname:song\x1fartist", reason="mb_name_search_miss")
    mb_name_search = InMemoryMusicBrainzNameSearchSource({("Song", "Artist"): "M-2"})
    resolver = LiveIdentityResolver(mb_name_search=mb_name_search, negative_cache=cache)

    ident = resolver.resolve(title="Song", artist="Artist")

    assert mb_name_search.calls == []  # short-circuited by the cached miss
    assert ident.level == "name"


def test_resolver_caches_mb_name_search_miss(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    mb_name_search = InMemoryMusicBrainzNameSearchSource()  # no match
    resolver = LiveIdentityResolver(mb_name_search=mb_name_search, negative_cache=cache)

    resolver.resolve(title="Song", artist="Artist")

    assert cache.get("mbname:song\x1fartist") is True


def test_resolver_caches_acoustid_low_score_miss(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    acoustid = InMemoryAcoustIdSource({"fp1": [AcoustIdMatch(score=0.1, mbid="M-x")]})
    resolver = LiveIdentityResolver(acoustid_source=acoustid, negative_cache=cache)

    resolver.resolve(title="Song", artist="Artist", fingerprint="fp1", duration_s=180)

    assert cache.get("acoustid:fp1") is True


def test_resolver_skips_acoustid_lookup_on_cached_negative(tmp_path: Path):
    cache = LiveNegativeCache(tmp_path, now=lambda: T0)
    cache.put("acoustid:fp1", reason="acoustid_no_high_score_match")
    acoustid = InMemoryAcoustIdSource({"fp1": [AcoustIdMatch(score=0.95, mbid="M-1")]})
    resolver = LiveIdentityResolver(acoustid_source=acoustid, negative_cache=cache)

    ident = resolver.resolve(title="Song", artist="Artist", fingerprint="fp1", duration_s=180)

    assert acoustid.calls == []
    assert ident.level == "name"


# --- AC5: provenance sidecar -------------------------------------------------- #


def test_provenance_sidecar_round_trip():
    sidecar = ProvenanceSidecar(
        raw_title="Song (Official Video)",
        raw_artist="Artist",
        app_id="Spotify.exe",
        captured_at=T0.isoformat(),
        chromaprint_fingerprint="fp1",
    )

    payload = sidecar.model_dump()

    assert payload["raw_title"] == "Song (Official Video)"
    assert payload["chromaprint_fingerprint"] == "fp1"


def test_provenance_sidecar_fingerprint_optional():
    sidecar = ProvenanceSidecar(
        raw_title="Song", raw_artist="Artist", app_id="Spotify.exe", captured_at=T0.isoformat()
    )
    assert sidecar.chromaprint_fingerprint is None


# --- AC7/AC9: AcoustID API source, mocked --------------------------------- #


def test_acoustid_api_source_requires_api_key(monkeypatch):
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)
    source = AcoustIdApiSource(api_key=None)

    with pytest.raises(RuntimeError):
        source.match("fp1", 180.0)


def test_acoustid_api_source_parses_results():
    source = AcoustIdApiSource(api_key="test-key")

    with respx.mock(assert_all_called=False) as router:
        router.get(ACOUSTID_LOOKUP_URL).respond(
            json={
                "status": "ok",
                "results": [
                    {"score": 0.93, "id": "r1", "recordings": [{"id": "M-1"}]},
                    {"score": 0.4, "id": "r2", "recordings": [{"id": "M-2"}]},
                ],
            }
        )
        matches = source.match("fp1", 180.0)

    assert AcoustIdMatch(score=0.93, mbid="M-1") in matches
    assert AcoustIdMatch(score=0.4, mbid="M-2") in matches
