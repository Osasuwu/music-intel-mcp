"""Artist identity resolution — the spotify artist URI -> artist MBID
waterfall (#102).

No live MusicBrainz dump and no Spotify API: in-memory index fixtures and a
tiny synthetic TSV stand in (mirrors test_identity.py's discipline). Asserts
the waterfall, transparent flagging of unresolved (name-only) artists, cache
reuse on re-run, and resolution-coverage reporting.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from music_intel_mcp.artist_identity import (
    ArtistIdentityCache,
    ArtistIdentityResolver,
    InMemoryArtistUrlMbidIndex,
    MusicBrainzArtistUrlIndex,
    ResolvedArtist,
    canonical_artist_key,
)
from music_intel_mcp.models import ArtistRef, MarqueeEntry

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mbdump"


# --- canonical key ----------------------------------------------------------- #


def test_canonical_key_prefers_uri_over_name():
    assert canonical_artist_key(uri="spotify:artist:1", name="Anything") == "uri:spotify:artist:1"


def test_canonical_key_falls_back_to_casefolded_name():
    assert canonical_artist_key(uri=None, name="Some Artist") == "name:some artist"
    assert canonical_artist_key(uri=None, name="SOME ARTIST") == "name:some artist"


# --- waterfall ---------------------------------------------------------------- #


def test_resolve_uri_to_mbid_via_index():
    index = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})
    resolver = ArtistIdentityResolver(index)

    ident = resolver.resolve(uri="spotify:artist:1", name="Artist One")

    assert ident.level == "mbid"
    assert ident.resolved is True
    assert ident.mbid == "M-1"
    assert index.lookups == ["spotify:artist:1"]


def test_resolve_uri_unmapped_stays_name_and_is_flagged():
    """URI present but absent from the dump -> not dropped, flagged unresolved."""
    index = InMemoryArtistUrlMbidIndex({})
    resolver = ArtistIdentityResolver(index)

    ident = resolver.resolve(uri="spotify:artist:404", name="Unknown Artist")

    assert ident.level == "name"
    assert ident.resolved is False
    assert ident.mbid is None
    assert ident.uri == "spotify:artist:404"


def test_resolve_name_only_no_uri_is_unresolved():
    resolver = ArtistIdentityResolver(InMemoryArtistUrlMbidIndex())

    ident = resolver.resolve(uri=None, name="Marquee Only")

    assert ident.level == "name"
    assert ident.resolved is False
    assert ident.uri is None


def test_resolve_name_index_seam_used_when_wired():
    """The name-match rung is unshipped by default, but the seam works when a
    caller wires one in explicitly (e.g. a future precision-measured index)."""
    resolver = ArtistIdentityResolver(
        InMemoryArtistUrlMbidIndex(), name_index={"marquee only": "M-9"}
    )

    ident = resolver.resolve(uri=None, name="Marquee Only")

    assert ident.level == "mbid"
    assert ident.mbid == "M-9"


def test_resolve_uri_lookup_wins_over_name_index():
    resolver = ArtistIdentityResolver(
        InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"}),
        name_index={"artist one": "M-WRONG"},
    )

    ident = resolver.resolve(uri="spotify:artist:1", name="Artist One")

    assert ident.mbid == "M-1"


# --- ref / marquee entry-points ------------------------------------------------ #


def test_resolve_ref_uses_uri_and_name():
    index = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})
    resolver = ArtistIdentityResolver(index)

    ident = resolver.resolve_ref(ArtistRef(name="Artist One", uri="spotify:artist:1"))

    assert ident.mbid == "M-1"
    assert ident.name == "Artist One"


def test_resolve_marquee_has_no_uri():
    resolver = ArtistIdentityResolver(InMemoryArtistUrlMbidIndex())

    ident = resolver.resolve_marquee(
        MarqueeEntry(artist_name="Marquee Artist", segment="Super Listeners")
    )

    assert ident.uri is None
    assert ident.level == "name"


# --- batch report: counts, dedup, coverage, unresolved ------------------------- #


def test_resolve_all_counts_dedup_and_coverage():
    index = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})
    resolver = ArtistIdentityResolver(index)
    artists = [
        ArtistRef(name="A", uri="spotify:artist:1"),  # uri -> mbid
        ArtistRef(name="A", uri="spotify:artist:1"),  # duplicate
        ArtistRef(name="B", uri="spotify:artist:404"),  # uri, unresolved
        ArtistRef(name="C"),  # name only, unresolved
    ]
    marquee = [
        MarqueeEntry(artist_name="D", segment="Super Listeners"),  # name only
        # NOT a dedup of ArtistRef A: canonical key is uri-keyed there, name-keyed
        # here — dedup is purely by canonical key, not by display name.
        MarqueeEntry(artist_name="A", segment="Previously Active"),
    ]

    report = resolver.resolve_all(artists, marquee)

    assert report.n_unique == 5  # the ArtistRef "A" duplicate collapses; Marquee "A" does not
    assert report.counts == {"mbid": 1, "name": 4}
    assert report.mbid_coverage == pytest.approx(1 / 5)
    assert len(report.unresolved) == 4


def test_resolve_all_empty_is_zero_coverage():
    report = ArtistIdentityResolver(InMemoryArtistUrlMbidIndex()).resolve_all([])
    assert report.n_unique == 0
    assert report.mbid_coverage == 0.0
    assert report.unresolved == []


# --- cache reuse (no re-resolution on re-run) ---------------------------------- #


def test_cache_reuse_skips_reresolution(tmp_path):
    index = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})
    cache = ArtistIdentityCache(root=tmp_path)
    resolver = ArtistIdentityResolver(index, cache=cache)

    first = resolver.resolve(uri="spotify:artist:1", name="Artist One")
    assert index.lookups == ["spotify:artist:1"]

    index2 = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})
    resolver2 = ArtistIdentityResolver(index2, cache=ArtistIdentityCache(root=tmp_path))
    second = resolver2.resolve(uri="spotify:artist:1", name="Artist One")

    assert second == first
    assert index2.lookups == []  # served from disk cache, dump untouched


def test_cache_put_get_roundtrip(tmp_path):
    cache = ArtistIdentityCache(root=tmp_path)
    ident = ResolvedArtist(
        input_key="uri:spotify:artist:1", uri="spotify:artist:1", mbid="M-1", name="n", level="mbid"
    )
    cache.put(ident)
    assert cache.get("uri:spotify:artist:1") == ident
    assert cache.get("uri:nope") is None


def test_cache_corrupt_entry_is_miss_not_crash(tmp_path):
    cache = ArtistIdentityCache(root=tmp_path)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    cache._path("uri:spotify:artist:1").write_text("{ not valid json", encoding="utf-8")
    assert cache.get("uri:spotify:artist:1") is None


def test_stale_unresolved_cache_entry_is_rewalked_not_trusted(tmp_path):
    """Only terminal (MBID) resolutions are cached — an unresolved (name-level)
    result is never persisted, so a since-grown index is re-consulted on the
    next run rather than trusting a frozen miss (mirrors decision dddc4d90)."""
    index = InMemoryArtistUrlMbidIndex({})  # miss on first run
    cache = ArtistIdentityCache(root=tmp_path)
    resolver = ArtistIdentityResolver(index, cache=cache)

    first = resolver.resolve(uri="spotify:artist:1", name="Artist One")
    assert first.level == "name"
    assert cache.get(first.input_key) is None  # nothing frozen

    index2 = InMemoryArtistUrlMbidIndex({"spotify:artist:1": "M-1"})  # index has since grown
    resolver2 = ArtistIdentityResolver(index2, cache=ArtistIdentityCache(root=tmp_path))
    second = resolver2.resolve(uri="spotify:artist:1", name="Artist One")

    assert second.level == "mbid"
    assert index2.lookups == ["spotify:artist:1"]  # re-walked, not trusted from a stale miss


def test_cache_writes_under_versioned_dir(tmp_path):
    cache = ArtistIdentityCache(root=tmp_path, schema_version=3)
    ident = ResolvedArtist(
        input_key="uri:spotify:artist:1", uri="spotify:artist:1", mbid="M-1", name="n", level="mbid"
    )
    path = cache.put(ident)
    assert path.parent == tmp_path / "artist_identity" / "v3"


# --- file-backed MusicBrainz artist index -------------------------------------- #


def test_musicbrainz_artist_index_missing_file_is_empty(tmp_path):
    index = MusicBrainzArtistUrlIndex(path=tmp_path / "absent.tsv")
    assert index.lookup("spotify:artist:1") is None


def test_musicbrainz_artist_index_reads_tsv(tmp_path):
    tsv = tmp_path / "artist_uri_to_mbid.tsv"
    tsv.write_text(
        "# comment line\nspotify:artist:1\tM-1\nspotify:artist:2\tM-2\n", encoding="utf-8"
    )
    index = MusicBrainzArtistUrlIndex(path=tsv)
    assert index.lookup("spotify:artist:1") == "M-1"
    assert index.lookup("spotify:artist:2") == "M-2"
    assert index.lookup("spotify:artist:404") is None


# --- record contract: no per-user data ----------------------------------------- #


def test_resolved_artist_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ResolvedArtist(
            input_key="uri:spotify:artist:1",
            name="n",
            level="name",
            user_id="petr",  # personal data has no place in an identity record
        )
