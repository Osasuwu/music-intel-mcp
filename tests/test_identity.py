"""Track identity resolution — the spotify_id -> ISRC -> MBID waterfall (#61).

No live MusicBrainz dump and no Spotify API: in-memory index/source fixtures and
a tiny synthetic TSV stand in. Asserts the waterfall, transparent flagging of
unresolved tracks, cache reuse on re-run, and resolution-coverage reporting.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from music_intel_mcp.identity import (
    IdentityCache,
    IdentityResolver,
    InMemoryIsrcMbidIndex,
    InMemorySpotifyIsrcSource,
    MusicBrainzIsrcIndex,
    ResolvedIdentity,
    to_metadata_records,
)
from music_intel_mcp.models import TrackRef
from music_intel_mcp.shared_store import InMemorySharedStore

FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index.tsv"
T0 = datetime(2026, 1, 1, tzinfo=UTC)


# --- waterfall ------------------------------------------------------------- #


def test_resolve_mbid_passthrough():
    """A track that already carries an MBID is resolved without any lookup."""
    index = InMemoryIsrcMbidIndex()
    resolver = IdentityResolver(index)

    ident = resolver.resolve(TrackRef(name="n", artist="a", mbid="M-1", isrc="I-1"))

    assert ident.level == "mbid"
    assert ident.resolved is True
    assert ident.mbid == "M-1"
    assert index.lookups == []  # no dump touched


def test_resolve_isrc_to_mbid_via_index():
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver = IdentityResolver(index)

    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-1"))

    assert ident.level == "mbid"
    assert ident.mbid == "M-1"
    assert ident.isrc == "I-1"
    assert index.lookups == ["I-1"]


def test_resolve_isrc_unmapped_stays_isrc_and_is_flagged():
    """ISRC present but absent from the dump -> not dropped, flagged at isrc."""
    index = InMemoryIsrcMbidIndex({})
    resolver = IdentityResolver(index)

    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-404"))

    assert ident.level == "isrc"
    assert ident.resolved is False
    assert ident.mbid is None
    assert ident.isrc == "I-404"


def test_resolve_spotify_to_isrc_to_mbid_full_chain():
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    source = InMemorySpotifyIsrcSource({"S-1": "I-1"})
    resolver = IdentityResolver(index, spotify_source=source)

    ident = resolver.resolve(TrackRef(name="n", artist="a", spotify_id="S-1"))

    assert ident.level == "mbid"
    assert ident.spotify_id == "S-1"
    assert ident.isrc == "I-1"
    assert ident.mbid == "M-1"


def test_resolve_spotify_without_source_stays_spotify():
    """No Spotify source wired -> the spotify->ISRC leg can't run; flagged."""
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver = IdentityResolver(index)  # no spotify_source

    ident = resolver.resolve(TrackRef(name="n", artist="a", spotify_id="S-1"))

    assert ident.level == "spotify"
    assert ident.resolved is False
    assert ident.isrc is None
    assert index.lookups == []


def test_resolve_name_only():
    resolver = IdentityResolver(InMemoryIsrcMbidIndex())

    ident = resolver.resolve(TrackRef(name="Only Name", artist="Some Artist"))

    assert ident.level == "name"
    assert ident.resolved is False


# --- batch report: counts, dedup, coverage, unresolved -------------------- #


def test_resolve_all_counts_dedup_and_coverage():
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver = IdentityResolver(index)
    tracks = [
        TrackRef(name="a", artist="x", mbid="M-9"),  # mbid
        TrackRef(name="b", artist="y", isrc="I-1"),  # isrc -> mbid
        TrackRef(name="b", artist="y", isrc="I-1"),  # duplicate of above
        TrackRef(name="c", artist="z", isrc="I-404"),  # isrc, unresolved
        TrackRef(name="d", artist="w", spotify_id="S-7"),  # spotify, unresolved
        TrackRef(name="e", artist="v"),  # name, unresolved
    ]

    report = resolver.resolve_all(tracks)

    assert report.n_unique == 5  # the duplicate collapses
    assert report.counts == {"mbid": 2, "isrc": 1, "spotify": 1, "name": 1}
    assert report.mbid_coverage == pytest.approx(2 / 5)
    assert len(report.unresolved) == 3  # isrc + spotify + name


def test_resolve_all_empty_is_zero_coverage():
    report = IdentityResolver(InMemoryIsrcMbidIndex()).resolve_all([])
    assert report.n_unique == 0
    assert report.mbid_coverage == 0.0
    assert report.unresolved == []


# --- cache reuse (no re-resolution on re-run) ----------------------------- #


def test_cache_reuse_skips_reresolution(tmp_path):
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    cache = IdentityCache(root=tmp_path)
    resolver = IdentityResolver(index, cache=cache)
    track = TrackRef(name="n", artist="a", isrc="I-1")

    first = resolver.resolve(track)
    assert index.lookups == ["I-1"]

    # second run: a fresh resolver + the same on-disk cache must NOT re-walk
    index2 = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver2 = IdentityResolver(index2, cache=IdentityCache(root=tmp_path))
    second = resolver2.resolve(track)

    assert second == first
    assert index2.lookups == []  # served from disk cache, dump untouched


def test_cache_put_get_roundtrip(tmp_path):
    cache = IdentityCache(root=tmp_path)
    ident = ResolvedIdentity(
        input_key="isrc:I-1", isrc="I-1", mbid="M-1", name="n", artist="a", level="mbid"
    )
    cache.put(ident)
    assert cache.get("isrc:I-1") == ident
    assert cache.get("isrc:NOPE") is None


# --- file-backed MusicBrainz index ---------------------------------------- #


def test_musicbrainz_index_reads_tsv_fixture():
    index = MusicBrainzIsrcIndex(path=FIXTURE_INDEX)
    assert index.lookup("USABC1234567") == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert index.lookup("GBXYZ0000001") == "11112222-3333-4444-5555-666677778888"
    assert index.lookup("NOT-IN-DUMP") is None


def test_musicbrainz_index_missing_file_is_empty(tmp_path):
    """No dump installed -> empty index (every lookup misses), never a crash."""
    index = MusicBrainzIsrcIndex(path=tmp_path / "absent.tsv")
    assert index.lookup("USABC1234567") is None


# --- shared-store write-back ---------------------------------------------- #


def test_to_metadata_records_keys_by_resolved_id_and_writes(tmp_path):
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver = IdentityResolver(index)
    report = resolver.resolve_all([TrackRef(name="n", artist="a", isrc="I-1")])

    records = to_metadata_records(report, now=T0)
    assert len(records) == 1
    rec = records[0]
    assert rec.track_id == "mbid:M-1"  # keyed by the resolved canonical id
    assert rec.mbid == "M-1" and rec.isrc == "I-1"

    store = InMemorySharedStore()
    store.upsert_tracks(records)
    assert set(store.get_tracks(["mbid:M-1"])) == {"mbid:M-1"}


# --- record contract: no per-user data ------------------------------------ #


def test_resolved_identity_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ResolvedIdentity(
            input_key="spotify:S",
            name="n",
            artist="a",
            level="spotify",
            user_id="petr",  # personal data has no place in an identity record
        )
