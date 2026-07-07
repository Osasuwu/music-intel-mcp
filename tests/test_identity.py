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
    disambiguate_mbids,
    to_metadata_records,
)
from music_intel_mcp.models import TrackRef
from music_intel_mcp.shared_store import InMemorySharedStore

FIXTURE_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index.tsv"
FIXTURE_MULTI_INDEX = Path(__file__).parent / "fixtures" / "isrc_mbid_index_multi.tsv"
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


# --- cache versioning (#87 AC-D): schema bump invalidates stale entries ---- #


def test_cache_writes_under_versioned_dir(tmp_path):
    """Entries live under ``identity/v<N>/`` so a schema bump lands in a fresh
    namespace and the old one is simply never consulted."""
    cache = IdentityCache(root=tmp_path, schema_version=3)
    ident = ResolvedIdentity(
        input_key="isrc:I-1", isrc="I-1", mbid="M-1", name="n", artist="a", level="mbid"
    )
    path = cache.put(ident)
    assert path.parent == tmp_path / "identity" / "v3"


def test_cache_version_bump_invalidates_stale_entry(tmp_path):
    """An entry written under one schema version is invisible to a cache reading
    the next version — a re-run re-resolves rather than trusting stale data."""
    ident = ResolvedIdentity(
        input_key="isrc:I-1", isrc="I-1", mbid="M-1", name="n", artist="a", level="mbid"
    )
    IdentityCache(root=tmp_path, schema_version=1).put(ident)
    assert IdentityCache(root=tmp_path, schema_version=2).get("isrc:I-1") is None
    # same version still hits
    assert IdentityCache(root=tmp_path, schema_version=1).get("isrc:I-1") == ident


def test_cache_corrupt_entry_is_miss_not_crash(tmp_path):
    """A cached file that no longer validates (schema drift within a version, or a
    truncated write) is treated as a miss — the pipeline re-resolves instead of
    crashing on a stale entry."""
    cache = IdentityCache(root=tmp_path)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    cache._path("isrc:I-1").write_text("{ not valid json", encoding="utf-8")
    assert cache.get("isrc:I-1") is None


# --- only terminal (MBID) resolutions are cacheable (#87) ------------------ #


def test_stale_sub_mbid_cache_entry_is_rewalked_not_trusted(tmp_path):
    """A cached entry that only reached ISRC level records a *negative*: the index
    at resolution time lacked that ISRC. A later run whose index DOES map it must
    re-walk and resolve to MBID, not serve the stale miss. Regression for #87 —
    an isrc-level cache written before the MB index existed masked the whole
    6M-pair join, pinning coverage at 0."""
    stale = ResolvedIdentity(
        input_key="isrc:I-1", isrc="I-1", mbid=None, name="n", artist="a", level="isrc"
    )
    IdentityCache(root=tmp_path).put(stale)  # a pre-index run's frozen miss

    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})  # the index has since grown
    resolver = IdentityResolver(index, cache=IdentityCache(root=tmp_path))
    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-1"))

    assert ident.level == "mbid"
    assert ident.mbid == "M-1"
    assert index.lookups == ["I-1"]  # re-walked; did not trust the stale miss


def test_partial_resolution_is_not_persisted(tmp_path):
    """Only terminal (MBID) resolutions are cached. A partial is provisional —
    left uncached so the next run retries it against then-current data instead of
    freezing a miss that a grown index could satisfy."""
    index = InMemoryIsrcMbidIndex({})  # ISRC absent -> resolution stalls at isrc
    cache = IdentityCache(root=tmp_path)
    resolver = IdentityResolver(index, cache=cache)

    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-404"))

    assert ident.level == "isrc"
    assert cache.get(ident.input_key) is None  # nothing frozen


def test_terminal_mbid_resolution_is_still_cached(tmp_path):
    """The valuable case is unchanged: an MBID-level resolution is persisted and a
    fresh resolver serves it from disk without re-walking the dump."""
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver = IdentityResolver(index, cache=IdentityCache(root=tmp_path))
    track = TrackRef(name="n", artist="a", isrc="I-1")

    first = resolver.resolve(track)
    assert first.level == "mbid"

    index2 = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    resolver2 = IdentityResolver(index2, cache=IdentityCache(root=tmp_path))
    second = resolver2.resolve(track)

    assert second == first
    assert index2.lookups == []  # served from disk, dump untouched


# --- resolved MBID enumeration (#87 AC-E: the AB build's MBID source) ------ #


def test_resolved_mbids_empty_when_no_cache_dir(tmp_path):
    """No resolutions yet -> empty list, not a crash on a missing dir."""
    assert IdentityCache(root=tmp_path).resolved_mbids() == []


def test_resolved_mbids_returns_distinct_sorted(tmp_path):
    """Enumerates the distinct resolved MBIDs (the AB feature build's input),
    deduped and sorted. Two ISRCs mapping to the same MBID collapse to one."""
    cache = IdentityCache(root=tmp_path)
    for key, isrc, mbid in [
        ("isrc:I-1", "I-1", "m-ccc"),
        ("isrc:I-2", "I-2", "m-aaa"),
        ("isrc:I-3", "I-3", "m-aaa"),  # same MBID as I-2 -> collapses
    ]:
        cache.put(
            ResolvedIdentity(
                input_key=key, isrc=isrc, mbid=mbid, name="n", artist="a", level="mbid"
            )
        )
    assert cache.resolved_mbids() == ["m-aaa", "m-ccc"]


def test_resolved_mbids_skips_corrupt_entry(tmp_path):
    """A torn/invalid cache file is skipped, not fatal — the rest still enumerate."""
    cache = IdentityCache(root=tmp_path)
    cache.put(
        ResolvedIdentity(
            input_key="isrc:I-1", isrc="I-1", mbid="m-aaa", name="n", artist="a", level="mbid"
        )
    )
    cache.cache_dir.joinpath("corrupt.json").write_text("{ not json", encoding="utf-8")
    assert cache.resolved_mbids() == ["m-aaa"]


# --- multi-valued ISRC disambiguation (#87 AC-B) --------------------------- #


def test_disambiguate_empty_is_none():
    assert disambiguate_mbids([]) is None


def test_disambiguate_no_ab_picks_smallest_mbid():
    """Absent AB info, the tie-break is the smallest MBID (deterministic)."""
    assert disambiguate_mbids(["m-ccc", "m-aaa", "m-bbb"]) == "m-aaa"


def test_disambiguate_prefers_ab_covered_over_smaller_uncovered():
    """An AB-covered recording wins even when a smaller MBID is uncovered — the
    join we actually need is to AcousticBrainz features."""
    covered = {"m-ccc"}
    picked = disambiguate_mbids(["m-aaa", "m-bbb", "m-ccc"], ab_covered=covered.__contains__)
    assert picked == "m-ccc"


def test_disambiguate_tie_breaks_smallest_among_ab_covered():
    covered = {"m-bbb", "m-ddd"}
    picked = disambiguate_mbids(["m-ddd", "m-aaa", "m-bbb"], ab_covered=covered.__contains__)
    assert picked == "m-bbb"


def test_disambiguate_falls_back_to_smallest_when_none_covered():
    picked = disambiguate_mbids(["m-ccc", "m-aaa"], ab_covered=lambda _m: False)
    assert picked == "m-aaa"


# --- multi-valued in-memory index ----------------------------------------- #


def test_inmemory_index_multi_valued_lookup_all_and_lookup():
    index = InMemoryIsrcMbidIndex({"I-1": ["M-2", "M-1"]})
    assert index.lookup_all("I-1") == ["M-2", "M-1"]  # insertion order preserved
    assert index.lookup("I-1") == "M-1"  # disambiguated: smallest
    assert index.lookup_all("I-404") == []


def test_inmemory_index_str_value_backward_compatible():
    """A plain ``{isrc: mbid}`` mapping (older fixtures/tests) still works."""
    index = InMemoryIsrcMbidIndex({"I-1": "M-1"})
    assert index.lookup_all("I-1") == ["M-1"]
    assert index.lookup("I-1") == "M-1"


def test_resolver_disambiguates_with_ab_coverage():
    """The resolver prefers the AB-covered MBID when an ISRC maps to several."""
    index = InMemoryIsrcMbidIndex({"I-1": ["m-aaa", "m-zzz"]})
    resolver = IdentityResolver(index, ab_covered={"m-zzz"}.__contains__)

    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-1"))

    assert ident.mbid == "m-zzz"  # AB-covered wins over smaller m-aaa
    assert ident.level == "mbid"


def test_resolver_without_ab_coverage_picks_smallest():
    index = InMemoryIsrcMbidIndex({"I-1": ["m-zzz", "m-aaa"]})
    resolver = IdentityResolver(index)  # no ab_covered

    ident = resolver.resolve(TrackRef(name="n", artist="a", isrc="I-1"))

    assert ident.mbid == "m-aaa"


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


def test_musicbrainz_index_multi_valued_tsv_accumulates():
    """Repeated ISRC rows accumulate into a candidate list; ``lookup`` returns the
    disambiguated (smallest, no AB info) MBID while ``lookup_all`` keeps all."""
    index = MusicBrainzIsrcIndex(path=FIXTURE_MULTI_INDEX)
    assert index.lookup_all("USMULTI00001") == [
        "22223333-4444-5555-6666-777788889999",
        "11112222-3333-4444-5555-666677778888",
    ]
    assert index.lookup("USMULTI00001") == "11112222-3333-4444-5555-666677778888"
    assert index.lookup_all("USONE00000001") == ["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"]


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
