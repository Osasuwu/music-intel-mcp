"""Shared metadata store + pull-and-cache (#60).

No live Supabase, no network — the in-memory store is the real store here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from music_intel_mcp.models import TrackRef
from music_intel_mcp.shared_store import (
    AudioFeatures,
    InMemorySharedStore,
    MetadataCache,
    TrackMetadataRecord,
    TrackTag,
    canonical_track_id,
    is_stale,
    pull_and_cache,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _record(track_id: str, *, fetched_at: datetime = T0, name: str = "Song") -> TrackMetadataRecord:
    return TrackMetadataRecord(
        track_id=track_id,
        name=name,
        artist="Artist",
        audio_features=AudioFeatures(bpm=120.0, energy=0.8, source="acousticbrainz"),
        tags=[TrackTag(tag="dream pop", weight=0.9, source="lastfm")],
        fetched_at=fetched_at,
    )


# --- identity -------------------------------------------------------------- #


def test_canonical_track_id_waterfall():
    assert canonical_track_id(TrackRef(name="n", artist="a", mbid="M", isrc="I")) == "mbid:M"
    assert canonical_track_id(TrackRef(name="n", artist="a", isrc="I", spotify_id="S")) == "isrc:I"
    assert canonical_track_id(TrackRef(name="n", artist="a", spotify_id="S")) == "spotify:S"
    # fallback is case-folded name + artist
    assert canonical_track_id(TrackRef(name="Né", artist="A")) == "name:né\x1fa"


# --- record contract: no per-user data ------------------------------------- #


def test_record_rejects_personal_fields():
    """The shared store holds anonymous facts only — a stray per-user field
    must be refused (extra='forbid')."""
    with pytest.raises(ValidationError):
        TrackMetadataRecord(
            track_id="spotify:X",
            name="n",
            artist="a",
            fetched_at=T0,
            user_id="petr",  # personal data has no place in the shared store
        )


# --- TTL ------------------------------------------------------------------- #


def test_is_stale_ttl_boundary():
    rec = _record("spotify:X", fetched_at=T0)
    assert not is_stale(rec, now=T0 + timedelta(days=90), ttl_days=90)
    assert is_stale(rec, now=T0 + timedelta(days=91), ttl_days=90)


def test_is_stale_treats_naive_as_utc():
    rec = _record("spotify:X", fetched_at=datetime(2026, 1, 1))  # naive
    assert not is_stale(rec, now=datetime(2026, 1, 2))  # naive now, both UTC


# --- InMemorySharedStore --------------------------------------------------- #


def test_in_memory_store_roundtrip_and_spy():
    store = InMemorySharedStore([_record("spotify:A"), _record("spotify:B")])
    got = store.get_tracks(["spotify:A", "spotify:MISSING"])
    assert set(got) == {"spotify:A"}
    assert store.get_calls == [["spotify:A", "spotify:MISSING"]]
    # returned copies are isolated from the store's internal rows
    got["spotify:A"].name = "mutated"
    assert store.get_tracks(["spotify:A"])["spotify:A"].name == "Song"


# --- MetadataCache --------------------------------------------------------- #


def test_cache_roundtrip(tmp_path):
    cache = MetadataCache(root=tmp_path)
    rec = _record("spotify:A")
    cache.write(rec)
    assert cache.read("spotify:A") == rec
    assert cache.read("spotify:MISSING") is None


def test_cache_filename_safe_for_fallback_id(tmp_path):
    """The name/artist fallback id contains '\\x1f' and spaces; the filename
    must be filesystem-safe and round-trip."""
    cache = MetadataCache(root=tmp_path)
    tid = "name:some song\x1fsome artist"
    cache.write(_record(tid))
    [path] = list(cache.cache_dir.glob("*.json"))
    assert "/" not in path.name and "\x1f" not in path.name and " " not in path.name
    assert cache.read(tid).track_id == tid


# --- pull_and_cache -------------------------------------------------------- #


def test_pull_cold_writes_cache_and_returns_pulled(tmp_path):
    store = InMemorySharedStore([_record("spotify:A"), _record("spotify:B")])
    cache = MetadataCache(root=tmp_path)

    result = pull_and_cache(["spotify:A", "spotify:B"], store, cache, now=T0)

    assert set(result.records) == {"spotify:A", "spotify:B"}
    assert sorted(result.pulled) == ["spotify:A", "spotify:B"]
    assert result.cache_hits == []
    # materialised locally
    assert cache.read("spotify:A") is not None
    assert cache.read("spotify:B") is not None


def test_pull_warm_cache_does_not_touch_store(tmp_path):
    """Second pull is served entirely from the local cache — the no-round-trip
    invariant: the store is not queried at all."""
    store = InMemorySharedStore([_record("spotify:A")])
    cache = MetadataCache(root=tmp_path)

    pull_and_cache(["spotify:A"], store, cache, now=T0)
    assert len(store.get_calls) == 1

    result = pull_and_cache(["spotify:A"], store, cache, now=T0 + timedelta(days=1))
    assert result.cache_hits == ["spotify:A"]
    assert result.pulled == []
    assert len(store.get_calls) == 1  # store NOT queried the second time


def test_pull_single_bulk_query_for_mixed_batch(tmp_path):
    """A batch of N uncached tracks triggers exactly one bulk get_tracks call,
    not N per-track round-trips."""
    store = InMemorySharedStore([_record(f"spotify:{i}") for i in range(5)])
    cache = MetadataCache(root=tmp_path)

    pull_and_cache([f"spotify:{i}" for i in range(5)], store, cache, now=T0)

    assert len(store.get_calls) == 1
    assert len(store.get_calls[0]) == 5


def test_pull_stale_cache_triggers_refetch(tmp_path):
    store = InMemorySharedStore([_record("spotify:A", fetched_at=T0)])
    cache = MetadataCache(root=tmp_path)

    # initial pull at T0 caches the T0 record
    pull_and_cache(["spotify:A"], store, cache, now=T0)
    assert len(store.get_calls) == 1

    # enricher refreshes the shared store with a newer record
    fresh = _record("spotify:A", fetched_at=T0 + timedelta(days=100), name="Refreshed")
    store.upsert_tracks([fresh])

    # 100 days later the cached entry is stale -> re-fetch from the store
    later = T0 + timedelta(days=100)
    result = pull_and_cache(["spotify:A"], store, cache, now=later)

    assert len(store.get_calls) == 2  # re-fetch path taken
    assert result.pulled == ["spotify:A"]
    assert result.records["spotify:A"].name == "Refreshed"
    assert cache.read("spotify:A").name == "Refreshed"  # cache rewritten


def test_pull_missing_track_surfaced(tmp_path):
    store = InMemorySharedStore([_record("spotify:A")])
    cache = MetadataCache(root=tmp_path)

    result = pull_and_cache(["spotify:A", "spotify:GHOST"], store, cache, now=T0)

    assert result.pulled == ["spotify:A"]
    assert result.missing == ["spotify:GHOST"]
    assert "spotify:GHOST" not in result.records


def test_pull_store_entry_past_ttl_flagged_stale(tmp_path):
    """A store record that is itself past TTL is materialised but flagged for
    re-enrichment, not treated as fresh."""
    store = InMemorySharedStore([_record("spotify:A", fetched_at=T0)])
    cache = MetadataCache(root=tmp_path)

    result = pull_and_cache(["spotify:A"], store, cache, now=T0 + timedelta(days=200))

    assert result.stale == ["spotify:A"]
    assert result.pulled == []
    assert "spotify:A" in result.records  # still usable for partial analysis
