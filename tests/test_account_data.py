"""Spotify **Account Data** explicit-preference importer (#97).

These tests pin the ``YourLibrary.json`` / ``Playlist1.json`` / ``Marquee.json``
-> ``Library`` mapping, the drop-accounting for unmapped source keys and
non-``spotify:track:`` playlist URIs (nothing silent), the idempotent
``data/library.json`` write, the prior-vs-new diff, and the ``import-account``
CLI report (measured MBID coverage, drop ledger, diff). Fixtures are built
in-test as JSON so no export is committed and the exact source shape is visible.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.account_data import (
    AccountDataStats,
    diff_libraries,
    load_account_data,
    load_marquee_file,
    load_playlists_file,
    load_your_library_file,
)
from music_intel_mcp.cli import main
from music_intel_mcp.models import (
    AlbumRef,
    ArtistRef,
    Library,
    MarqueeEntry,
    Playlist,
    TrackRef,
)
from music_intel_mcp.store import UserStore

# --------------------------------------------------------------------------- #
# fixture builders — mirror the real Account Data export shapes
# --------------------------------------------------------------------------- #


def _your_library(**overrides) -> dict:
    base = {
        "tracks": [
            {
                "artist": "Flower Face",
                "album": "Baby Teeth",
                "track": "Maniac",
                "uri": "spotify:track:56O4WlGoUJiwfwoyRYqTe9",
            }
        ],
        "albums": [
            {"artist": "Radiohead", "album": "In Rainbows", "uri": "spotify:album:albumAAA"}
        ],
        "shows": [{"name": "Some Podcast", "uri": "spotify:show:showAAA"}],
        "episodes": [],
        "bannedTracks": [{"artist": "X", "track": "Y", "uri": "spotify:track:bannedTrk"}],
        "artists": [{"name": "Aphex Twin", "uri": "spotify:artist:artistAAA"}],
        "bannedArtists": [{"name": "Spam Bot", "uri": "spotify:artist:bannedArt"}],
        "other": [{"foo": "bar"}],
    }
    base.update(overrides)
    return base


def _playlists(**overrides) -> dict:
    base = {
        "playlists": [
            {
                "name": "Discover Weekly Archive",
                "lastModifiedDate": "2023-05-01",
                "items": [
                    {
                        "track": {
                            "trackName": "Maniac",
                            "artistName": "Flower Face",
                            "albumName": "Baby Teeth",
                            "trackUri": "spotify:track:56O4WlGoUJiwfwoyRYqTe9",
                        },
                        "episode": None,
                        "localTrack": None,
                        "addedDate": "2023-04-30",
                    },
                    {  # a local track — not a spotify:track: URI, must be drop-counted
                        "track": {
                            "trackName": "Home Recording",
                            "artistName": "Me",
                            "albumName": None,
                            "trackUri": "spotify:local:::Home+Recording:180",
                        },
                        "episode": None,
                        "localTrack": None,
                        "addedDate": "2023-04-29",
                    },
                ],
                "description": "Spotify's picks that landed",
                "numberOfFollowers": 0,
            }
        ]
    }
    base.update(overrides)
    return base


def _marquee() -> list[dict]:
    return [
        {"artistName": "Aphex Twin", "segment": "Super Listeners"},
        {"artistName": "Radiohead", "segment": "Moderate Listeners"},
    ]


def _write_export(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "YourLibrary.json").write_text(json.dumps(_your_library()), encoding="utf-8")
    (dir_path / "Playlist1.json").write_text(json.dumps(_playlists()), encoding="utf-8")
    (dir_path / "Marquee.json").write_text(json.dumps(_marquee()), encoding="utf-8")
    return dir_path


NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# TrackRef.album — backward-compatible optional field
# --------------------------------------------------------------------------- #


def test_trackref_album_defaults_none_and_old_history_line_validates():
    assert TrackRef(name="X", artist="Y").album is None
    old = '{"track":{"name":"X","artist":"Y"},"played_at":"2024-01-01T00:00:00Z","source":"ifttt"}'
    from music_intel_mcp.models import ListenEvent

    e = ListenEvent.model_validate_json(old)
    assert e.track.album is None


def test_trackref_album_carries_when_present():
    assert TrackRef(name="X", artist="Y", album="Z").album == "Z"


# --------------------------------------------------------------------------- #
# YourLibrary.json parsing
# --------------------------------------------------------------------------- #


def test_your_library_maps_likes_bans_follows_albums(tmp_path):
    path = tmp_path / "YourLibrary.json"
    path.write_text(json.dumps(_your_library()), encoding="utf-8")
    stats = AccountDataStats()
    parsed = load_your_library_file(path, stats=stats)

    (liked,) = parsed.liked_tracks
    assert liked == TrackRef(
        spotify_id="56O4WlGoUJiwfwoyRYqTe9",
        name="Maniac",
        artist="Flower Face",
        album="Baby Teeth",
    )
    assert liked.mbid is None and liked.isrc is None  # resolution never persisted

    assert parsed.followed_artists == [ArtistRef(name="Aphex Twin", uri="spotify:artist:artistAAA")]
    assert parsed.banned_artists == [ArtistRef(name="Spam Bot", uri="spotify:artist:bannedArt")]
    assert parsed.saved_albums == [
        AlbumRef(name="In Rainbows", artist="Radiohead", uri="spotify:album:albumAAA")
    ]


def test_your_library_drop_accounts_unmapped_keys(tmp_path):
    path = tmp_path / "YourLibrary.json"
    path.write_text(json.dumps(_your_library()), encoding="utf-8")
    stats = AccountDataStats()
    load_your_library_file(path, stats=stats)
    # bannedTracks(1) + shows(1) + episodes(0) + other(1) counted, never imported
    assert stats.dropped_keys["bannedTracks"] == 1
    assert stats.dropped_keys["shows"] == 1
    assert stats.dropped_keys["other"] == 1
    assert stats.total_dropped >= 3


# --------------------------------------------------------------------------- #
# Playlist1.json parsing — per-item addedDate + non-track URI drop
# --------------------------------------------------------------------------- #


def test_playlists_map_items_with_added_at_and_drop_non_track_uris(tmp_path):
    path = tmp_path / "Playlist1.json"
    path.write_text(json.dumps(_playlists()), encoding="utf-8")
    stats = AccountDataStats()
    playlists = load_playlists_file(path, stats=stats)

    (pl,) = playlists
    assert isinstance(pl, Playlist)
    assert pl.name == "Discover Weekly Archive"
    assert pl.last_modified_date == "2023-05-01"
    assert pl.number_of_followers == 0
    # the local-track item is dropped + counted; only the spotify:track: item survives
    assert len(pl.items) == 1
    item = pl.items[0]
    assert item.track.spotify_id == "56O4WlGoUJiwfwoyRYqTe9"
    assert item.track.album == "Baby Teeth"
    assert item.added_at == "2023-04-30"
    assert stats.dropped_non_track_playlist_uris == 1


# --------------------------------------------------------------------------- #
# Marquee.json parsing — verbatim segment strings, no artist URI
# --------------------------------------------------------------------------- #


def test_marquee_maps_verbatim_segments(tmp_path):
    path = tmp_path / "Marquee.json"
    path.write_text(json.dumps(_marquee()), encoding="utf-8")
    entries = load_marquee_file(path)
    assert entries == [
        MarqueeEntry(artist_name="Aphex Twin", segment="Super Listeners"),
        MarqueeEntry(artist_name="Radiohead", segment="Moderate Listeners"),
    ]


# --------------------------------------------------------------------------- #
# load_account_data — the whole Library, header, schema_version
# --------------------------------------------------------------------------- #


def test_load_account_data_builds_full_library(tmp_path):
    export = _write_export(tmp_path / "export")
    stats = AccountDataStats()
    lib = load_account_data(export, now=NOW, stats=stats)

    assert isinstance(lib, Library)
    assert lib.schema_version == "v1"
    assert lib.header.source == "spotify_account_data"
    assert lib.header.imported_at == NOW
    assert len(lib.liked_tracks) == 1
    assert len(lib.banned_artists) == 1
    assert len(lib.followed_artists) == 1
    assert len(lib.saved_albums) == 1
    assert len(lib.playlists) == 1
    assert len(lib.marquee) == 2


def test_library_forbids_extra_fields():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        Library.model_validate(
            {
                "schema_version": "v1",
                "header": {"source": "spotify_account_data", "imported_at": NOW.isoformat()},
                "surprise": 1,
            }
        )


# --------------------------------------------------------------------------- #
# UserStore.load_library / write_library — idempotent replace
# --------------------------------------------------------------------------- #


def test_store_write_then_load_round_trips(tmp_path):
    export = _write_export(tmp_path / "export")
    lib = load_account_data(export, now=NOW)
    store = UserStore(root=tmp_path / "data")
    assert store.load_library() is None  # honest-empty before any import
    store.write_library(lib)
    loaded = store.load_library()
    assert loaded == lib


def test_store_write_is_idempotent(tmp_path):
    export = _write_export(tmp_path / "export")
    store = UserStore(root=tmp_path / "data")
    store.write_library(load_account_data(export, now=NOW))
    first = (tmp_path / "data" / "library.json").read_text(encoding="utf-8")
    store.write_library(load_account_data(export, now=NOW))
    second = (tmp_path / "data" / "library.json").read_text(encoding="utf-8")
    assert first == second


# --------------------------------------------------------------------------- #
# diff_libraries — added/removed likes/bans/follows
# --------------------------------------------------------------------------- #


def test_diff_reports_added_and_removed(tmp_path):
    export = _write_export(tmp_path / "export")
    old = load_account_data(export, now=NOW)

    yl = _your_library(
        tracks=[
            {  # a new like replaces the old one
                "artist": "Boards of Canada",
                "album": "Geogaddi",
                "track": "1969",
                "uri": "spotify:track:newLike",
            }
        ],
        bannedArtists=[],  # the ban was lifted
    )
    (export / "YourLibrary.json").write_text(json.dumps(yl), encoding="utf-8")
    new = load_account_data(export, now=NOW)

    diff = diff_libraries(old, new)
    assert diff.likes_added == 1
    assert diff.likes_removed == 1
    assert diff.bans_removed == 1
    assert diff.bans_added == 0
    assert diff.follows_added == 0 and diff.follows_removed == 0


# --------------------------------------------------------------------------- #
# CLI import-account — report, drop ledger, coverage, idempotency
# --------------------------------------------------------------------------- #


def test_cli_import_account_writes_library_and_reports(tmp_path, capsys):
    export = _write_export(tmp_path / "export")
    rc = main(["import-account", "--from", str(export), "--data-dir", str(tmp_path / "data")])
    assert rc == 0

    lib = UserStore(root=tmp_path / "data").load_library()
    assert lib is not None
    assert len(lib.liked_tracks) == 1

    out = capsys.readouterr().out
    assert "likes=1" in out
    assert "banned_artists=1" in out
    # drop-accounting surfaced, never silent
    assert "dropped" in out
    # measured MBID coverage over liked + playlist tracks (report-only)
    assert "mbid coverage" in out.lower()


def test_cli_import_account_is_idempotent(tmp_path):
    export = _write_export(tmp_path / "export")
    data_dir = tmp_path / "data"
    main(["import-account", "--from", str(export), "--data-dir", str(data_dir)])
    first = (data_dir / "library.json").read_text(encoding="utf-8")
    main(["import-account", "--from", str(export), "--data-dir", str(data_dir)])
    second = (data_dir / "library.json").read_text(encoding="utf-8")
    assert first == second
