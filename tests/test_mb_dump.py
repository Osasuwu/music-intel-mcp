"""Builder for the ISRC -> recording-MBID TSV from raw MusicBrainz dump tables.

The raw dump lives OUTSIDE the repo (env-pointed, far too large to commit); the
builder distils the two relevant tables into the small ``isrc_to_mbid.tsv`` the
resolver consumes. Tests use tiny synthetic ``isrc`` / ``recording`` tables that
mimic the real column layout (verified against the MusicBrainz schema):

- ``isrc``      : id(0), recording(1), isrc(2), source, edits_pending, created
- ``recording`` : id(0), gid(1), name, ...  (front two columns stable)

The join is isrc.recording -> recording.id -> recording.gid (the MBID).
"""

from __future__ import annotations

from pathlib import Path

from music_intel_mcp.artist_identity import MusicBrainzArtistUrlIndex
from music_intel_mcp.identity import MusicBrainzIsrcIndex
from music_intel_mcp.mb_dump import build_artist_mbid_tsv, build_isrc_mbid_tsv

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mbdump"
ISRC_DUMP = FIXTURE_DIR / "isrc"
RECORDING_DUMP = FIXTURE_DIR / "recording"
URL_DUMP = FIXTURE_DIR / "url"
L_ARTIST_URL_DUMP = FIXTURE_DIR / "l_artist_url"
ARTIST_DUMP = FIXTURE_DIR / "artist"


def test_build_joins_isrc_to_recording_gid(tmp_path):
    out = tmp_path / "isrc_to_mbid.tsv"
    n = build_isrc_mbid_tsv(ISRC_DUMP, RECORDING_DUMP, out)

    # Three joinable pairs; the orphan ISRC (recording 999 absent) is dropped.
    assert n == 3
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert lines == [
        "USMULTI00001\t22223333-4444-5555-6666-777788889999",
        "USMULTI00001\t11112222-3333-4444-5555-666677778888",
        "USONE00000001\taaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    ]


def test_built_tsv_is_consumable_by_index(tmp_path):
    """The build output is exactly what :class:`MusicBrainzIsrcIndex` reads —
    the multi-valued ISRC round-trips to a two-candidate list."""
    out = tmp_path / "isrc_to_mbid.tsv"
    build_isrc_mbid_tsv(ISRC_DUMP, RECORDING_DUMP, out)

    index = MusicBrainzIsrcIndex(path=out)
    assert index.lookup_all("USMULTI00001") == [
        "22223333-4444-5555-6666-777788889999",
        "11112222-3333-4444-5555-666677778888",
    ]
    assert index.lookup("USONE00000001") == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert index.lookup_all("ORPHANISRC01") == []


def test_build_drops_isrc_pointing_at_unknown_recording(tmp_path):
    """An ISRC whose recording id is not in the recording table is skipped, not
    written with a blank/garbage MBID."""
    out = tmp_path / "isrc_to_mbid.tsv"
    build_isrc_mbid_tsv(ISRC_DUMP, RECORDING_DUMP, out)
    assert "ORPHANISRC01" not in out.read_text(encoding="utf-8")


def test_build_missing_dump_writes_empty(tmp_path):
    """A missing dump table yields an empty (header-only) index, never a crash —
    mirrors the index's own missing-file-is-empty stance."""
    out = tmp_path / "isrc_to_mbid.tsv"
    n = build_isrc_mbid_tsv(tmp_path / "absent_isrc", RECORDING_DUMP, out)
    assert n == 0
    assert MusicBrainzIsrcIndex(path=out).lookup_all("USMULTI00001") == []


# --- artist leg: url -> l_artist_url -> artist (#102) ----------------------- #


def test_build_artist_joins_url_to_artist_gid(tmp_path):
    out = tmp_path / "artist_uri_to_mbid.tsv"
    n = build_artist_mbid_tsv(URL_DUMP, L_ARTIST_URL_DUMP, ARTIST_DUMP, out)

    # Two joinable pairs: spotArtist1->artist 200, spotArtist2->artist 201.
    # Dropped: non-spotify url (12), unreferenced spotify url (13), and the
    # relationship pointing at an artist row absent from the artist table (203).
    assert n == 2
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    assert lines == [
        "spotify:artist:spotArtist1\tbbbbbbbb-1111-1111-1111-111111111111",
        "spotify:artist:spotArtist2\tbbbbbbbb-2222-2222-2222-222222222222",
    ]


def test_built_artist_tsv_is_consumable_by_index(tmp_path):
    out = tmp_path / "artist_uri_to_mbid.tsv"
    build_artist_mbid_tsv(URL_DUMP, L_ARTIST_URL_DUMP, ARTIST_DUMP, out)

    index = MusicBrainzArtistUrlIndex(path=out)
    assert index.lookup("spotify:artist:spotArtist1") == "bbbbbbbb-1111-1111-1111-111111111111"
    assert index.lookup("spotify:artist:spotArtist2") == "bbbbbbbb-2222-2222-2222-222222222222"
    assert index.lookup("spotify:artist:nope") is None


def test_build_artist_drops_url_never_used_in_relationship(tmp_path):
    """A Spotify artist URL that matches the regex but never turns up as an
    l_artist_url relationship is dropped, not emitted with a blank MBID."""
    out = tmp_path / "artist_uri_to_mbid.tsv"
    build_artist_mbid_tsv(URL_DUMP, L_ARTIST_URL_DUMP, ARTIST_DUMP, out)
    assert "spotArtistOrphanUrl" not in out.read_text(encoding="utf-8")


def test_build_artist_drops_relationship_pointing_at_unknown_artist(tmp_path):
    """A relationship whose artist id is absent from the artist table is
    dropped rather than emitted with a blank MBID."""
    out = tmp_path / "artist_uri_to_mbid.tsv"
    build_artist_mbid_tsv(URL_DUMP, L_ARTIST_URL_DUMP, ARTIST_DUMP, out)
    assert "spotArtistNoArtistRow" not in out.read_text(encoding="utf-8")


def test_build_artist_ignores_non_spotify_urls(tmp_path):
    out = tmp_path / "artist_uri_to_mbid.tsv"
    n = build_artist_mbid_tsv(URL_DUMP, L_ARTIST_URL_DUMP, ARTIST_DUMP, out)
    assert n == 2  # the musicbrainz.org url never contributes a pair


def test_build_artist_missing_dump_writes_empty(tmp_path):
    """A missing dump table yields an empty (header-only) index, never a
    crash — mirrors the track join's own missing-file-is-empty stance."""
    out = tmp_path / "artist_uri_to_mbid.tsv"
    n = build_artist_mbid_tsv(tmp_path / "absent_url", L_ARTIST_URL_DUMP, ARTIST_DUMP, out)
    assert n == 0
    assert MusicBrainzArtistUrlIndex(path=out).lookup("spotify:artist:spotArtist1") is None
