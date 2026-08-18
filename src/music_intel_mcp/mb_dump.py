"""Distil the raw MusicBrainz dump into small join TSVs (#87 AC-B, #102).

The full MusicBrainz export is tens of millions of rows and lives OUTSIDE the
repo (env-pointed, never committed — see CLAUDE.md). Neither
:class:`~music_intel_mcp.identity.MusicBrainzIsrcIndex` nor
:class:`~music_intel_mcp.artist_identity.MusicBrainzArtistUrlIndex` reads the
raw dump directly; this module builds their compact TSV extracts once,
offline.

## Track leg: ISRC -> recording MBID (:func:`build_isrc_mbid_tsv`)

Two raw tables are joined (column layout verified against the MusicBrainz
schema; the front columns are stable across dump history):

- ``isrc``      : id(0), **recording(1)**, **isrc(2)**, source, edits_pending, created
- ``recording`` : id(0), **gid(1)**, name, ...

Join path: ``isrc.recording`` -> ``recording.id`` -> ``recording.gid`` (the MBID).

Memory is bounded to the ISRC table (the set of *referenced* recording ids), not
the whole recording table: pass 1 reads ``isrc`` and remembers which recording
ids it needs; pass 2 streams ``recording`` and keeps a gid only for those. One
ISRC may reference several recordings — every joinable pair is emitted (in ISRC-
dump order), so the multi-valued index downstream sees all candidates. A missing
input table yields an empty output rather than an error, mirroring the index's
own missing-file-is-empty stance.

## Artist leg: Spotify artist URI -> artist MBID (:func:`build_artist_mbid_tsv`)

Greenfield for #102 — nothing in the repo touched MB's artist/relationship
tables before this. Three raw tables are joined (same column-layout caveat as
above; unverified against a real dump in this environment, per the documented
MusicBrainz schema — see CONTEXT.md for the "no dump present" note):

- ``url``           : id(0), **gid(1)**, **url(2)**, edits_pending
- ``l_artist_url``  : id(0), link, **entity0(2)** (artist.id), **entity1(3)** (url.id), ...
- ``artist``        : id(0), **gid(1)**, name, ...

Join path: ``url.url`` matches a ``https://open.spotify.com/artist/<id>``
pattern -> ``url.id`` -> ``l_artist_url.entity1`` -> ``l_artist_url.entity0`` ->
``artist.id`` -> ``artist.gid`` (the MBID). The Spotify id extracted from the
URL is re-keyed to the ``spotify:artist:<id>`` URI form the app carries
end-to-end (:class:`~music_intel_mcp.models.ArtistRef.uri`), so the built TSV
is directly consumable by :class:`~music_intel_mcp.artist_identity.
MusicBrainzArtistUrlIndex` without another translation step downstream.

Three passes, memory bounded the same way as the track join: pass 1 streams
``url`` and keeps only rows matching the Spotify-artist URL shape (id -> uri);
pass 2 streams ``l_artist_url`` and keeps only entity0/entity1 pairs whose
``entity1`` was kept in pass 1; pass 3 streams ``artist`` and keeps a gid only
for the entity0 ids pass 2 needs. A missing input table yields an empty output,
same honest-empty stance as the track join.
"""

from __future__ import annotations

import re
from pathlib import Path

# 0-based column positions in the raw MusicBrainz TSV dump tables.
_ISRC_RECORDING_COL = 1
_ISRC_ISRC_COL = 2
_RECORDING_ID_COL = 0
_RECORDING_GID_COL = 1

_URL_ID_COL = 0
_URL_GID_COL = 1
_URL_URL_COL = 2
_L_ARTIST_URL_ENTITY0_COL = 2
_L_ARTIST_URL_ENTITY1_COL = 3
_ARTIST_ID_COL = 0
_ARTIST_GID_COL = 1

_HEADER = (
    "# ISRC\tMBID — built from the MusicBrainz dump by mb_dump.build_isrc_mbid_tsv.\n"
    "# One row per (ISRC, recording-MBID); an ISRC may repeat when it maps to several.\n"
)

_ARTIST_HEADER = (
    "# spotify:artist:<id>\tMBID — built from the MusicBrainz dump by\n"
    "# mb_dump.build_artist_mbid_tsv. One row per resolved artist (#102).\n"
)

_SPOTIFY_ARTIST_URL_RE = re.compile(r"open\.spotify\.com/artist/([A-Za-z0-9]+)")


def _iter_rows(path: Path):
    """Yield tab-split rows from a raw dump table, skipping blank lines. A missing
    file yields nothing (honest-empty, never a crash)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def build_isrc_mbid_tsv(
    isrc_dump: str | Path,
    recording_dump: str | Path,
    out_path: str | Path,
) -> int:
    """Build ``out_path`` (ISRC\\tMBID TSV) from the raw dump tables.

    Returns the number of ISRC->MBID pairs written. Referenced-but-unknown
    recordings (an ISRC pointing at a recording id absent from the recording
    table) are dropped rather than emitted with a blank MBID.
    """
    isrc_dump, recording_dump, out_path = Path(isrc_dump), Path(recording_dump), Path(out_path)

    # Pass 1 — collect (recording_id, isrc) pairs and the set of ids we need.
    pairs: list[tuple[str, str]] = []
    needed: set[str] = set()
    for cols in _iter_rows(isrc_dump):
        if len(cols) <= _ISRC_ISRC_COL:
            continue
        recording_id = cols[_ISRC_RECORDING_COL].strip()
        isrc = cols[_ISRC_ISRC_COL].strip()
        if not recording_id or not isrc:
            continue
        pairs.append((recording_id, isrc))
        needed.add(recording_id)

    # Pass 2 — stream the (much larger) recording table, keeping only needed gids.
    gid_by_recording: dict[str, str] = {}
    for cols in _iter_rows(recording_dump):
        if len(cols) <= _RECORDING_GID_COL:
            continue
        recording_id = cols[_RECORDING_ID_COL].strip()
        if recording_id not in needed:
            continue
        gid = cols[_RECORDING_GID_COL].strip()
        if gid:
            gid_by_recording[recording_id] = gid

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(_HEADER)
        for recording_id, isrc in pairs:
            gid = gid_by_recording.get(recording_id)
            if gid is None:
                continue
            out.write(f"{isrc}\t{gid}\n")
            written += 1
    return written


def build_artist_mbid_tsv(
    url_dump: str | Path,
    l_artist_url_dump: str | Path,
    artist_dump: str | Path,
    out_path: str | Path,
) -> int:
    """Build ``out_path`` (``spotify:artist:<id>``\\tMBID TSV) from the raw
    dump tables (#102).

    Returns the number of artist URI->MBID pairs written. A Spotify artist URL
    that never turns up as an ``l_artist_url`` relationship, or an artist row
    absent from the artist table, is dropped rather than emitted with a blank
    MBID — same "referenced-but-unknown" discipline as the track join.
    """
    url_dump = Path(url_dump)
    l_artist_url_dump = Path(l_artist_url_dump)
    artist_dump = Path(artist_dump)
    out_path = Path(out_path)

    # Pass 1 — collect url.id -> spotify artist id, for Spotify-artist URLs only.
    spotify_id_by_url: dict[str, str] = {}
    for cols in _iter_rows(url_dump):
        if len(cols) <= _URL_URL_COL:
            continue
        url_id = cols[_URL_ID_COL].strip()
        url = cols[_URL_URL_COL].strip()
        if not url_id or not url:
            continue
        match = _SPOTIFY_ARTIST_URL_RE.search(url)
        if match:
            spotify_id_by_url[url_id] = match.group(1)

    # Pass 2 — stream l_artist_url, keeping only rows whose url side matched.
    spotify_id_by_artist: dict[str, str] = {}
    needed_artists: set[str] = set()
    for cols in _iter_rows(l_artist_url_dump):
        if len(cols) <= _L_ARTIST_URL_ENTITY1_COL:
            continue
        artist_id = cols[_L_ARTIST_URL_ENTITY0_COL].strip()
        url_id = cols[_L_ARTIST_URL_ENTITY1_COL].strip()
        spotify_id = spotify_id_by_url.get(url_id)
        if artist_id and spotify_id:
            spotify_id_by_artist[artist_id] = spotify_id
            needed_artists.add(artist_id)

    # Pass 3 — stream the (much larger) artist table, keeping only needed gids.
    gid_by_artist: dict[str, str] = {}
    for cols in _iter_rows(artist_dump):
        if len(cols) <= _ARTIST_GID_COL:
            continue
        artist_id = cols[_ARTIST_ID_COL].strip()
        if artist_id not in needed_artists:
            continue
        gid = cols[_ARTIST_GID_COL].strip()
        if gid:
            gid_by_artist[artist_id] = gid

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(_ARTIST_HEADER)
        for artist_id, spotify_id in spotify_id_by_artist.items():
            gid = gid_by_artist.get(artist_id)
            if gid is None:
                continue
            out.write(f"spotify:artist:{spotify_id}\t{gid}\n")
            written += 1
    return written
