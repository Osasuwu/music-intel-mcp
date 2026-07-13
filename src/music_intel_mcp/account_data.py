"""Spotify **Account Data** explicit-preference ingestion (#97).

Distinct from the Extended Streaming History (``spotify_extended.py``, implicit
listening events): the *Account Data* export is what the user **told** Spotify —
liked tracks, followed/banned artists, saved albums (``YourLibrary.json``),
playlists with per-item add timestamps (``Playlist1.json``), and the per-artist
Marquee listener segments (``Marquee.json``). It becomes the explicit-signal
label layer for the anti-bubble recommender, persisted as a single current-state
document ``data/library.json`` (see :class:`~.models.Library`).

Grill decision 7e564052-f8e4-469e-a3d6-2182c26d11bb locked the schema. Key
discipline points enforced here:

- **Identity is report-only, never persisted.** Track signals reuse ``TrackRef``
  and store spotify_id + name/artist/album only; mbid/isrc stay None. Measured
  MBID coverage is a CLI report over the *existing* ``resolve()`` chain, computed
  at import time and thrown away (decision dddc4d90 — the identity cache is the
  single home for resolved ids).
- **Nothing dropped silently.** Unmapped ``YourLibrary`` keys (bannedTracks,
  shows, episodes, podcastChapters, other) and non-``spotify:track:`` playlist
  URIs (episodes, local tracks) are counted on :class:`AccountDataStats`.
- **Lossless.** Playlists carry description/lastModifiedDate/numberOfFollowers;
  Marquee segment strings are verbatim (not normalized to an enum).
- **No artist-MBID resolution** — explicitly out of scope (#102).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from .models import (
    AlbumRef,
    ArtistRef,
    Library,
    LibraryHeader,
    MarqueeEntry,
    Playlist,
    PlaylistItem,
    TrackRef,
)
from .shared_store import canonical_track_id

SOURCE = "spotify_account_data"

_YOUR_LIBRARY_FILE = "YourLibrary.json"
_PLAYLIST_GLOB = "Playlist*.json"
_MARQUEE_FILE = "Marquee.json"
_TRACK_URI_PREFIX = "spotify:track:"

# YourLibrary keys we map into typed signals — everything else in the object is
# drop-accounted (counted, never imported) so a schema drift is loud, not silent.
_MAPPED_LIBRARY_KEYS = frozenset({"tracks", "albums", "artists", "bannedArtists"})


@dataclass
class AccountDataStats:
    """What the Account Data load dropped, surfaced by the CLI rather than left
    silent. ``dropped_keys`` counts each unmapped ``YourLibrary`` key by its
    source name (bannedTracks/shows/episodes/podcastChapters/other, …);
    ``dropped_non_track_playlist_uris`` counts playlist items whose URI is not a
    ``spotify:track:`` (episodes, local files)."""

    dropped_keys: dict[str, int] = field(default_factory=dict)
    dropped_non_track_playlist_uris: int = 0

    def _drop_key(self, key: str, count: int) -> None:
        if count:
            self.dropped_keys[key] = self.dropped_keys.get(key, 0) + count

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped_keys.values()) + self.dropped_non_track_playlist_uris


class ParsedYourLibrary(BaseModel):
    """Intermediate result of parsing ``YourLibrary.json`` — the four typed
    signal lists. Composed into :class:`~.models.Library` by
    :func:`load_account_data`."""

    liked_tracks: list[TrackRef]
    followed_artists: list[ArtistRef]
    banned_artists: list[ArtistRef]
    saved_albums: list[AlbumRef]


def _str(value: object) -> str:
    """Normalise a JSON value to a stripped string (``None`` -> ``""``)."""
    return "" if value is None else str(value).strip()


def _spotify_id_from_uri(uri: str, prefix: str) -> str | None:
    """Return the id tail of ``spotify:<kind>:<id>``; None if the prefix differs."""
    return uri[len(prefix) :] if uri.startswith(prefix) else None


def load_your_library_file(
    path: str | Path, *, stats: AccountDataStats | None = None
) -> ParsedYourLibrary:
    """Parse ``YourLibrary.json`` into typed signal lists. Unmapped top-level
    keys (bannedTracks/shows/episodes/podcastChapters/other) are counted on
    ``stats`` — nothing outside :data:`_MAPPED_LIBRARY_KEYS` is imported, and
    every dropped key's element count is recorded."""
    stats = stats if stats is not None else AccountDataStats()
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    liked_tracks = [
        TrackRef(
            spotify_id=_spotify_id_from_uri(_str(row.get("uri")), _TRACK_URI_PREFIX) or None,
            name=_str(row.get("track")),
            artist=_str(row.get("artist")),
            album=_str(row.get("album")) or None,
        )
        for row in data.get("tracks", [])
    ]
    followed_artists = [
        ArtistRef(name=_str(row.get("name")), uri=_str(row.get("uri")) or None)
        for row in data.get("artists", [])
    ]
    banned_artists = [
        ArtistRef(name=_str(row.get("name")), uri=_str(row.get("uri")) or None)
        for row in data.get("bannedArtists", [])
    ]
    saved_albums = [
        AlbumRef(
            name=_str(row.get("album")),
            artist=_str(row.get("artist")) or None,
            uri=_str(row.get("uri")) or None,
        )
        for row in data.get("albums", [])
    ]

    # Drop-account every top-level key we did not map. Count list length where the
    # value is a list; a non-list unmapped key counts as 1 so its presence shows.
    for key, value in data.items():
        if key in _MAPPED_LIBRARY_KEYS:
            continue
        stats._drop_key(key, len(value) if isinstance(value, list) else 1)

    return ParsedYourLibrary(
        liked_tracks=liked_tracks,
        followed_artists=followed_artists,
        banned_artists=banned_artists,
        saved_albums=saved_albums,
    )


def _playlist_item(row: dict, stats: AccountDataStats) -> PlaylistItem | None:
    """Map one playlist item; return None (and count) for non-``spotify:track:``
    URIs (episodes, local files)."""
    track = row.get("track") or {}
    uri = _str(track.get("trackUri"))
    spotify_id = _spotify_id_from_uri(uri, _TRACK_URI_PREFIX)
    if spotify_id is None:
        stats.dropped_non_track_playlist_uris += 1
        return None
    return PlaylistItem(
        track=TrackRef(
            spotify_id=spotify_id,
            name=_str(track.get("trackName")),
            artist=_str(track.get("artistName")),
            album=_str(track.get("albumName")) or None,
        ),
        added_at=_str(row.get("addedDate")) or None,
    )


def load_playlists_file(
    path: str | Path, *, stats: AccountDataStats | None = None
) -> list[Playlist]:
    """Parse ``Playlist1.json`` into :class:`~.models.Playlist` objects. Playlist
    metadata (description/lastModifiedDate/numberOfFollowers) is preserved
    losslessly; non-track items are drop-accounted on ``stats``."""
    stats = stats if stats is not None else AccountDataStats()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    playlists: list[Playlist] = []
    for pl in data.get("playlists", []):
        items = [item for row in pl.get("items", []) if (item := _playlist_item(row, stats))]
        playlists.append(
            Playlist(
                name=_str(pl.get("name")),
                items=items,
                last_modified_date=_str(pl.get("lastModifiedDate")) or None,
                description=_str(pl.get("description")) or None,
                number_of_followers=pl.get("numberOfFollowers"),
            )
        )
    return playlists


def load_marquee_file(path: str | Path) -> list[MarqueeEntry]:
    """Parse ``Marquee.json`` (a JSON array of ``{artistName, segment}``) into
    verbatim :class:`~.models.MarqueeEntry` objects. The source carries no artist
    URI, and the segment string is stored as-is."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        MarqueeEntry(artist_name=_str(row.get("artistName")), segment=_str(row.get("segment")))
        for row in data
    ]


def load_account_data(
    directory: str | Path,
    *,
    now: datetime,
    exported_at: datetime | None = None,
    stats: AccountDataStats | None = None,
) -> Library:
    """Load a whole Account Data export directory into a :class:`~.models.Library`.

    Missing per-file inputs are tolerated (a partial export still imports what it
    has): only files present are parsed. ``now`` stamps ``imported_at``; pass
    ``stats`` to receive the drop ledger. Playlists come from every
    ``Playlist*.json`` (Spotify chunks large libraries into Playlist1/2/…)."""
    stats = stats if stats is not None else AccountDataStats()
    root = Path(directory)

    lib_path = root / _YOUR_LIBRARY_FILE
    parsed = (
        load_your_library_file(lib_path, stats=stats)
        if lib_path.exists()
        else ParsedYourLibrary(
            liked_tracks=[], followed_artists=[], banned_artists=[], saved_albums=[]
        )
    )

    playlists: list[Playlist] = []
    for pl_file in sorted(root.glob(_PLAYLIST_GLOB)):
        playlists.extend(load_playlists_file(pl_file, stats=stats))

    marquee_path = root / _MARQUEE_FILE
    marquee = load_marquee_file(marquee_path) if marquee_path.exists() else []

    return Library(
        header=LibraryHeader(source=SOURCE, exported_at=exported_at, imported_at=now),
        liked_tracks=parsed.liked_tracks,
        banned_artists=parsed.banned_artists,
        followed_artists=parsed.followed_artists,
        saved_albums=parsed.saved_albums,
        playlists=playlists,
        marquee=marquee,
    )


# --------------------------------------------------------------------------- #
# Diff — added/removed likes/bans/follows vs a prior library.json
# --------------------------------------------------------------------------- #


@dataclass
class LibraryDiff:
    """What changed between a prior ``library.json`` and a fresh import. Counts
    only — the identity sets are keyed by canonical track id (likes) and artist
    name casefolded (bans/follows), so a re-import with no change reports all
    zeros (idempotent)."""

    likes_added: int = 0
    likes_removed: int = 0
    bans_added: int = 0
    bans_removed: int = 0
    follows_added: int = 0
    follows_removed: int = 0


def _content_without_stamp(library: Library) -> dict:
    """The library's serialisable content minus the wall-clock ``imported_at`` —
    the equality basis for idempotent re-import."""
    data = library.model_dump(mode="json")
    data["header"].pop("imported_at", None)
    return data


def preserve_import_timestamp(prior: Library | None, new: Library) -> Library:
    """Return ``new`` with its ``imported_at`` carried over from ``prior`` when the
    two are otherwise byte-identical. Makes re-importing an unchanged export write
    a byte-identical ``library.json`` (the idempotency contract) despite
    ``imported_at`` being a fresh wall-clock each run. A changed export gets the
    new stamp."""
    if prior is not None and _content_without_stamp(prior) == _content_without_stamp(new):
        return new.model_copy(
            update={
                "header": new.header.model_copy(update={"imported_at": prior.header.imported_at})
            }
        )
    return new


def _artist_keys(artists: list[ArtistRef]) -> set[str]:
    return {a.uri or a.name.casefold() for a in artists}


def diff_libraries(old: Library | None, new: Library) -> LibraryDiff:
    """Compare explicit-signal membership. ``old=None`` (first import) reports
    everything in ``new`` as added. Tracks keyed by canonical id (the identity
    waterfall's stable string); artists by URI when present else casefolded
    name."""
    old_likes = {canonical_track_id(t) for t in old.liked_tracks} if old else set()
    new_likes = {canonical_track_id(t) for t in new.liked_tracks}
    old_bans = _artist_keys(old.banned_artists) if old else set()
    new_bans = _artist_keys(new.banned_artists)
    old_follows = _artist_keys(old.followed_artists) if old else set()
    new_follows = _artist_keys(new.followed_artists)

    return LibraryDiff(
        likes_added=len(new_likes - old_likes),
        likes_removed=len(old_likes - new_likes),
        bans_added=len(new_bans - old_bans),
        bans_removed=len(old_bans - new_bans),
        follows_added=len(new_follows - old_follows),
        follows_removed=len(old_follows - new_follows),
    )
