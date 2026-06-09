-- Shared metadata store — V0 schema (issue #60, decision f7a9fcbd).
--
-- Anonymous, multi-user-ready track-level facts. There are NO per-user columns
-- anywhere in this schema by design (no user_id, no played_at, no play-context):
-- listening history and RootProfile snapshots never leave the user's machine.
-- Every fact row carries fetched_at as its ~90-day TTL anchor.

-- Artist-level facts. Defined for the identity waterfall (#61) and future
-- artist enrichment; the V0 TrackMetadataRecord denormalises name onto tracks.
create table if not exists artists (
    id          uuid primary key default gen_random_uuid(),
    mbid        text unique,
    name        text not null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Canonical track identity + denormalised display facts. The primary key is our
-- canonical string id (mbid:… / isrc:… / spotify:… / name:…); see
-- shared_store.canonical_track_id.
create table if not exists tracks (
    id          text primary key,
    spotify_id  text,
    isrc        text,
    mbid        text,
    name        text not null,
    artist      text not null,
    artist_id   uuid references artists (id) on delete set null,
    fetched_at  timestamptz not null default now(),
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Identity-waterfall lookup indexes (#61).
create index if not exists tracks_spotify_id_idx on tracks (spotify_id);
create index if not exists tracks_isrc_idx on tracks (isrc);
create index if not exists tracks_mbid_idx on tracks (mbid);

-- Scalar audio descriptors (AcousticBrainz / future enrichers). One row per
-- track; all features nullable (a track may be only partially enriched).
create table if not exists audio_features (
    track_id        text primary key references tracks (id) on delete cascade,
    bpm             real,
    energy          real,
    valence         real,
    danceability    real,
    acousticness    real,
    instrumentalness real,
    source          text,
    fetched_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

-- Scene/cultural tags (Last.fm and similar). Many rows per track.
create table if not exists tags (
    id          bigint generated always as identity primary key,
    track_id    text not null references tracks (id) on delete cascade,
    tag         text not null,
    weight      real,
    source      text,
    fetched_at  timestamptz not null default now(),
    unique (track_id, tag, source)
);

create index if not exists tags_track_id_idx on tags (track_id);
