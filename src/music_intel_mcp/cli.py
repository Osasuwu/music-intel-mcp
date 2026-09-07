"""CLI surface. V0 exposes ``analyze``, ``resolve``, ``import-ifttt``,
``import-spotify``, ``import-youtube``, ``import-account``, ``capture-spike``,
``capture-loop``, ``build-mb-index``, ``build-artist-index``, and
``build-ab-index``.

    music-intel import-ifttt --from <dir> [--data-dir ./data]
    music-intel import-spotify --from <dir> [--data-dir ./data]
    music-intel import-youtube --from <watch-history.json> [--data-dir ./data]
    music-intel import-account --from <dir> [--data-dir ./data]
                        [--mb-index PATH] [--artist-index PATH]
    music-intel analyze --user-id petr [--data-dir ./data]
                        [--with-audio] [--with-scene]
                        [--with-musicbrainz-genre] [--with-discogs]
                        [--ab-index PATH] [--mb-index PATH]
                        [--shared-store local|supabase|memory]
                        [--shared-store-path PATH]
    music-intel resolve [--data-dir ./data] [--mb-index PATH] [--with-spotify]
    music-intel build-mb-index --isrc-dump PATH --recording-dump PATH --out PATH
    music-intel build-artist-index --url-dump PATH --l-artist-url-dump PATH
                        --artist-dump PATH --out PATH
    music-intel build-ab-index --out PATH [--data-dir ./data] [--raw-cache PATH]

``import-ifttt`` merges a directory of IFTTT Spotify ``.xlsx`` exports into the
per-user ``history.jsonl`` (dedup + idempotent re-import). ``import-spotify``
merges the official Spotify Extended Streaming History JSON export (#89), a richer
per-play source that supersedes the thin IFTTT rows (source-scoped, decision
23fcf92c). ``import-youtube`` merges a Google Takeout ``watch-history.json``
export (#164) for pilot participants whose history lives in YouTube Music
rather than Spotify; it carries no cross-source supersede (self-referential
only — re-import displaces just a prior run of this importer) since Takeout
always ships the complete history in one file. ``analyze`` loads that
history, runs the derivation engine, writes a RootProfile snapshot, and prints
the path + a one-line summary. ``resolve`` walks the history through the
spotify_id -> ISRC -> MBID identity waterfall and reports resolution coverage
(caching resolved identities for re-runs). ``resolve --with-spotify`` adds the
live ``spotify_id -> ISRC`` leg via the Spotify Web API for tracks that arrive
without an ISRC — credential-gated on ``SPOTIFY_CLIENT_ID`` + ``SPOTIFY_CLIENT_SECRET``,
prefetched in batches, cached to a local sidecar (#87 AC-A).

Enrichment is **off by default**: with no ``--with-*`` flag ``analyze`` produces
an honest-empty profile (the V0 baseline). ``--with-audio`` / ``--with-scene``
opt the run into the audio / scene derivation stages by constructing the
production source adapters (the env-pointed :class:`AcousticBrainzDump` and the
``LASTFM_API_KEY``-backed :class:`LastfmTagSource`) plus a shared metadata store,
then handing them to :func:`analyze`. ``--with-musicbrainz-genre``
(``MUSICBRAINZ_APP_CONTACT``-gated :class:`MusicBrainzGenreSource`, MBID-only)
and ``--with-discogs`` (``DISCOGS_TOKEN``-gated :class:`DiscogsStyleSource`) fill
the residual tag gap Last.fm leaves (#122): when more than one tag source flag
is set they combine via :class:`CompositeTagSource`, which tries each source in
order and keeps the first non-empty result per track — one ``enrich_tags`` pass,
no change to its single-source signature. The pipelines themselves are locked
(#63/#64); this module only wires the already-built adapters to the CLI.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from dotenv import find_dotenv, load_dotenv

from .account_data import (
    AccountDataStats,
    diff_libraries,
    load_account_data,
    preserve_import_timestamp,
)
from .account_history import (
    AccountHistoryStats,
    load_account_history_dir,
)
from .acousticbrainz import AcousticBrainzApiClient, build_acousticbrainz_index
from .analyzer import analyze
from .artist_identity import (
    ArtistIdentityCache,
    ArtistIdentityResolver,
    MusicBrainzArtistUrlIndex,
)
from .audio import AcousticBrainzDump, AudioFeatureSource
from .backfill_playlist import run_continuous_backfill
from .continuous_capture import run_continuous_capture
from .identity import IdentityCache, IdentityResolver, MusicBrainzIsrcIndex
from .ingest import IngestStats, dedup_events, load_ifttt_dir
from .live_identity import (
    AcoustIdApiSource,
    LiveIdentityResolver,
    LiveNegativeCache,
    TitleArtistYoutubeIndex,
)
from .live_pipeline import run_live_capture_spike
from .mb_dump import build_artist_mbid_tsv, build_isrc_mbid_tsv
from .replay_queue import DEFAULT_REPLAY_QUEUE_CAP, MIN_VALID_PLAYS
from .scene import (
    CompositeTagSource,
    DiscogsStyleSource,
    LastfmTagSource,
    MusicBrainzGenreSource,
    TagSource,
)
from .shared_store import (
    InMemorySharedStore,
    LocalSharedStore,
    SharedStore,
    SupabaseSharedStore,
)
from .spotify_api import SpotifyApiIsrcSource
from .spotify_extended import (
    SOURCE as SPOTIFY_SOURCE,
)
from .spotify_extended import (
    SUPERSEDES as SPOTIFY_SUPERSEDES,
)
from .spotify_extended import (
    SpotifyExtendedStats,
    load_replay_ledger_windows,
    load_spotify_extended_dir,
)
from .store import UserStore
from .youtube_music import (
    SUPERSEDES as YOUTUBE_SUPERSEDES,
)
from .youtube_music import (
    YoutubeMusicStats,
    load_watch_history_file,
)

# Canonical env-var names this CLI checks for *presence* (never value) before an
# enrichment run, so a missing credential fails fast with a clear message rather
# than mid-pipeline. They mirror the constants owned by ``scene``/``shared_store``
# (kept as literals here because they are the CLI's documented env contract).
_LASTFM_API_KEY_ENV = "LASTFM_API_KEY"
_MUSICBRAINZ_APP_CONTACT_ENV = "MUSICBRAINZ_APP_CONTACT"
_DISCOGS_TOKEN_ENV = "DISCOGS_TOKEN"
_SUPABASE_URL_ENV = "SUPABASE_URL"
_SUPABASE_KEY_ENV = "SUPABASE_KEY"
_SPOTIFY_CLIENT_ID_ENV = "SPOTIFY_CLIENT_ID"
_SPOTIFY_CLIENT_SECRET_ENV = "SPOTIFY_CLIENT_SECRET"


def plan_spotify_source(
    with_spotify: bool,
    env: Mapping[str, str],
    *,
    data_dir: str | None,
) -> tuple[SpotifyApiIsrcSource | None, list[str]]:
    """Resolve ``--with-spotify`` against ``env`` into a live ISRC source (#87 AC-A).

    Returns ``(source, errors)``. Without the flag it is ``(None, [])`` — the
    offline default, spotify-only tracks honestly flagged at the ``spotify`` rung.
    With the flag, ``SPOTIFY_CLIENT_ID`` and ``SPOTIFY_CLIENT_SECRET`` are checked
    for *presence* (never read or printed here — the source reads their values
    from the environment itself); a missing one comes back as an error so the
    caller aborts before any network. The source's JSONL cache lands under the
    data root (local-only, per history-never-leaves-the-machine)."""
    if not with_spotify:
        return None, []
    missing = [v for v in (_SPOTIFY_CLIENT_ID_ENV, _SPOTIFY_CLIENT_SECRET_ENV) if not env.get(v)]
    if missing:
        return None, [f"--with-spotify needs {' and '.join(missing)} set."]
    return SpotifyApiIsrcSource(data_root=data_dir), []


def audio_identity_metrics(diag: Mapping[str, int]) -> dict[str, float]:
    """Derive the #87 AC-E honest-measurement rates from the audio enrichment
    buckets (``generated_from.enrichment_diagnostics["audio"]``).

    - ``mbid_coverage`` — share of considered tracks that carry an MBID (identity
      reach: everything except the ``no_mbid`` bucket). The ceiling the ISRC->MBID
      chain can reach on this library.
    - ``features_given_mbid`` — of the MBID-bearing tracks, the share the features
      source actually covered (``enriched`` + ``already_present``). This is
      P(features|MBID): how far a *ready-made* audio dump gets once identity is
      solved.

    Each is ``0.0`` when its denominator is empty — an honest zero, never a
    divide-by-zero. The rates are diagnostic, not thresholds; nothing gates on
    them (AC-F validates thresholds, it does not calibrate against these)."""
    total = sum(diag.values())
    with_mbid = total - diag.get("no_mbid", 0)
    with_features = diag.get("enriched", 0) + diag.get("already_present", 0)
    return {
        "mbid_coverage": (with_mbid / total) if total else 0.0,
        "features_given_mbid": (with_features / with_mbid) if with_mbid else 0.0,
    }


def _build_shared_store(kind: str, path: str | None = None) -> SharedStore:
    """Construct the requested :class:`SharedStore`.

    - ``local`` (V0 default) — file-backed persistent cache at ``path`` (default
      ``<data root>/shared_cache.jsonl``); no creds, cross-run caching.
    - ``memory`` — ephemeral single-run store (no persistence, no creds).
    - ``supabase`` — the multi-user cloud cache (lazy client; no network until
      first read/write). Deferred to V2+ (decision 813de040).
    """
    if kind == "memory":
        return InMemorySharedStore()
    if kind == "supabase":
        return SupabaseSharedStore()
    return LocalSharedStore(path)


def plan_enrichment(
    args: argparse.Namespace,
    env: Mapping[str, str],
) -> tuple[SharedStore | None, AudioFeatureSource | None, TagSource | None, list[str]]:
    """Resolve the ``--with-*`` flags against ``env`` into enrichment sources.

    Returns ``(shared_store, audio_source, tag_source, errors)``. With no
    ``--with-*`` flag every source is ``None`` and ``errors`` is empty — the
    honest-empty baseline, unchanged. When a flag is set, required credentials
    are checked for *presence* (never read or printed); any missing one is
    appended to ``errors`` and all sources come back ``None`` so the caller can
    abort before touching history or the store. No network/dump access happens
    here — adapters are constructed lazily and only read when ``analyze`` runs.
    """
    errors: list[str] = []
    with_musicbrainz_genre = getattr(args, "with_musicbrainz_genre", False)
    with_discogs = getattr(args, "with_discogs", False)
    if not (args.with_audio or args.with_scene or with_musicbrainz_genre or with_discogs):
        return None, None, None, errors

    if args.shared_store == "supabase" and not (
        env.get(_SUPABASE_URL_ENV) and env.get(_SUPABASE_KEY_ENV)
    ):
        errors.append(
            f"--shared-store supabase needs {_SUPABASE_URL_ENV} and {_SUPABASE_KEY_ENV} set "
            "(or use --shared-store memory for an ephemeral local run)."
        )
    if args.with_scene and not env.get(_LASTFM_API_KEY_ENV):
        errors.append(f"--with-scene needs {_LASTFM_API_KEY_ENV} set.")
    if with_musicbrainz_genre and not env.get(_MUSICBRAINZ_APP_CONTACT_ENV):
        errors.append(f"--with-musicbrainz-genre needs {_MUSICBRAINZ_APP_CONTACT_ENV} set.")
    if with_discogs and not env.get(_DISCOGS_TOKEN_ENV):
        errors.append(f"--with-discogs needs {_DISCOGS_TOKEN_ENV} set.")
    if errors:
        return None, None, None, errors

    shared_store = _build_shared_store(args.shared_store, getattr(args, "shared_store_path", None))
    audio_source = AcousticBrainzDump(path=args.ab_index) if args.with_audio else None

    tag_sources: list[TagSource] = []
    if args.with_scene:
        tag_sources.append(LastfmTagSource())
    if with_musicbrainz_genre:
        tag_sources.append(MusicBrainzGenreSource())
    if with_discogs:
        tag_sources.append(DiscogsStyleSource())
    tag_source: TagSource | None
    if len(tag_sources) > 1:
        tag_source = CompositeTagSource(tag_sources)
    elif tag_sources:
        tag_source = tag_sources[0]
    else:
        tag_source = None
    return shared_store, audio_source, tag_source, errors


def _cmd_import_ifttt(args: argparse.Namespace) -> int:
    store = UserStore(root=args.data_dir)
    before = store.load_history()
    stats = IngestStats()
    imported = load_ifttt_dir(args.source, stats=stats)
    # Merge existing-first so events from other sources survive, then dedup so a
    # re-import over the same dir is idempotent.
    merged = dedup_events([*before, *imported])
    store.replace_history(merged)

    added = len(merged) - len(before)
    print(f"imported {len(imported)} IFTTT plays from {args.source}")
    print(f"  history.jsonl: {len(before)} -> {len(merged)} events (+{added} new after dedup)")
    if stats.total_skipped:
        # Surfaced, never silent: a large unparseable count signals export drift.
        print(
            f"  skipped {stats.total_skipped} rows "
            f"(empty={stats.skipped_empty} no-identity={stats.skipped_no_identity} "
            f"unparseable-timestamp={stats.skipped_unparseable})"
        )
        if stats.unparseable_samples:
            print(f"    unparseable e.g.: {stats.unparseable_samples}")
    return 0


def _cmd_import_spotify(args: argparse.Namespace) -> int:
    store = UserStore(root=args.data_dir)
    before = store.load_history()
    stats = SpotifyExtendedStats()
    # #167 AC2: rows falling inside a replay-loop window are tagged
    # agent-originated (not dropped) rather than imported as organic history.
    from .replay_capture import replay_ledger_path

    ledger_windows = load_replay_ledger_windows(replay_ledger_path(store))
    imported = load_spotify_extended_dir(args.source, stats=stats, ledger_windows=ledger_windows)
    # Source-scoped supersede (decision 23fcf92c): the authoritative Spotify export
    # replaces the thin IFTTT rows and any prior run of this importer — a play
    # logged by both is the *same* play — while events from every other source are
    # preserved untouched. Dedup after so a re-import over the same dir is
    # idempotent.
    kept = [e for e in before if e.source not in SPOTIFY_SUPERSEDES]
    merged = dedup_events([*kept, *imported])

    # #93 guardrail: supersede assumes the new export is a full-history superset
    # of whatever spotify_extended history it is about to replace (true for a
    # normal Spotify re-export, which always re-ships the full history) — but
    # nothing enforces that at import time. A partial, older, or wrong export
    # file would silently drop real history via replace_history's overwrite.
    # Warn (never block) when a play about to be dropped predates the new
    # import's earliest play — that's the signature of history the new import
    # doesn't actually cover.
    if imported:
        new_earliest = min(e.played_at for e in imported)
        stale_spotify = [
            e for e in before if e.source == SPOTIFY_SOURCE and e.played_at < new_earliest
        ]
        if stale_spotify:
            print(
                f"  WARNING: {len(stale_spotify)} prior Spotify Extended plays predate "
                f"this import's earliest play ({new_earliest.isoformat()}) and are about "
                "to be dropped. This export may not be a full-history superset of what "
                "was previously imported — double-check the export file before trusting "
                "the result."
            )

    store.replace_history(merged)

    superseded = len(before) - len(kept)
    print(f"imported {len(imported)} Spotify Extended plays from {args.source}")
    print(
        f"  history.jsonl: {len(before)} -> {len(merged)} events "
        f"(superseded {superseded} from {sorted(SPOTIFY_SUPERSEDES)})"
    )
    if stats.total_skipped:
        # Surfaced, never silent: the lossless projection drops only non-audio-track
        # rows and no-identity rows; a large unparseable count signals export drift.
        print(
            f"  skipped {stats.total_skipped} non-track/unplaceable rows "
            f"(episode={stats.skipped_episode} audiobook={stats.skipped_audiobook} "
            f"no-identity={stats.skipped_no_identity} unparseable-ts={stats.skipped_unparseable})"
        )
        if stats.unparseable_samples:
            print(f"    unparseable e.g.: {stats.unparseable_samples}")
    if stats.tagged_agent_originated:
        # #167 AC3: rows inside a replay-loop window (#167 AC1/AC2) were tagged
        # agent-originated, not dropped -- surfaced separately from the skip counts.
        print(f"  ledger-tagged {stats.tagged_agent_originated} rows as agent-originated")
    return 0


def _cmd_import_youtube(args: argparse.Namespace) -> int:
    store = UserStore(root=args.data_dir)
    before = store.load_history()
    stats = YoutubeMusicStats()
    imported = load_watch_history_file(args.source, stats=stats)
    # Self-referential supersede (#164): no prior source overlaps YouTube Music
    # watch history, so this only ever displaces a prior run of this importer —
    # unlike import-spotify there is no cross-source displacement, so no #93-style
    # not-a-superset guardrail is needed: Takeout always ships the complete
    # history in one file, never a partial export.
    kept = [e for e in before if e.source not in YOUTUBE_SUPERSEDES]
    merged = dedup_events([*kept, *imported])
    store.replace_history(merged)

    superseded = len(before) - len(kept)
    print(f"imported {len(imported)} YouTube Music plays from {args.source}")
    print(
        f"  history.jsonl: {len(before)} -> {len(merged)} events "
        f"(superseded {superseded} from {sorted(YOUTUBE_SUPERSEDES)})"
    )
    if stats.total_skipped:
        print(
            f"  skipped {stats.total_skipped} non-music/unplaceable rows "
            f"(non-music={stats.skipped_non_music} removed={stats.skipped_removed} "
            f"no-video-id={stats.skipped_no_video_id} unparseable-ts={stats.skipped_unparseable})"
        )
        if stats.unparseable_samples:
            print(f"    unparseable e.g.: {stats.unparseable_samples}")
    return 0


def _cmd_import_account(args: argparse.Namespace) -> int:
    """Import the Spotify **Account Data** export (#97) into the explicit-signal
    ``library.json``. Re-import replaces the file (idempotent). The report
    surfaces signal counts, the prior-vs-new diff, the drop ledger (nothing
    silent), and a **report-only** measured MBID coverage over liked + playlist
    tracks (computed via the existing resolve() chain, never persisted —
    decision dddc4d90)."""
    store = UserStore(root=args.data_dir)
    prior = store.load_library()

    stats = AccountDataStats()
    library = load_account_data(args.source, now=datetime.now(UTC), stats=stats)
    diff = diff_libraries(prior, library)
    # Idempotency: an unchanged export must re-write byte-identically, so carry the
    # prior imported_at forward when nothing else changed (the wall-clock now would
    # otherwise make every re-import differ).
    library = preserve_import_timestamp(prior, library)
    store.write_library(library)

    n_playlist_tracks = sum(len(pl.items) for pl in library.playlists)
    print(f"imported Spotify Account Data from {args.source} -> {store.library_path}")
    print(
        f"  signals: likes={len(library.liked_tracks)} "
        f"banned_artists={len(library.banned_artists)} "
        f"followed_artists={len(library.followed_artists)} "
        f"saved_albums={len(library.saved_albums)} "
        f"playlists={len(library.playlists)} ({n_playlist_tracks} tracks) "
        f"marquee={len(library.marquee)}"
    )
    print(
        f"  diff vs prior: likes +{diff.likes_added}/-{diff.likes_removed} "
        f"bans +{diff.bans_added}/-{diff.bans_removed} "
        f"follows +{diff.follows_added}/-{diff.follows_removed}"
    )
    # Drop-accounting, never silent: unmapped YourLibrary keys (bannedTracks/shows/
    # episodes/podcastChapters/other) and non-spotify:track: playlist URIs.
    key_ledger = " ".join(f"{k}={v}" for k, v in sorted(stats.dropped_keys.items())) or "none"
    print(
        f"  dropped (accounted, not imported): {stats.total_dropped} total — "
        f"unmapped-library-keys[{key_ledger}] "
        f"non-track-playlist-uris={stats.dropped_non_track_playlist_uris}"
    )

    # Report-only MBID coverage over liked + playlist tracks through the existing
    # waterfall. Measured and printed, then discarded — the identity cache is the
    # single home for resolved ids (decision dddc4d90); library.json stores none.
    resolver = _build_resolver(args)
    tracks = [*library.liked_tracks, *(it.track for pl in library.playlists for it in pl.items)]
    report = resolver.resolve_all(tracks)
    print(
        f"  MBID coverage (report-only, over {report.n_unique} unique liked+playlist "
        f"tracks): {report.mbid_coverage:.2f} ({report.counts['mbid']}/{report.n_unique})"
    )

    # Artist-level MBID coverage (#102), same report-only/never-persisted discipline.
    # No measured baseline exists for this number yet (unlike the track-level
    # figure) — see CONTEXT.md and the artist_identity module docstring.
    artist_resolver = _build_artist_resolver(args)
    artists = [*library.followed_artists, *library.banned_artists]
    artist_report = artist_resolver.resolve_all(artists, library.marquee)
    ac = artist_report.counts
    print(
        f"  artist MBID coverage (report-only, over {artist_report.n_unique} unique "
        f"followed+banned+Marquee artists): {artist_report.mbid_coverage:.2f} "
        f"({ac['mbid']}/{artist_report.n_unique}) — no measured baseline yet, see CONTEXT.md"
    )
    return 0


def _cmd_import_account_history(args: argparse.Namespace) -> int:
    """Import the Spotify **Account Data** ``StreamingHistory_music_*.json``
    play log (#103) into ``history.jsonl``, extending past the newest existing
    ``spotify_extended`` play (time-cutoff dedup — Account Data carries no
    per-play context, so it must never shadow the richer ESH rows it overlaps
    with). Rows are resolved onto an existing ``spotify:<id>`` canonical id via
    a name/artist lookup built from the current ``spotify_extended`` history
    where unambiguous; unresolved rows fall back to the ``name:`` rung."""
    store = UserStore(root=args.data_dir)
    before = store.load_history()
    stats = AccountHistoryStats()
    imported = load_account_history_dir(args.source, existing_events=before, stats=stats)
    merged = dedup_events([*before, *imported])
    store.replace_history(merged)

    print(f"imported {len(imported)} Spotify Account Data plays from {args.source}")
    print(f"  history.jsonl: {len(before)} -> {len(merged)} events")
    print(
        f"  resolved-to-spotify-id={stats.resolved_via_spotify_lookup} "
        f"ambiguous-lookup-keys={stats.ambiguous_lookup_keys}"
    )
    if stats.total_skipped:
        print(
            f"  dropped {stats.total_skipped} rows (accounted, not imported): "
            f"before-cutoff={stats.dropped_before_cutoff} "
            f"no-identity={stats.skipped_no_identity} "
            f"unparseable-timestamp={stats.skipped_unparseable}"
        )
        if stats.unparseable_samples:
            print(f"    unparseable e.g.: {stats.unparseable_samples}")
    return 0


def _build_resolver(
    args: argparse.Namespace,
    audio_source: AudioFeatureSource | None = None,
) -> IdentityResolver:
    """Construct the identity resolver for an audio-enrichment run (the P9
    bridge, #87). The MusicBrainz ISRC->MBID index + a local IdentityCache are
    wired here; the live spotify_id->ISRC source (AC-A) plugs into the
    ``spotify_source`` seam once credential-gated. Cheap to build — the index is
    read lazily on first lookup, the cache is filesystem-backed.

    When an ``audio_source`` is passed, its presence check becomes the
    ``ab_covered`` predicate so a multi-valued ISRC disambiguates toward the
    recording that actually has AcousticBrainz features (#87 AC-B). The *same*
    dump instance is shared with the enrichment stage, so it is loaded once."""
    index = MusicBrainzIsrcIndex(path=getattr(args, "mb_index", None))
    cache = IdentityCache(root=args.data_dir)
    ab_covered = (
        (lambda mbid: audio_source.lookup(mbid) is not None) if audio_source is not None else None
    )
    return IdentityResolver(index, cache=cache, ab_covered=ab_covered)


def _build_live_resolver(args: argparse.Namespace) -> LiveIdentityResolver:
    """Construct the live-capture identity waterfall (#139 AC2): AcoustID ->
    Spotify search -> ISRC -> MBID -> MB name search -> name key. Every
    network-backed rung is optional and gated on its own credential — an
    unset ``ACOUSTID_API_KEY``/Spotify creds simply skips that rung and the
    waterfall falls through, per the fragmentation-over-false-merge bias
    (AC3). The ISRC->MBID index and the negative cache are always local and
    cheap, so those wire in unconditionally. The #170 AC4/AC5 youtube-history
    rung is built from this participant's own imported Takeout history --
    also always local, so it wires in unconditionally alongside the other
    local rungs."""
    from .spotify_api import SpotifyApiIsrcSource, SpotifySearchApiSource

    acoustid_source = AcoustIdApiSource() if os.environ.get("ACOUSTID_API_KEY") else None
    spotify_search = None
    spotify_isrc = None
    if os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET"):
        isrc_source = SpotifyApiIsrcSource()
        spotify_search = SpotifySearchApiSource(isrc_source=isrc_source)
        spotify_isrc = isrc_source

    store = UserStore(root=args.data_dir)
    youtube_history_index = TitleArtistYoutubeIndex.from_events(store.load_history())

    return LiveIdentityResolver(
        acoustid_source=acoustid_source,
        spotify_search=spotify_search,
        spotify_isrc=spotify_isrc,
        isrc_index=MusicBrainzIsrcIndex(path=getattr(args, "mb_index", None)),
        negative_cache=LiveNegativeCache(root=args.data_dir),
        youtube_history_index=youtube_history_index,
    )


def _build_artist_resolver(args: argparse.Namespace) -> ArtistIdentityResolver:
    """Construct the artist-identity resolver (#102), mirroring
    :func:`_build_resolver`. No live name-match source is wired at V0 — see
    the artist_identity module docstring for why."""
    index = MusicBrainzArtistUrlIndex(path=getattr(args, "artist_index", None))
    cache = ArtistIdentityCache(root=args.data_dir)
    return ArtistIdentityResolver(index, cache=cache)


def _cmd_analyze(args: argparse.Namespace) -> int:
    shared_store, audio_source, tag_source, errors = plan_enrichment(args, os.environ)
    if errors:
        for err in errors:
            print(f"error: {err}")
        return 2

    # The resolver bridges raw ingest ids -> MBID so the audio enricher can join
    # (P9, #87). Only built for --with-audio: audio is the stage that needs the
    # MBID; scene keys on name/artist and is left on its raw-identity path. The
    # audio source is shared in so a multi-valued ISRC disambiguates toward an
    # AB-covered recording (AC-B) and the dump loads once.
    resolver = _build_resolver(args, audio_source) if args.with_audio else None

    user_store = UserStore(root=args.data_dir)
    events = user_store.load_history()
    # #165 AC4: node tier never asks a phone-only participant for a separate
    # identifier -- user_id defaults to the participant root's own name.
    user_id = args.user_id or user_store.root.name
    profile = analyze(
        events,
        user_id=user_id,
        shared_store=shared_store,
        audio_source=audio_source,
        tag_source=tag_source,
        resolver=resolver,
    )
    path = user_store.write_profile(profile)

    gf = profile.generated_from
    cov = gf.coverage_per_category
    print(f"snapshot: {path}")
    print(
        f"  events={gf.n_events} unique_tracks={gf.n_unique_tracks} "
        f"span_days={gf.history_span_days} sources={','.join(gf.data_sources) or '-'}"
    )
    print(
        f"  coverage: audio={cov.get('audio', 0.0):.2f} "
        f"scene={cov.get('scene', 0.0):.2f} temporal={cov.get('temporal', 0.0):.2f}"
    )
    print(
        f"  roots={len(profile.roots)} tendencies={len(profile.tendencies)} "
        f"epochs={len(profile.epochs)} maturity={profile.model_maturity}"
    )
    # #87 AC-E honest measurement, surfaced on the run itself: split coverage into
    # its two independent legs — how far identity resolution reached (mbid_coverage)
    # and, given an MBID, how far the ready-made features dump reached
    # (p(features|mbid)). Both derive from the persisted enrichment buckets, so the
    # numbers a saved profile carries are exactly what prints here.
    audio_diag = gf.enrichment_diagnostics.get("audio")
    if audio_diag is not None:
        m = audio_identity_metrics(audio_diag)
        total = sum(audio_diag.values())
        with_mbid = total - audio_diag.get("no_mbid", 0)
        with_features = audio_diag["enriched"] + audio_diag["already_present"]
        print(
            f"  audio identity: mbid_coverage={m['mbid_coverage']:.2f} ({with_mbid}/{total}) "
            f"p(features|mbid)={m['features_given_mbid']:.2f} ({with_features}/{with_mbid})"
        )
        print(
            f"    buckets: enriched={audio_diag['enriched']} "
            f"already_present={audio_diag['already_present']} "
            f"missing_features={audio_diag['missing_features']} no_mbid={audio_diag['no_mbid']}"
        )
    # Honest diagnostic: an enrichment flag that yielded zero coverage means the
    # source had nothing (dump not installed / no MBIDs / Last.fm misses), not a
    # bug — surface it so a real run isn't silently empty.
    if args.with_audio and cov.get("audio", 0.0) == 0.0:
        print(
            "  note: audio coverage 0 — check the AcousticBrainz dump "
            "(--ab-index / ACOUSTICBRAINZ_FEATURES_INDEX / ACOUSTICBRAINZ_DUMP_DIR) "
            "and that tracks resolve to MBIDs (run `resolve` first)."
        )
    if args.with_scene and cov.get("scene", 0.0) == 0.0:
        print("  note: scene coverage 0 — Last.fm returned no tags for any track.")
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    store = UserStore(root=args.data_dir)
    events = store.load_history()

    spotify_source, errors = plan_spotify_source(
        getattr(args, "with_spotify", False), os.environ, data_dir=args.data_dir
    )
    if errors:
        for err in errors:
            print(f"error: {err}")
        return 2

    index = MusicBrainzIsrcIndex(path=args.mb_index)
    cache = IdentityCache(root=args.data_dir)
    resolver = IdentityResolver(index, cache=cache, spotify_source=spotify_source)

    if spotify_source is not None:
        # Batch the spotify_id -> ISRC prefetch (<=50/call) before the per-track
        # waterfall, so resolution reads the local cache instead of firing one API
        # call per unresolved track. Already-cached ids (incl. a resumed run) skip.
        prefetched = spotify_source.warm(e.track.spotify_id for e in events if e.track.spotify_id)
        print(f"spotify: prefetched {prefetched} new spotify_id->ISRC lookups")

    report = resolver.resolve_all([e.track for e in events])

    c = report.counts
    print(
        f"resolved {c['mbid']}/{report.n_unique} unique tracks to MBID "
        f"(coverage={report.mbid_coverage:.2f})"
    )
    print(f"  levels: mbid={c['mbid']} isrc={c['isrc']} spotify={c['spotify']} name={c['name']}")
    if report.unresolved:
        print(f"  unresolved (flagged, not dropped): {len(report.unresolved)}")
    return 0


def _cmd_crosswalk(args: argparse.Namespace) -> int:
    """#178: re-run the spotify->ISRC->MBID resolution over history-only keys
    using only already-materialized indexes (MusicBrainz ISRC dump + the local
    Spotify ISRC cache) -- no live API call at run time."""
    from .metadata_crosswalk import run_metadata_crosswalk
    from .spotify_api import SpotifyApiIsrcSource

    index = MusicBrainzIsrcIndex(path=args.mb_index)
    spotify_source = SpotifyApiIsrcSource(data_root=args.data_dir)

    result = run_metadata_crosswalk(
        args.data_dir,
        isrc_index=index,
        spotify_isrc_lookup=spotify_source.lookup_cached,
    )

    print(f"crosswalk: wrote {len(result.aliases_written)} alias(es)")
    if result.ambiguous:
        print(f"  ambiguous (not aliased): {len(result.ambiguous)}")
        for match in result.ambiguous:
            print(f"    spotify:{match.spotify_id} isrc={match.isrc} mbids={match.mbids}")
    return 0


def _cmd_capture_spike(args: argparse.Namespace) -> int:
    """Run one pass of the WASAPI per-process loopback capture spike (#124):
    now-playing -> identity -> capture -> librosa/onnxruntime inference ->
    local-only store. Windows-only; needs the ``live-capture`` extra
    (``pip install -e .[live-capture]``) and the SMTC session + target app
    actually playing — this is the manual/interactive entry point for the
    live smoke session, not something CI exercises."""
    from .capture import WasapiProcessLoopbackCapture
    from .inference import DiscogsEffnetOnnxModel, MtgJamendoClassifier
    from .nowplaying import (
        DEFAULT_DROP_LOG_FILENAME,
        InMemoryNowPlayingSource,
        SmtcNowPlayingSource,
    )
    from .store import resolve_data_root

    live_resolver = _build_live_resolver(args)
    drop_log_path = resolve_data_root(args.data_dir) / DEFAULT_DROP_LOG_FILENAME
    now_playing = SmtcNowPlayingSource(
        identity_resolver=live_resolver, drop_log_path=drop_log_path
    ).current()
    if now_playing is None:
        print("nothing is currently playing (SMTC reports no active session)")
        return 1
    if now_playing.process_id is None:
        print(
            f"could not resolve a process id for '{now_playing.app_id}' — "
            "cannot scope loopback capture"
        )
        return 1

    print(f"now playing: {now_playing.artist} - {now_playing.title} (pid={now_playing.process_id})")
    capture = WasapiProcessLoopbackCapture(target_pid=now_playing.process_id)
    store = UserStore(root=args.data_dir)

    from .live_pipeline import live_capture_journal_path

    result = run_live_capture_spike(
        duration_s=args.duration,
        now_playing_source=InMemoryNowPlayingSource(now_playing),
        live_identity_resolver=live_resolver,
        capture=capture,
        embedding_model=DiscogsEffnetOnnxModel(),
        classifier=MtgJamendoClassifier(),
        store=store,
        journal_path=live_capture_journal_path(store),
    )
    if result is None:
        print("nothing playing by the time capture ran")
        return 1
    if result.outcome in ("silent", "short"):
        print(f"discarded capture ({result.outcome}): no audio-analysis file written")
        return 1

    print(f"identity: mbid={result.identity.mbid} level={result.identity.level}")
    if result.inference is not None:
        print(f"embedding: shape={result.inference.embedding.shape}")
        print(f"tags: {result.inference.tags}")
    print(f"wrote local-only analysis to {result.analysis_path}")
    return 0


def _cmd_capture_loop(args: argparse.Namespace) -> int:
    """Run the continuous capture loop (#136): poll SMTC now-playing, capture
    + analyze + store each new track, keep going until Ctrl+C. Same
    Windows-only / ``live-capture`` extra requirements as ``capture-spike``,
    but meant to be left running unattended rather than run once."""
    from .capture import WasapiProcessLoopbackCapture
    from .inference import DiscogsEffnetOnnxModel, MtgJamendoClassifier
    from .nowplaying import DEFAULT_DROP_LOG_FILENAME, SmtcNowPlayingSource

    live_resolver = _build_live_resolver(args)
    store = UserStore(root=args.data_dir)
    embedding_model = DiscogsEffnetOnnxModel()
    classifier = MtgJamendoClassifier()

    def capture_factory(now_playing):
        return WasapiProcessLoopbackCapture(target_pid=now_playing.process_id)

    def on_result(now_playing, result) -> None:
        if result is None:
            print(f"skipped: {now_playing.artist} - {now_playing.title} (nothing to capture)")
            return
        if result.outcome in ("silent", "short"):
            print(
                f"discarded ({result.outcome}): {now_playing.artist} - {now_playing.title} "
                "(no audio-analysis file written)"
            )
            return
        print(
            f"captured: {now_playing.artist} - {now_playing.title} "
            f"(mbid={result.identity.mbid}) -> {result.analysis_path}"
        )

    def on_error(now_playing, exc: Exception) -> None:
        print(f"error on '{now_playing.artist} - {now_playing.title}': {exc}")

    print(f"capture-loop running (poll every {args.poll_interval}s, Ctrl+C to stop)...")
    try:
        run_continuous_capture(
            now_playing_source=SmtcNowPlayingSource(
                identity_resolver=live_resolver,
                drop_log_path=store.root / DEFAULT_DROP_LOG_FILENAME,
            ),
            live_identity_resolver=live_resolver,
            capture_factory=capture_factory,
            embedding_model=embedding_model,
            classifier=classifier,
            store=store,
            capture_duration_s=args.duration,
            poll_interval_s=args.poll_interval,
            on_result=on_result,
            on_error=on_error,
        )
    except KeyboardInterrupt:
        print("capture-loop stopped.")
    return 0


def _cmd_build_mb_index(args: argparse.Namespace) -> int:
    """Distil the raw MusicBrainz ``isrc`` + ``recording`` dump tables into the
    compact ISRC->MBID TSV the resolver reads (#87 AC-B). Offline, one-off; the
    raw dump lives outside the repo (env-pointed, never committed)."""
    n = build_isrc_mbid_tsv(args.isrc_dump, args.recording_dump, args.out)
    print(f"wrote {n} ISRC->MBID pairs to {args.out}")
    if n == 0:
        print(
            "  note: 0 pairs — check the dump paths (raw MusicBrainz `isrc` and "
            "`recording` tables); a missing table yields an empty index."
        )
    return 0


def _cmd_build_artist_index(args: argparse.Namespace) -> int:
    """Distil the raw MusicBrainz ``url`` + ``l_artist_url`` + ``artist`` dump
    tables into the compact spotify-artist-URI->MBID TSV the artist resolver
    reads (#102). Offline, one-off; the raw dump lives outside the repo
    (env-pointed, never committed)."""
    n = build_artist_mbid_tsv(args.url_dump, args.l_artist_url_dump, args.artist_dump, args.out)
    print(f"wrote {n} artist URI->MBID pairs to {args.out}")
    if n == 0:
        print(
            "  note: 0 pairs — check the dump paths (raw MusicBrainz `url`, "
            "`l_artist_url`, and `artist` tables); a missing table yields an empty index."
        )
    return 0


def _cmd_build_ab_index(args: argparse.Namespace) -> int:
    """Fetch AcousticBrainz features for the MBIDs the identity chain already
    resolved and build the features JSONL the audio pipeline reads (#87 AC-E).

    The MBID source is the local IdentityCache (no re-run of the waterfall); the
    fetch is resumable through a raw-scalar sidecar, so a run cut short by the
    harness resumes cheaply. Point ``analyze --ab-index`` at ``--out`` afterwards."""
    cache = IdentityCache(root=args.data_dir)
    mbids = cache.resolved_mbids()
    if not mbids:
        print(
            f"no resolved MBIDs in {cache.cache_dir} — run `resolve` first "
            "(the identity cache is the MBID source for this build)."
        )
        return 0

    raw_cache = args.raw_cache or str(cache.root / "acousticbrainz_raw.jsonl")
    print(f"building AcousticBrainz features for {len(mbids)} resolved MBIDs -> {args.out}")
    client = AcousticBrainzApiClient()
    report = build_acousticbrainz_index(mbids, client, out_path=args.out, raw_cache_path=raw_cache)
    print(
        f"  fetched={report.fetched} hits={report.hits} misses={report.misses} "
        f"(cached total={report.total_cached})"
    )
    print(f"  wrote {report.written} feature rows to {args.out}")
    if report.total_cached:
        hit_rate = report.cached_hits / report.total_cached
        print(f"  p(features|mbid) so far = {hit_rate:.2f}")
    return 0


def _wait_for_oauth_callback(*, port: int, expected_state: str, timeout: float = 120.0) -> str:
    """Blocks until Spotify's authorization redirect hits
    ``http://127.0.0.1:<port>/callback``, then returns the authorization
    code. A one-shot stdlib HTTP server — no browser-automation dependency."""
    import queue
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    received: queue.Queue = queue.Queue(maxsize=1)

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib-mandated method name
            params = parse_qs(urlparse(self.path).query)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Authorized. You can close this tab.")
            received.put(params)

        def log_message(self, *_args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", port), _Handler)
    server.timeout = timeout
    server.handle_request()
    try:
        params = received.get_nowait()
    except queue.Empty:
        raise RuntimeError(f"OAuth callback timed out after {timeout}s") from None
    state = (params.get("state") or [None])[0]
    if state != expected_state:
        raise RuntimeError("OAuth state mismatch — possible CSRF, aborting")
    code = (params.get("code") or [None])[0]
    if not code:
        raise RuntimeError(f"OAuth callback missing 'code': {params}")
    return code


def _cmd_spotify_login(args: argparse.Namespace) -> int:
    app_id = os.environ.get(_SPOTIFY_CLIENT_ID_ENV) or ""
    if not app_id:
        print(f"error: {_SPOTIFY_CLIENT_ID_ENV} not set")
        return 2

    import secrets
    from urllib.parse import urlparse

    from .spotify_user_auth import (
        PLAYBACK_SCOPES,
        PLAYLIST_SCOPES,
        SpotifyUserAuth,
        build_authorize_url,
        generate_pkce_pair,
    )
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    auth = SpotifyUserAuth(client_id=app_id, token_path=store.root / "spotify_user_token.json")

    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)
    redirect_uri = args.redirect_uri
    # one combined login covers both #127 (playlist) and #128 (automated
    # playback) scopes, rather than a second OAuth flow.
    authorize_url = build_authorize_url(
        client_id=app_id,
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
        scopes=PLAYLIST_SCOPES + PLAYBACK_SCOPES,
    )
    print(f"open this URL to authorize: {authorize_url}")

    port = urlparse(redirect_uri).port or 8888
    try:
        code = _wait_for_oauth_callback(port=port, expected_state=state, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 2

    auth.exchange_code(code=code, redirect_uri=redirect_uri, code_verifier=verifier)
    print("authorized — token saved")
    return 0


def _cmd_backfill_playlist(args: argparse.Namespace) -> int:
    from .backfill_playlist import is_backfill_enabled

    if not is_backfill_enabled():
        print("backfill playlist disabled")
        return 1

    app_id = os.environ.get(_SPOTIFY_CLIENT_ID_ENV) or ""
    if not app_id:
        print(f"error: {_SPOTIFY_CLIENT_ID_ENV} not set")
        return 2

    from .spotify_user_auth import SpotifyUserAuth
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    auth = SpotifyUserAuth(client_id=app_id, token_path=store.root / "spotify_user_token.json")
    if not auth.is_authorized():
        print("not authorized — run `music-intel spotify-login` first")
        return 2

    from .backfill_playlist import (
        DEFAULT_PLAYLIST_NAME,
        PlaylistDiff,
        SpotifyPlaylistClient,
        fetch_current_user_id,
        fetch_saved_track_refs,
        played_track_ids,
        select_backfill_tracks,
        sync_backfill_playlist,
    )
    from .identity import MusicBrainzIsrcIndex

    get_token = lambda: auth.access_token()  # noqa: E731
    fetched_user_id = fetch_current_user_id(access_token=get_token)
    playlist_client = SpotifyPlaylistClient(access_token=get_token, user_id=fetched_user_id)
    mb_index = MusicBrainzIsrcIndex(path=getattr(args, "mb_index", None))

    def sync_once() -> PlaylistDiff:
        events = store.load_history()
        played_ids = played_track_ids(events)
        candidates = fetch_saved_track_refs(access_token=get_token)
        # #158 AC2: "already analyzed" consults audio-analysis presence
        # (UserStore), not SharedStore metadata presence.
        selected = select_backfill_tracks(
            candidates,
            played_ids=played_ids,
            has_audio_analysis=store.has_audio_analysis,
            resolve_mbid=lambda t: mb_index.lookup(t.isrc) if t.isrc else None,
        )
        # Built from spotify_id directly, NOT canonical_track_id(t): a selected
        # track may carry an isrc (fetch_saved_track_refs populates it), which
        # would make canonical_track_id prefer isrc:... over spotify:... per
        # the identity waterfall -- spotify_track_uri() requires a
        # spotify:-prefixed id and raises on anything else.
        desired_ids = [f"spotify:{t.spotify_id}" for t in selected]
        diff = sync_backfill_playlist(
            playlist_client, desired_ids=desired_ids, playlist_name=DEFAULT_PLAYLIST_NAME
        )
        print(f"backfill playlist: {len(candidates)} saved tracks, {len(selected)} selected")
        print(f"  synced: +{len(diff.to_add)} -{len(diff.to_remove)}")
        return diff

    if not args.loop:
        sync_once()
        return 0

    def on_error(exc: Exception) -> None:
        print(f"backfill-playlist cycle failed: {exc}")

    interval_s = args.interval_hours * 3600.0
    print(
        f"backfill-playlist running in loop mode (every {args.interval_hours}h, Ctrl+C to stop)..."
    )
    try:
        run_continuous_backfill(sync_once=sync_once, interval_s=interval_s, on_error=on_error)
    except KeyboardInterrupt:
        print("backfill-playlist loop stopped.")
    return 0


def _cmd_replay_queue(args: argparse.Namespace) -> int:
    from .replay_queue import (
        COVERAGE_TARGET,
        measure_coverage_ceiling,
        replay_queue_coverage,
        youtube_crosswalk_resolve_mbid,
    )
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    events = store.load_history()
    # History-import TrackRefs never carry mbid/isrc (only spotify_id/youtube_id/
    # name+artist), while has_audio_analysis is keyed on the mbid-prefixed id the
    # live capture pipeline resolved to -- bridge via the same resolve_mbid seam
    # PR #177 used for select_backfill_tracks, reusing the existing _build_resolver
    # helper (no live Spotify calls: no spotify_source is wired here).
    # #170 AC3: youtube-origin tracks carry no isrc/mbid for resolver.resolve to
    # bridge, so wrap it with the alias-chain crosswalk (store.resolve_track_key)
    # for the youtube_id case.
    resolver = _build_resolver(args)
    resolve_mbid = youtube_crosswalk_resolve_mbid(
        resolve_track_key=store.resolve_track_key,
        resolve_mbid=lambda t: resolver.resolve(t).mbid,
    )
    stats = replay_queue_coverage(
        events,
        has_audio_analysis=store.has_audio_analysis,
        min_valid_plays=args.min_valid_plays,
        cap=args.cap,
        resolve_mbid=resolve_mbid,
    )
    print(
        f"replay queue: {stats.queued_count} tracks queued "
        f"({stats.eligible_track_count} eligible, {stats.already_analyzed_count} already analyzed)"
    )
    print(f"  valid-play coverage: {stats.valid_play_coverage:.1%}")

    # #170 AC8: measure the achievable ceiling before AC5.1's >=80% criterion
    # is applied to a real capture run, and state which knob -- --cap or
    # --min-valid-plays -- is the binding constraint if the target isn't met.
    ceiling_report = measure_coverage_ceiling(
        events,
        has_audio_analysis=store.has_audio_analysis,
        min_valid_plays=args.min_valid_plays,
        cap=args.cap,
        resolve_mbid=resolve_mbid,
    )
    print(
        f"  coverage ceiling: {ceiling_report.ceiling_coverage:.1%} (target {COVERAGE_TARGET:.0%})"
    )
    if ceiling_report.cap_should_be_lifted:
        print("  recommendation: raise --cap to close the gap to the ceiling")
    elif ceiling_report.min_valid_plays_should_be_lifted:
        print("  recommendation: lower --min-valid-plays; the ceiling itself is below target")
    return 0


def _cmd_replay_journal_summary(args: argparse.Namespace) -> int:
    from .replay_capture import replay_journal_path, summarize_replay_journal
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    counts = summarize_replay_journal(replay_journal_path(store))
    if not counts:
        print("replay journal: no attempts recorded yet")
        return 0
    attempts = sum(count for outcome, count in counts.items() if outcome != "requeued")
    print(f"replay journal: {attempts} attempts")
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count}")
    return 0


def _cmd_replay_capture_youtube(args: argparse.Namespace) -> int:
    """#170 AC9: run one YouTube Music participant's replay queue end-to-end
    -- select the queue via the same AC3 crosswalk-aware selector
    ``_cmd_replay_queue`` uses, print the AC8 coverage ceiling against it,
    then decode+analyze each queued track through the real yt-dlp backend
    (#170 AC1), journaling every attempt (AC2). A journaled ``"unavailable"``
    outcome from a prior run is excluded from this run's candidate selection
    via :func:`journaled_unavailable_track_ids` composed into
    ``has_audio_analysis`` -- not re-queued (AC2), and not requeued in-loop
    either (:func:`process_stream_decode_queue`'s own docstring)."""
    from .inference import DiscogsEffnetOnnxModel, MtgJamendoClassifier
    from .replay_queue import (
        COVERAGE_TARGET,
        measure_coverage_ceiling,
        select_replay_queue,
        youtube_crosswalk_resolve_mbid,
    )
    from .store import UserStore
    from .stream_decode import (
        YtDlpStreamDecodeSource,
        journaled_unavailable_track_ids,
        process_stream_decode_queue,
        stream_decode_journal_path,
    )

    store = UserStore(root=args.data_dir)
    events = store.load_history()
    resolver = _build_resolver(args)
    resolve_mbid = youtube_crosswalk_resolve_mbid(
        resolve_track_key=store.resolve_track_key,
        resolve_mbid=lambda t: resolver.resolve(t).mbid,
    )
    journal_path = stream_decode_journal_path(store)
    unavailable_ids = journaled_unavailable_track_ids(journal_path)

    def has_audio_analysis(key: str) -> bool:
        return store.has_audio_analysis(key) or key in unavailable_ids

    ceiling_report = measure_coverage_ceiling(
        events,
        has_audio_analysis=has_audio_analysis,
        min_valid_plays=args.min_valid_plays,
        cap=args.cap,
        resolve_mbid=resolve_mbid,
    )
    print(f"coverage ceiling: {ceiling_report.ceiling_coverage:.1%} (target {COVERAGE_TARGET:.0%})")

    queue = select_replay_queue(
        events,
        has_audio_analysis=has_audio_analysis,
        min_valid_plays=args.min_valid_plays,
        cap=args.cap,
        resolve_mbid=resolve_mbid,
    )
    print(f"replay-capture-youtube: {len(queue)} tracks queued")

    results = process_stream_decode_queue(
        queue=queue,
        source=YtDlpStreamDecodeSource(),
        embedding_model=DiscogsEffnetOnnxModel(),
        classifier=MtgJamendoClassifier(),
        store=store,
        journal_path=journal_path,
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count}")
    return 0


def _cmd_automated_playback_consent(args: argparse.Namespace) -> int:
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    if args.revoke:
        store.revoke_automated_playback_consent()
        print("automated-playback consent revoked")
        return 0

    if not args.grantor:
        print("error: --grantor is required with --grant")
        return 2

    store.grant_automated_playback_consent(
        grantor=args.grantor,
        granted_at=datetime.now(UTC).isoformat(),
        scope=args.scope,
    )
    print("automated-playback consent granted")
    return 0


def _cmd_migrate_audio_analysis_keys(args: argparse.Namespace) -> int:
    from .store import UserStore, migrate_audio_analysis_keys

    store = UserStore(root=args.data_dir)
    report = migrate_audio_analysis_keys(store)
    print(f"migrated {len(report.migrated)}")
    print(f"conflicts {len(report.conflicts)}")
    for old_id, new_id in report.conflicts:
        if new_id == "<unclassifiable>":
            print(
                f"  conflict: {old_id} -> {new_id} "
                "(no MBID/ISRC/Spotify format match and no usable provenance, left untouched)"
            )
        else:
            print(f"  conflict: {old_id} -> {new_id} (target already exists, left untouched)")
    return 0


def _cmd_purge(args: argparse.Namespace) -> int:
    store = UserStore(root=args.data_dir)
    if store.latest_profile() is None and not args.force:
        print(
            "error: no RootProfile has been written for this root -- "
            "run `analyze` first, or pass --force to purge anyway"
        )
        return 2

    root = store.root
    if root.exists():
        shutil.rmtree(root)
    print(f"purged {root}")
    return 0


def _cmd_near_dup_scan(args: argparse.Namespace) -> int:
    from .near_dup import scan
    from .store import UserStore

    store = UserStore(root=args.data_dir)
    json_path = scan(store)
    print(f"wrote report {json_path}")
    return 0


def _cmd_near_dup_apply(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .near_dup import apply

    aliases_path = apply(Path(args.report))
    written = (
        [line for line in aliases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if aliases_path.exists()
        else []
    )
    print(f"wrote {len(written)} alias(es) to {aliases_path}")
    return 0


def _cmd_automated_playback(args: argparse.Namespace) -> int:
    from .store import ConsentFormatError, UserStore

    store = UserStore(root=args.data_dir)
    try:
        has_consent = store.has_automated_playback_consent()
    except ConsentFormatError as exc:
        print(f"error: {exc}")
        return 2
    if not has_consent:
        print(
            "automated playback consent not granted — "
            "run `music-intel automated-playback-consent --grant` first"
        )
        return 1

    app_id = os.environ.get(_SPOTIFY_CLIENT_ID_ENV) or ""
    if not app_id:
        print(f"error: {_SPOTIFY_CLIENT_ID_ENV} not set")
        return 2

    from .spotify_user_auth import SpotifyUserAuth

    auth = SpotifyUserAuth(client_id=app_id, token_path=store.root / "spotify_user_token.json")
    if not auth.is_authorized():
        print("not authorized — run `music-intel spotify-login` first")
        return 2

    from .automated_playback import (
        DeviceNotFoundError,
        SpotifyPlaybackClient,
        TrackSkipped,
        attempt_play,
        build_automated_play_event,
        run_automated_playback,
    )
    from .backfill_playlist import (
        fetch_saved_track_refs,
        played_track_ids,
        select_backfill_tracks,
    )
    from .identity import MusicBrainzIsrcIndex

    get_token = lambda: auth.access_token()  # noqa: E731
    played_ids = played_track_ids(store.load_history())
    candidates = fetch_saved_track_refs(access_token=get_token)
    mb_index = MusicBrainzIsrcIndex(path=getattr(args, "mb_index", None))
    # #158 AC2: "already analyzed" consults audio-analysis presence
    # (UserStore), not SharedStore metadata presence.
    queue = select_backfill_tracks(
        candidates,
        played_ids=played_ids,
        has_audio_analysis=store.has_audio_analysis,
        resolve_mbid=lambda t: mb_index.lookup(t.isrc) if t.isrc else None,
    )
    if not queue:
        print("automated playback: nothing to play")
        return 0

    playback_client = SpotifyPlaybackClient(access_token=get_token)
    try:
        playback_client.resolve_device_id(args.device_name)
    except DeviceNotFoundError:
        print(f"automated playback: device '{args.device_name}' not found")
        return 3

    def on_play(track) -> None:
        store.append_events([build_automated_play_event(track, played_at=datetime.now(UTC))])

    # #159 AC2/AC3: gate + bounded-retry each play through attempt_play().
    # ``queue`` is a live list mutated by the wrapper below (not just read),
    # so a "requeued"/"deferred" track re-appears later in the same
    # ``for track in queue`` iteration run_automated_playback is doing.
    retry_counts: dict[str, int] = {}
    defer_counts: dict[str, int] = {}
    max_defers = 3

    def journal(track, reason: str) -> None:
        print(f"automated playback: {track.name} -- {reason}")

    queue_len = len(queue)

    def play_track(track) -> None:
        attempt = attempt_play(playback_client, track, retry_counts=retry_counts, journal=journal)
        if attempt.status == "played":
            return
        if attempt.status == "requeued":
            queue.append(track)
        elif attempt.status == "deferred":
            count = defer_counts.get(track.spotify_id, 0) + 1
            defer_counts[track.spotify_id] = count
            if count <= max_defers:
                queue.append(track)
        raise TrackSkipped()

    result = run_automated_playback(
        queue=queue,
        play_track=play_track,
        # track_duration_s() strips the id's prefix and sends the remainder
        # to Spotify's GET /tracks/{id} -- it must stay spotify_id-based, not
        # canonical_track_id(t), which would send an isrc/mbid value where
        # Spotify expects its own track id (#158 identity waterfall).
        track_duration_s=lambda t: playback_client.track_duration_s(f"spotify:{t.spotify_id}"),
        has_consent=store.has_automated_playback_consent,
        on_play=on_play,
        pause=playback_client.pause,
        sleep=time.sleep,
    )
    print(f"automated playback: played {len(result.played)}/{queue_len}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-intel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="derive a RootProfile from history")
    p_analyze.add_argument(
        "--user-id",
        required=False,
        default=None,
        help="per-user store identifier (default: --data-dir basename, #165 AC4)",
    )
    p_analyze.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_analyze.add_argument(
        "--with-audio",
        action="store_true",
        help="run the audio derivation stage (needs an AcousticBrainz dump)",
    )
    p_analyze.add_argument(
        "--with-scene",
        action="store_true",
        help="run the scene derivation stage (needs LASTFM_API_KEY)",
    )
    p_analyze.add_argument(
        "--with-musicbrainz-genre",
        action="store_true",
        help="fill scene tag gaps from MusicBrainz recording genres/tags "
        "(needs MUSICBRAINZ_APP_CONTACT; MBID-only, no name-search fallback)",
    )
    p_analyze.add_argument(
        "--with-discogs",
        action="store_true",
        help="fill scene tag gaps from Discogs release styles (needs DISCOGS_TOKEN)",
    )
    p_analyze.add_argument(
        "--ab-index",
        default=None,
        help="AcousticBrainz features JSONL (default: $ACOUSTICBRAINZ_FEATURES_INDEX "
        "or $ACOUSTICBRAINZ_DUMP_DIR/acousticbrainz_features.jsonl)",
    )
    p_analyze.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV for the identity bridge under "
        "--with-audio (default: $MUSICBRAINZ_ISRC_INDEX or "
        "$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv)",
    )
    p_analyze.add_argument(
        "--shared-store",
        choices=["local", "supabase", "memory"],
        default="local",
        help="metadata store for enrichment: 'local' (persistent file cache, no "
        "creds), 'supabase' (multi-user cloud cache, V2+), or 'memory' "
        "(ephemeral, single-run). Default: local.",
    )
    p_analyze.add_argument(
        "--shared-store-path",
        default=None,
        help="path to the local shared-store JSONL "
        "(only with --shared-store local; default: <data root>/shared_cache.jsonl)",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_resolve = sub.add_parser("resolve", help="resolve track identity (spotify->ISRC->MBID)")
    p_resolve.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_resolve.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV (default: $MUSICBRAINZ_ISRC_INDEX)",
    )
    p_resolve.add_argument(
        "--with-spotify",
        action="store_true",
        help=(
            "bridge spotify_id->ISRC via the live Spotify API for tracks lacking an "
            "ISRC (needs SPOTIFY_CLIENT_ID + SPOTIFY_CLIENT_SECRET; caches locally)"
        ),
    )
    p_resolve.set_defaults(func=_cmd_resolve)

    p_crosswalk = sub.add_parser(
        "crosswalk",
        help="metadata cross-walk: alias history-only keys to MBIDs via the ISRC index (#178)",
    )
    p_crosswalk.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_crosswalk.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV (default: $MUSICBRAINZ_ISRC_INDEX)",
    )
    p_crosswalk.set_defaults(func=_cmd_crosswalk)

    p_import = sub.add_parser("import-ifttt", help="import IFTTT .xlsx history exports")
    p_import.add_argument(
        "--from",
        dest="source",
        required=True,
        help="directory of IFTTT Spotify_data*.xlsx exports",
    )
    p_import.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_import.set_defaults(func=_cmd_import_ifttt)

    p_import_spotify = sub.add_parser(
        "import-spotify", help="import a Spotify Extended Streaming History export"
    )
    p_import_spotify.add_argument(
        "--from",
        dest="source",
        required=True,
        help="directory of Streaming_History_Audio_*.json files from the export",
    )
    p_import_spotify.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_import_spotify.set_defaults(func=_cmd_import_spotify)

    p_import_youtube = sub.add_parser(
        "import-youtube", help="import a Google Takeout watch-history.json (YouTube Music, #164)"
    )
    p_import_youtube.add_argument(
        "--from",
        dest="source",
        required=True,
        help="path to watch-history.json from a Google Takeout export",
    )
    p_import_youtube.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_import_youtube.set_defaults(func=_cmd_import_youtube)

    p_import_account = sub.add_parser(
        "import-account",
        help="import a Spotify Account Data export (likes/bans/follows/playlists/Marquee)",
    )
    p_import_account.add_argument(
        "--from",
        dest="source",
        required=True,
        help="directory of the Account Data export "
        "(YourLibrary.json, Playlist*.json, Marquee.json)",
    )
    p_import_account.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_import_account.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV for the report-only coverage measure "
        "(default: $MUSICBRAINZ_ISRC_INDEX)",
    )
    p_import_account.add_argument(
        "--artist-index",
        dest="artist_index",
        default=None,
        help="MusicBrainz artist-URI->MBID index TSV for the report-only artist "
        "coverage measure (#102; default: $MUSICBRAINZ_ARTIST_INDEX)",
    )
    p_import_account.set_defaults(func=_cmd_import_account)

    p_import_account_history = sub.add_parser(
        "import-account-history",
        help="import Spotify Account Data StreamingHistory_music_*.json plays "
        "(extends past the newest existing spotify_extended play)",
    )
    p_import_account_history.add_argument(
        "--from",
        dest="source",
        required=True,
        help="directory of the Account Data export (StreamingHistory_music_*.json)",
    )
    p_import_account_history.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_import_account_history.set_defaults(func=_cmd_import_account_history)

    p_capture_spike = sub.add_parser(
        "capture-spike",
        help="live WASAPI per-process loopback capture -> librosa/onnxruntime spike"
        " (#124, Windows-only)",
    )
    p_capture_spike.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="seconds of audio to capture and analyze (default: 120.0, #139 AC1)",
    )
    p_capture_spike.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_capture_spike.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV for identity resolution "
        "(default: $MUSICBRAINZ_ISRC_INDEX)",
    )
    p_capture_spike.set_defaults(func=_cmd_capture_spike)

    p_capture_loop = sub.add_parser(
        "capture-loop",
        help="continuous WASAPI capture -> librosa/onnxruntime, until Ctrl+C (#136, Windows-only)",
    )
    p_capture_loop.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="seconds of audio to capture per track (default: 120.0, #139 AC1)",
    )
    p_capture_loop.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="seconds between now-playing polls (default: 5.0)",
    )
    p_capture_loop.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_capture_loop.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV for identity resolution "
        "(default: $MUSICBRAINZ_ISRC_INDEX)",
    )
    p_capture_loop.set_defaults(func=_cmd_capture_loop)

    p_build_mb = sub.add_parser(
        "build-mb-index",
        help="build the ISRC->MBID TSV from raw MusicBrainz dump tables (offline)",
    )
    p_build_mb.add_argument(
        "--isrc-dump",
        dest="isrc_dump",
        required=True,
        help="raw MusicBrainz `isrc` table file (columns id, recording, isrc, ...)",
    )
    p_build_mb.add_argument(
        "--recording-dump",
        dest="recording_dump",
        required=True,
        help="raw MusicBrainz `recording` table file (columns id, gid, ...)",
    )
    p_build_mb.add_argument(
        "--out",
        required=True,
        help="output ISRC->MBID TSV (point --mb-index / MUSICBRAINZ_ISRC_INDEX here)",
    )
    p_build_mb.set_defaults(func=_cmd_build_mb_index)

    p_build_artist = sub.add_parser(
        "build-artist-index",
        help="build the artist-URI->MBID TSV from raw MusicBrainz dump tables (#102, offline)",
    )
    p_build_artist.add_argument(
        "--url-dump",
        dest="url_dump",
        required=True,
        help="raw MusicBrainz `url` table file (columns id, gid, url, ...)",
    )
    p_build_artist.add_argument(
        "--l-artist-url-dump",
        dest="l_artist_url_dump",
        required=True,
        help="raw MusicBrainz `l_artist_url` table file (columns id, link, entity0, entity1, ...)",
    )
    p_build_artist.add_argument(
        "--artist-dump",
        dest="artist_dump",
        required=True,
        help="raw MusicBrainz `artist` table file (columns id, gid, ...)",
    )
    p_build_artist.add_argument(
        "--out",
        required=True,
        help="output artist-URI->MBID TSV (point --artist-index / MUSICBRAINZ_ARTIST_INDEX here)",
    )
    p_build_artist.set_defaults(func=_cmd_build_artist_index)

    p_build_ab = sub.add_parser(
        "build-ab-index",
        help="fetch AcousticBrainz features for resolved MBIDs into the features JSONL",
    )
    p_build_ab.add_argument(
        "--data-dir",
        default=None,
        help="data root holding the identity cache (MBID source) and raw sidecar "
        "(default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_build_ab.add_argument(
        "--out",
        required=True,
        help="output AcousticBrainz features JSONL "
        "(point --ab-index / ACOUSTICBRAINZ_FEATURES_INDEX here)",
    )
    p_build_ab.add_argument(
        "--raw-cache",
        dest="raw_cache",
        default=None,
        help="resumable raw-scalar sidecar (default: <data root>/acousticbrainz_raw.jsonl)",
    )
    p_build_ab.set_defaults(func=_cmd_build_ab_index)

    p_backfill = sub.add_parser(
        "backfill-playlist",
        help="sync the opt-in 'to-analyze' Spotify playlist (#127, daily refresh)",
    )
    p_backfill.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_backfill.add_argument(
        "--loop",
        action="store_true",
        help="keep re-syncing on --interval-hours until Ctrl+C, instead of a single run (AC2)",
    )
    p_backfill.add_argument(
        "--interval-hours",
        dest="interval_hours",
        type=float,
        default=24.0,
        help="hours between refreshes in --loop mode (default: 24.0, AC2's daily cadence)",
    )
    p_backfill.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV, bridging a saved-track candidate's ISRC "
        "to the MBID the live pipeline keys audio-analysis by, for cross-pipeline dedup "
        "(default: $MUSICBRAINZ_ISRC_INDEX or $MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv)",
    )
    p_backfill.set_defaults(func=_cmd_backfill_playlist)

    p_replay_queue = sub.add_parser(
        "replay-queue",
        help="pilot slice 1 (#163): select+cap the replay-queue from local "
        "history and print the AC5.1 valid-play coverage stat",
    )
    p_replay_queue.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_replay_queue.add_argument(
        "--min-valid-plays",
        dest="min_valid_plays",
        type=int,
        default=MIN_VALID_PLAYS,
        help=f"valid plays (>=30s, AC1) a canonical key needs to be eligible "
        f"(default: {MIN_VALID_PLAYS})",
    )
    p_replay_queue.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_REPLAY_QUEUE_CAP,
        help=f"max tracks in the stratified queue (default: {DEFAULT_REPLAY_QUEUE_CAP})",
    )
    p_replay_queue.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV, bridging a history-import candidate's "
        "resolved ISRC to the MBID the live pipeline keys audio-analysis by, for "
        "cross-pipeline dedup (default: $MUSICBRAINZ_ISRC_INDEX or "
        "$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv)",
    )
    p_replay_queue.set_defaults(func=_cmd_replay_queue)

    p_replay_journal_summary = sub.add_parser(
        "replay-journal-summary",
        help="print per-outcome counts from the replay capture journal "
        "for the weekly checkpoint (#166 AC5)",
    )
    p_replay_journal_summary.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_replay_journal_summary.set_defaults(func=_cmd_replay_journal_summary)

    p_replay_capture_youtube = sub.add_parser(
        "replay-capture-youtube",
        help="#170 AC9: run one YouTube Music participant's replay queue "
        "end-to-end via yt-dlp stream decode, against the measured coverage "
        "ceiling (AC8)",
    )
    p_replay_capture_youtube.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_replay_capture_youtube.add_argument(
        "--min-valid-plays",
        dest="min_valid_plays",
        type=int,
        default=MIN_VALID_PLAYS,
        help=f"valid plays (>=30s, AC1) a canonical key needs to be eligible "
        f"(default: {MIN_VALID_PLAYS})",
    )
    p_replay_capture_youtube.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_REPLAY_QUEUE_CAP,
        help=f"max tracks in the stratified queue (default: {DEFAULT_REPLAY_QUEUE_CAP})",
    )
    p_replay_capture_youtube.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV, bridging a history-import candidate's "
        "resolved ISRC to the MBID the live pipeline keys audio-analysis by, for "
        "cross-pipeline dedup (default: $MUSICBRAINZ_ISRC_INDEX or "
        "$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv)",
    )
    p_replay_capture_youtube.set_defaults(func=_cmd_replay_capture_youtube)

    p_login = sub.add_parser(
        "spotify-login",
        help="authorize the backfill playlist's Spotify user-OAuth scopes (#127 AC1/AC5)",
    )
    p_login.add_argument(
        "--data-dir",
        default=None,
        help="data root to write the local token file into "
        "(default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_login.add_argument(
        "--redirect-uri",
        default="http://127.0.0.1:8888/callback",
        help="registered Spotify app redirect URI (default: http://127.0.0.1:8888/callback)",
    )
    p_login.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the browser redirect before giving up (default: 120)",
    )
    p_login.set_defaults(func=_cmd_spotify_login)

    p_playback_consent = sub.add_parser(
        "automated-playback-consent",
        help="grant/revoke consent for automated playback mode (#128 AC1/AC3)",
    )
    p_playback_consent.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    consent_group = p_playback_consent.add_mutually_exclusive_group(required=True)
    consent_group.add_argument(
        "--grant",
        action="store_true",
        help="grant consent for the agent to drive automated playback",
    )
    consent_group.add_argument(
        "--revoke",
        action="store_true",
        help="revoke consent immediately, stopping any running session (AC3)",
    )
    p_playback_consent.add_argument(
        "--grantor",
        default=None,
        help="who is granting consent (#165 AC3, required with --grant)",
    )
    p_playback_consent.add_argument(
        "--scope",
        default="automated-playback",
        help="what the consent covers (#165 AC3, default: automated-playback)",
    )
    p_playback_consent.set_defaults(func=_cmd_automated_playback_consent)

    p_playback = sub.add_parser(
        "automated-playback",
        help="play through the backfill queue via the real Spotify client, human-paced (#128)",
    )
    p_playback.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_playback.add_argument(
        "--device-name",
        required=True,
        help="exact Spotify device name to replay through (#159 AC1) -- "
        "see `GET /me/player/devices` on the account, or the device's own UI",
    )
    p_playback.add_argument(
        "--mb-index",
        default=None,
        help="MusicBrainz ISRC->MBID index TSV, bridging a saved-track candidate's ISRC "
        "to the MBID the live pipeline keys audio-analysis by, for cross-pipeline dedup "
        "(default: $MUSICBRAINZ_ISRC_INDEX or $MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv)",
    )
    p_playback.set_defaults(func=_cmd_automated_playback)

    p_purge = sub.add_parser(
        "purge",
        help="delete a participant data root entirely; never touches the pool "
        "or SharedStore (#165 AC2)",
    )
    p_purge.add_argument(
        "--data-dir",
        required=True,
        help="participant data root to delete (no default -- purge never "
        "falls back to $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_purge.add_argument(
        "--force",
        action="store_true",
        help="purge even if no RootProfile has been written for this root yet",
    )
    p_purge.set_defaults(func=_cmd_purge)

    p_migrate_keys = sub.add_parser(
        "migrate-audio-analysis-keys",
        help=(
            "one-shot rename of bare-key audio-analysis files to the prefixed "
            "canonical_track_id form (#158 AC4)"
        ),
    )
    p_migrate_keys.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_migrate_keys.set_defaults(func=_cmd_migrate_audio_analysis_keys)

    p_near_dup = sub.add_parser(
        "near-dup",
        help="offline embedding-space near-duplicate batch merge (#140)",
    )
    near_dup_sub = p_near_dup.add_subparsers(dest="near_dup_command", required=True)

    p_near_dup_scan = near_dup_sub.add_parser(
        "scan", help="scan one data root and write a transparent report (#140 AC2)"
    )
    p_near_dup_scan.add_argument(
        "--data-dir",
        default=None,
        help="data root (default: $MUSIC_INTEL_DATA_DIR or ./data)",
    )
    p_near_dup_scan.set_defaults(func=_cmd_near_dup_scan)

    p_near_dup_apply = near_dup_sub.add_parser(
        "apply", help="apply a reviewed scan report, writing aliases.jsonl (#140 AC4)"
    )
    p_near_dup_apply.add_argument("report", help="path to a reviewed scan report JSON file")
    p_near_dup_apply.set_defaults(func=_cmd_near_dup_apply)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Populate os.environ from a gitignored .env (the documented credential
    # mechanism: copy .env.example -> .env, fill in the keys). Anchored on the
    # invocation directory (usecwd=True) so it works both from a source checkout
    # and an installed wheel; host env wins (override=False) and a missing .env
    # is a silent no-op. This is the single load point — every subcommand reads
    # its credentials through os.environ downstream.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
