"""CLI surface. V0 exposes ``analyze``, ``resolve``, and ``import-ifttt``.

    music-intel import-ifttt --from <dir> [--data-dir ./data]
    music-intel analyze --user-id petr [--data-dir ./data]
                        [--with-audio] [--with-scene]
                        [--ab-index PATH] [--shared-store supabase|memory]
    music-intel resolve [--data-dir ./data] [--mb-index PATH]

``import-ifttt`` merges a directory of IFTTT Spotify ``.xlsx`` exports into the
per-user ``history.jsonl`` (dedup + idempotent re-import). ``analyze`` loads that
history, runs the derivation engine, writes a RootProfile snapshot, and prints
the path + a one-line summary. ``resolve`` walks the history through the
spotify_id -> ISRC -> MBID identity waterfall and reports resolution coverage
(caching resolved identities for re-runs).

Enrichment is **off by default**: with no ``--with-*`` flag ``analyze`` produces
an honest-empty profile (the V0 baseline). ``--with-audio`` / ``--with-scene``
opt the run into the audio / scene derivation stages by constructing the
production source adapters (the env-pointed :class:`AcousticBrainzDump` and the
``LASTFM_API_KEY``-backed :class:`LastfmTagSource`) plus a shared metadata store,
then handing them to :func:`analyze`. The pipelines themselves are locked
(#63/#64); this module only wires the already-built adapters to the CLI.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence

from .analyzer import analyze
from .audio import AcousticBrainzDump, AudioFeatureSource
from .identity import IdentityCache, IdentityResolver, MusicBrainzIsrcIndex
from .ingest import IngestStats, dedup_events, load_ifttt_dir
from .scene import LastfmTagSource, TagSource
from .shared_store import InMemorySharedStore, SharedStore, SupabaseSharedStore
from .store import UserStore

# Canonical env-var names this CLI checks for *presence* (never value) before an
# enrichment run, so a missing credential fails fast with a clear message rather
# than mid-pipeline. They mirror the constants owned by ``scene``/``shared_store``
# (kept as literals here because they are the CLI's documented env contract).
_LASTFM_API_KEY_ENV = "LASTFM_API_KEY"
_SUPABASE_URL_ENV = "SUPABASE_URL"
_SUPABASE_KEY_ENV = "SUPABASE_KEY"


def _build_shared_store(kind: str) -> SharedStore:
    """Construct the requested :class:`SharedStore`. ``memory`` is an ephemeral
    single-run store (no persistence, no creds); ``supabase`` is the production
    shared cache (lazy client — no network until first read/write)."""
    if kind == "memory":
        return InMemorySharedStore()
    return SupabaseSharedStore()


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
    if not (args.with_audio or args.with_scene):
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
    if errors:
        return None, None, None, errors

    shared_store = _build_shared_store(args.shared_store)
    audio_source = AcousticBrainzDump(path=args.ab_index) if args.with_audio else None
    tag_source = LastfmTagSource() if args.with_scene else None
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    shared_store, audio_source, tag_source, errors = plan_enrichment(args, os.environ)
    if errors:
        for err in errors:
            print(f"error: {err}")
        return 2

    user_store = UserStore(root=args.data_dir)
    events = user_store.load_history()
    profile = analyze(
        events,
        user_id=args.user_id,
        shared_store=shared_store,
        audio_source=audio_source,
        tag_source=tag_source,
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
    index = MusicBrainzIsrcIndex(path=args.mb_index)
    cache = IdentityCache(root=args.data_dir)
    resolver = IdentityResolver(index, cache=cache)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-intel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="derive a RootProfile from history")
    p_analyze.add_argument("--user-id", required=True, help="per-user store identifier")
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
        "--ab-index",
        default=None,
        help="AcousticBrainz features JSONL (default: $ACOUSTICBRAINZ_FEATURES_INDEX "
        "or $ACOUSTICBRAINZ_DUMP_DIR/acousticbrainz_features.jsonl)",
    )
    p_analyze.add_argument(
        "--shared-store",
        choices=["supabase", "memory"],
        default="supabase",
        help="metadata store for enrichment: 'supabase' (shared cache) or "
        "'memory' (ephemeral, single-run). Default: supabase.",
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
    p_resolve.set_defaults(func=_cmd_resolve)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
