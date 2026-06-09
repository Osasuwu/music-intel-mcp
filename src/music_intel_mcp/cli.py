"""CLI surface. V0 exposes ``analyze``, ``resolve``, and ``import-ifttt``.

    music-intel import-ifttt --from <dir> [--data-dir ./data]
    music-intel analyze --user-id petr [--data-dir ./data]
    music-intel resolve [--data-dir ./data] [--mb-index PATH]

``import-ifttt`` merges a directory of IFTTT Spotify ``.xlsx`` exports into the
per-user ``history.jsonl`` (dedup + idempotent re-import). ``analyze`` loads that
history, runs the derivation engine, writes a RootProfile snapshot, and prints
the path + a one-line summary. ``resolve`` walks the history through the
spotify_id -> ISRC -> MBID identity waterfall and reports resolution coverage
(caching resolved identities for re-runs).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .analyzer import analyze
from .identity import IdentityCache, IdentityResolver, MusicBrainzIsrcIndex
from .ingest import IngestStats, dedup_events, load_ifttt_dir
from .store import UserStore


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
    store = UserStore(root=args.data_dir)
    events = store.load_history()
    profile = analyze(events, user_id=args.user_id)
    path = store.write_profile(profile)

    gf = profile.generated_from
    print(f"snapshot: {path}")
    print(
        f"  events={gf.n_events} unique_tracks={gf.n_unique_tracks} "
        f"span_days={gf.history_span_days} sources={','.join(gf.data_sources) or '-'}"
    )
    print(
        f"  roots={len(profile.roots)} tendencies={len(profile.tendencies)} "
        f"epochs={len(profile.epochs)} maturity={profile.model_maturity}"
    )
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
