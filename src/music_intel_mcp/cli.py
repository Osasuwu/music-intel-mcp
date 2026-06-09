"""CLI surface. V0 exposes a single ``analyze`` command.

    music-intel analyze --user-id petr [--data-dir ./data]

Loads the per-user history, runs the derivation engine, writes a RootProfile
snapshot to the per-user store, and prints the snapshot path + a one-line
summary.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .analyzer import analyze
from .store import UserStore


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
