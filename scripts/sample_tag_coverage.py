"""One-off measurement script for #132 — combined Last.fm/MusicBrainz/Discogs
tag coverage on a representative sample of the real shared cache.

Not part of the package; run directly:
    python scripts/sample_tag_coverage.py

Samples ~SAMPLE_SIZE track_ids from data/shared_cache.jsonl, stratified by
has-mbid/no-mbid in proportion to the full population (so the sample doesn't
skew toward the MusicBrainz-eligible or -ineligible leg), then runs the same
enrich_tags() used in production against the *same* persistent store — so
progress is never wasted relative to a future full run.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from music_intel_mcp.scene import (  # noqa: E402
    CompositeTagSource,
    DiscogsStyleSource,
    LastfmTagSource,
    MusicBrainzGenreSource,
)
from music_intel_mcp.shared_store import LocalSharedStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "data" / "shared_cache.jsonl"
SAMPLE_SIZE = 7000
SEED = 132  # deterministic sample, tied to the issue number

random.seed(SEED)


def _load_ids_by_mbid_bucket() -> tuple[list[str], list[str]]:
    with_mbid: list[str] = []
    without_mbid: list[str] = []
    with CACHE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            (with_mbid if row.get("mbid") else without_mbid).append(row["track_id"])
    return with_mbid, without_mbid


def main() -> None:
    with_mbid, without_mbid = _load_ids_by_mbid_bucket()
    total = len(with_mbid) + len(without_mbid)
    frac_mbid = len(with_mbid) / total
    n_mbid = round(SAMPLE_SIZE * frac_mbid)
    n_no_mbid = SAMPLE_SIZE - n_mbid

    sample = random.sample(with_mbid, min(n_mbid, len(with_mbid))) + random.sample(
        without_mbid, min(n_no_mbid, len(without_mbid))
    )
    random.shuffle(sample)

    print(
        f"population: {total} total, {len(with_mbid)} with mbid, {len(without_mbid)} without",
        flush=True,
    )
    print(
        f"sample: {len(sample)} tracks ({n_mbid} with mbid, {n_no_mbid} without), seed={SEED}",
        flush=True,
    )

    store = LocalSharedStore(path=CACHE_PATH)
    source = CompositeTagSource([LastfmTagSource(), MusicBrainzGenreSource(), DiscogsStyleSource()])

    from music_intel_mcp.scene import enrich_tags

    report = enrich_tags(sample, store, source, now=datetime.now(UTC))

    print("---- sample tag-coverage report (#132) ----", flush=True)
    print(f"enriched:        {len(report.enriched)}", flush=True)
    print(f"already_present: {len(report.already_present)}", flush=True)
    print(f"missing_tags:    {len(report.missing_tags)}", flush=True)
    print(f"errors:          {len(report.errors)}", flush=True)
    print(f"total_considered:{report.total_considered}", flush=True)
    print(f"coverage:        {report.coverage:.4%}", flush=True)


if __name__ == "__main__":
    main()
