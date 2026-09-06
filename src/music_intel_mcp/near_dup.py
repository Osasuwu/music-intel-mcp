"""Offline embedding-space near-duplicate batch merge (#140).

Reconciles distinct track keys (per the canonical waterfall in ``store.py``)
that turn out to be the same recording -- e.g. the same track captured once
via Spotify metadata and once via a bare name-key fallback. Runs offline,
batch, over a ``UserStore`` root: it never touches the live capture path.

Two-tier decision: embedding cosine distance proposes candidate pairs, raw
chromaprint fingerprints (from ``chromaprint_fpcalc.compute_raw_fingerprint``,
written by ``store.write_fingerprint`` per #140 AC1) either confirm or veto
them. See CONTEXT.md "Offline embedding-space near-duplicate batch merge"
for the full decision-tree writeup this module implements piece by piece.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .store import UserStore

# --- AC3: fingerprint match rule --------------------------------------- #

# A confirmed match must clear this many overlapping frames -- below this,
# even a perfect BER is too little evidence to act on (CONTEXT.md: raw
# chromaprint frames are ~1/8s each, so 240 frames is ~30s of overlap).
FINGERPRINT_MIN_OVERLAP_FRAMES = 240

# Bit error rate at the best-aligned offset must be at or under this to
# treat two fingerprints as the same recording.
FINGERPRINT_BER_THRESHOLD = 0.25


@dataclass(frozen=True)
class FingerprintMatchResult:
    offset: int
    ber: float
    overlap: int


def _ber_at_offset(fp_a: list[int], fp_b: list[int], offset: int) -> tuple[float, int]:
    """Bit error rate between ``fp_a`` and ``fp_b`` shifted by ``offset``
    frames (positive: ``fp_b`` starts later than ``fp_a``)."""
    if offset >= 0:
        pairs = zip(fp_a[offset:], fp_b, strict=False)
    else:
        pairs = zip(fp_a, fp_b[-offset:], strict=False)

    overlap = 0
    mismatched_bits = 0
    for a, b in pairs:
        overlap += 1
        mismatched_bits += bin(a ^ b).count("1")

    if overlap == 0:
        return 1.0, 0
    return mismatched_bits / (overlap * 32), overlap


def match_fingerprints(fp_a: list[int], fp_b: list[int]) -> FingerprintMatchResult | None:
    """Search offsets in ``[-max_len/2, +max_len/2]`` for the best-aligned
    (lowest-BER) overlap between two raw chromaprint frame arrays, and report
    that measurement.

    Returns ``None`` when no offset in range reaches
    ``FINGERPRINT_MIN_OVERLAP_FRAMES`` of overlap -- there simply isn't
    enough shared evidence to measure a BER on, regardless of threshold.
    Reports the measurement even when its BER exceeds
    ``FINGERPRINT_BER_THRESHOLD``: thresholding is the caller's decision
    (the scan decision tree), not this pure function's.
    """
    max_len = max(len(fp_a), len(fp_b))
    max_offset = max_len // 2

    best: FingerprintMatchResult | None = None
    for offset in range(-max_offset, max_offset + 1):
        ber, overlap = _ber_at_offset(fp_a, fp_b, offset)
        if overlap < FINGERPRINT_MIN_OVERLAP_FRAMES:
            continue
        if best is None or ber < best.ber:
            best = FingerprintMatchResult(offset=offset, ber=ber, overlap=overlap)

    return best


# --- AC2: near-dup scan -------------------------------------------------- #

# How many nearest neighbours (by cosine distance) each record is compared
# against. Pilot-scale batches don't need an ANN index -- a chunked brute
# force over top-K candidates per record is enough (CONTEXT.md "Near-duplicate
# key reconciliation").
TOP_K_CANDIDATES = 5

# A pair confirmed by a matching raw fingerprint can tolerate a looser
# embedding distance than a pair with no fingerprint evidence at all.
COSINE_DISTANCE_THRESHOLD_FINGERPRINT_EMBEDDING = 0.15
COSINE_DISTANCE_THRESHOLD_EMBEDDING_ONLY = 0.05

# Hub guard (CONTEXT.md "Post-CRITIC refinements"): a record with this many
# neighbours at ~zero distance is almost certainly a degenerate capture
# (silence/ad) that mean-pools to one vector, not a genuine cluster of
# duplicates -- every pair touching it is rejected rather than proposed.
HUB_GUARD_NEIGHBOR_COUNT = 3
HUB_GUARD_DISTANCE_EPS = 1e-6

TIER_FINGERPRINT_EMBEDDING = "fingerprint+embedding"
TIER_EMBEDDING_ONLY = "embedding_only"

VERDICT_PROPOSED = "proposed"
VERDICT_REJECTED = "rejected"

REASON_ABOVE_THRESHOLD = "above_threshold"
REASON_FINGERPRINT_DISAGREE = "fingerprint_disagree"
REASON_HUB_SUSPECT = "hub_suspect"


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def _hub_suspects(distances: np.ndarray) -> set[int]:
    n = distances.shape[0]
    suspects: set[int] = set()
    for i in range(n):
        neighbor_count = sum(
            1 for j in range(n) if j != i and distances[i, j] <= HUB_GUARD_DISTANCE_EPS
        )
        if neighbor_count >= HUB_GUARD_NEIGHBOR_COUNT:
            suspects.add(i)
    return suspects


def _candidate_pairs(distances: np.ndarray) -> set[frozenset[int]]:
    n = distances.shape[0]
    pairs: set[frozenset[int]] = set()
    for i in range(n):
        order = sorted((j for j in range(n) if j != i), key=lambda j: distances[i, j])
        for j in order[:TOP_K_CANDIDATES]:
            pairs.add(frozenset((i, j)))
    return pairs


def _evaluate_pair(store: UserStore, key_a: str, key_b: str, distance: float) -> dict:
    fp_a = store.read_fingerprint(key_a)
    fp_b = store.read_fingerprint(key_b)

    if fp_a is not None and fp_b is not None:
        tier = TIER_FINGERPRINT_EMBEDDING
        match = match_fingerprints(fp_a, fp_b)
        if match is None or match.ber > FINGERPRINT_BER_THRESHOLD:
            return {
                "tier": tier,
                "fingerprint": None if match is None else _match_payload(match),
                "verdict": VERDICT_REJECTED,
                "reason": REASON_FINGERPRINT_DISAGREE,
            }
        if distance > COSINE_DISTANCE_THRESHOLD_FINGERPRINT_EMBEDDING:
            return {
                "tier": tier,
                "fingerprint": _match_payload(match),
                "verdict": VERDICT_REJECTED,
                "reason": REASON_ABOVE_THRESHOLD,
            }
        return {
            "tier": tier,
            "fingerprint": _match_payload(match),
            "verdict": VERDICT_PROPOSED,
            "reason": None,
        }

    tier = TIER_EMBEDDING_ONLY
    if distance > COSINE_DISTANCE_THRESHOLD_EMBEDDING_ONLY:
        return {
            "tier": tier,
            "fingerprint": None,
            "verdict": VERDICT_REJECTED,
            "reason": REASON_ABOVE_THRESHOLD,
        }
    return {"tier": tier, "fingerprint": None, "verdict": VERDICT_PROPOSED, "reason": None}


def _match_payload(match: FingerprintMatchResult) -> dict:
    return {"ber": match.ber, "offset": match.offset, "overlap": match.overlap}


def _thresholds_payload() -> dict:
    return {
        "fingerprint_ber_threshold": FINGERPRINT_BER_THRESHOLD,
        "fingerprint_min_overlap_frames": FINGERPRINT_MIN_OVERLAP_FRAMES,
        "cosine_distance_fingerprint_embedding": COSINE_DISTANCE_THRESHOLD_FINGERPRINT_EMBEDDING,
        "cosine_distance_embedding_only": COSINE_DISTANCE_THRESHOLD_EMBEDDING_ONLY,
        "hub_guard_neighbor_count": HUB_GUARD_NEIGHBOR_COUNT,
        "hub_guard_distance_eps": HUB_GUARD_DISTANCE_EPS,
        "top_k_candidates": TOP_K_CANDIDATES,
    }


def _render_markdown(report: dict) -> str:
    lines = [
        "# Near-duplicate scan report",
        "",
        f"Root: `{report['root']}`  ",
        f"Generated: {report['generated_at']}",
        "",
        "| keys | distance | tier | verdict | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for pair in report["pairs"]:
        keys = " / ".join(pair["keys"])
        lines.append(
            f"| {keys} | {pair['distance']:.6f} | {pair['tier'] or '-'} "
            f"| {pair['verdict']} | {pair['reason'] or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def scan(
    store: UserStore,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    """AC2: scan one ``UserStore`` root for embedding-space near-duplicate
    candidates and write a transparent report -- every considered pair,
    including rejections and why, ``accepted: null`` on every entry. This
    never merges anything; that is ``apply``'s job (AC4) over a reviewed copy
    of this report."""
    records = store.list_audio_analyses()
    keys = [r.track_id for r in records]
    embeddings = np.array([r.embedding for r in records], dtype=float)

    pairs_report: list[dict] = []
    if len(records) >= 2:
        distances = _cosine_distance_matrix(embeddings)
        hub_suspects = _hub_suspects(distances)
        candidate_pairs = _candidate_pairs(distances)

        for pair in candidate_pairs:
            i, j = tuple(pair)
            key_a, key_b = sorted((keys[i], keys[j]))
            idx_a, idx_b = keys.index(key_a), keys.index(key_b)
            distance = float(distances[idx_a, idx_b])

            if i in hub_suspects or j in hub_suspects:
                pairs_report.append(
                    {
                        "keys": [key_a, key_b],
                        "distance": distance,
                        "tier": None,
                        "fingerprint": None,
                        "verdict": VERDICT_REJECTED,
                        "reason": REASON_HUB_SUSPECT,
                        "accepted": None,
                    }
                )
                continue

            evaluation = _evaluate_pair(store, key_a, key_b, distance)
            pairs_report.append(
                {
                    "keys": [key_a, key_b],
                    "distance": distance,
                    "accepted": None,
                    **evaluation,
                }
            )

    pairs_report.sort(key=lambda p: p["keys"])

    generated_at = now()
    report = {
        "root": str(store.root),
        "generated_at": generated_at.isoformat(),
        "thresholds": _thresholds_payload(),
        "pairs": pairs_report,
    }

    near_dup_dir = store.root / "near_dup"
    near_dup_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = near_dup_dir / f"report-{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = json_path.with_suffix(".md")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    return json_path


# --- AC4: near-dup apply ------------------------------------------------- #

# Winner-selection prefix rank (CONTEXT.md "Offline embedding-space
# near-duplicate batch merge" -- AC4): earlier wins. ``youtube:`` is out of
# scope for #140's pool membership rules but still ranked here per the
# issue's own winner-rank waterfall.
_WINNER_PREFIX_RANK = ("mbid:", "isrc:", "spotify:", "youtube:", "name:")


def _prefix_rank(key: str) -> int:
    for rank, prefix in enumerate(_WINNER_PREFIX_RANK):
        if key.startswith(prefix):
            return rank
    return len(_WINNER_PREFIX_RANK)


def _pick_winner(key_a: str, key_b: str) -> tuple[str, str]:
    """Returns ``(winner, loser)`` by prefix rank, tie -> lexicographically
    smaller wins."""
    rank_a, rank_b = _prefix_rank(key_a), _prefix_rank(key_b)
    if rank_a != rank_b:
        return (key_a, key_b) if rank_a < rank_b else (key_b, key_a)
    return (key_a, key_b) if key_a < key_b else (key_b, key_a)


def apply(report_path: Path) -> Path:
    """AC4: consume a *reviewed* scan report and write ``{loser, winner,
    tier}`` lines to ``aliases.jsonl`` beside ``audio_analysis/`` for
    ``accepted: true`` pairs only.

    Never touches ``audio_analysis/`` itself -- no analysis file is deleted
    or rewritten, no mtime is consulted. Idempotent: reapplying the same (or
    an overlapping) report does not duplicate alias lines for a loser key
    that already has one recorded.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    root = Path(report["root"])
    aliases_path = root / "aliases.jsonl"

    existing_losers: set[str] = set()
    if aliases_path.exists():
        for line in aliases_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_losers.add(json.loads(line)["loser"])

    new_lines: list[str] = []
    for pair in report["pairs"]:
        if pair.get("accepted") is not True:
            continue
        key_a, key_b = pair["keys"]
        winner, loser = _pick_winner(key_a, key_b)
        if loser in existing_losers:
            continue
        existing_losers.add(loser)
        new_lines.append(json.dumps({"loser": loser, "winner": winner, "tier": pair["tier"]}))

    if new_lines:
        with aliases_path.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")

    return aliases_path
