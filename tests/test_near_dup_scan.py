"""Offline embedding-space near-duplicate batch merge (#140).

AC2: ``near-dup scan <root>`` writes a transparent report over one
``UserStore`` root -- every considered pair, including rejections, never a
merge itself. AC3's "a pair with one missing [fingerprint] array lands in
``embedding_only``" sub-clause is a scan-level tier-assignment behavior, so
it is covered here rather than in ``test_near_dup.py``'s pure
``match_fingerprints`` suite.

AC6: no live API/model/fpcalc call -- everything here is synthetic
embeddings/fingerprints written straight through ``UserStore``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from music_intel_mcp.near_dup import (
    HUB_GUARD_NEIGHBOR_COUNT,
    REASON_ABOVE_THRESHOLD,
    REASON_HUB_SUSPECT,
    TIER_EMBEDDING_ONLY,
    TIER_FINGERPRINT_EMBEDDING,
    VERDICT_PROPOSED,
    VERDICT_REJECTED,
    scan,
)
from music_intel_mcp.store import UserStore


def _fp(n: int, seed: int = 0) -> list[int]:
    state = seed or 1
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out.append(state)
    return out


def _write(store: UserStore, track_id: str, embedding: list[float]) -> None:
    store.write_audio_analysis(track_id=track_id, embedding=embedding, tags={})


def _find(pairs: list[dict], key_a: str, key_b: str) -> dict:
    wanted = sorted((key_a, key_b))
    for pair in pairs:
        if pair["keys"] == wanted:
            return pair
    raise AssertionError(f"no report entry for pair {wanted}")


def _build_synthetic_store(tmp_path) -> UserStore:
    store = UserStore(root=tmp_path / "root")

    # Planted duplicate: identical embedding + identical fingerprint.
    dup_fp = _fp(300, seed=1)
    _write(store, "spotify:dup", [1.0, 0.0, 0.0, 0.0])
    _write(store, "name:dup|artist", [1.0, 0.0, 0.0, 0.0])
    store.write_fingerprint(track_id="spotify:dup", fingerprint=dup_fp, duration_s=30.0)
    store.write_fingerprint(track_id="name:dup|artist", fingerprint=dup_fp, duration_s=30.0)

    # Near-miss: embedding_only tier (no fingerprints), distance just over
    # the stricter embedding-only threshold.
    _write(store, "spotify:missa", [0.0, 1.0, 0.0, 0.0])
    _write(store, "spotify:missb", [0.0, 0.94, 0.3412, 0.0])

    # Hub: four records sharing (near-)identical embeddings -- each has
    # HUB_GUARD_NEIGHBOR_COUNT neighbours at distance ~0, so every pair
    # touching this cluster must be rejected hub_suspect rather than
    # proposed as a batch of false merges.
    assert HUB_GUARD_NEIGHBOR_COUNT <= 3
    for hub_id in ("spotify:hub", "spotify:h1", "spotify:h2", "spotify:h3"):
        _write(store, hub_id, [0.0, 0.0, 0.0, 1.0])

    # AC3 sub-clause: one side of the pair has no fingerprint sidecar at
    # all -- must land in embedding_only, not fingerprint+embedding, even
    # though the other side does have one.
    ma_fp = _fp(300, seed=2)
    _write(store, "spotify:ma_a", [1.0, 1.0, 0.0, 0.0])
    _write(store, "spotify:ma_b", [1.0, 1.0001, 0.0, 0.0])
    store.write_fingerprint(track_id="spotify:ma_a", fingerprint=ma_fp, duration_s=30.0)

    return store


def test_scan_writes_json_and_markdown_reports(tmp_path):
    store = _build_synthetic_store(tmp_path)
    fixed_now = datetime(2026, 9, 6, 12, 0, 0, tzinfo=UTC)

    json_path = scan(store, now=lambda: fixed_now)

    assert json_path.exists()
    assert json_path.parent == store.root / "near_dup"
    assert json_path.name == "report-20260906T120000Z.json"
    md_path = json_path.with_suffix(".md")
    assert md_path.exists()
    assert md_path.read_text(encoding="utf-8").startswith("# Near-duplicate scan report")


def test_scan_proposes_planted_duplicate_via_fingerprint_confirmation(tmp_path):
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    pair = _find(report["pairs"], "spotify:dup", "name:dup|artist")
    assert pair["verdict"] == VERDICT_PROPOSED
    assert pair["reason"] is None
    assert pair["tier"] == TIER_FINGERPRINT_EMBEDDING
    assert pair["fingerprint"]["ber"] == 0.0
    assert pair["accepted"] is None


def test_scan_rejects_near_miss_above_threshold(tmp_path):
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    pair = _find(report["pairs"], "spotify:missa", "spotify:missb")
    assert pair["verdict"] == VERDICT_REJECTED
    assert pair["reason"] == REASON_ABOVE_THRESHOLD
    assert pair["tier"] == TIER_EMBEDDING_ONLY
    assert pair["fingerprint"] is None
    assert pair["accepted"] is None


def test_scan_rejects_hub_suspect_pairs(tmp_path):
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    pair = _find(report["pairs"], "spotify:hub", "spotify:h1")
    assert pair["verdict"] == VERDICT_REJECTED
    assert pair["reason"] == REASON_HUB_SUSPECT
    assert pair["accepted"] is None


def test_scan_pair_with_one_missing_fingerprint_lands_embedding_only(tmp_path):
    """AC3: a pair where only one side has a raw fingerprint array cannot be
    fingerprint-confirmed -- it must fall to the embedding_only tier rather
    than being silently dropped or treated as a full fingerprint match."""
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    pair = _find(report["pairs"], "spotify:ma_a", "spotify:ma_b")
    assert pair["tier"] == TIER_EMBEDDING_ONLY
    assert pair["fingerprint"] is None
    assert pair["verdict"] == VERDICT_PROPOSED
    assert pair["accepted"] is None


def test_scan_rejects_pair_with_only_one_side_hub_suspect(tmp_path):
    """A candidate pair only needs one side to be a hub suspect to be
    rejected -- the guard must not require both sides to be degenerate
    (that would let a hub member's spurious near-match against a genuine,
    unrelated record slip through as an ordinary embedding_only proposal
    instead of being flagged hub_suspect).

    ``spotify:h1`` (hub member, 3 neighbours at ~zero distance -> hub
    suspect) and ``name:dup|artist`` (an ordinary record, only 1 neighbour
    at ~zero distance -> not a hub suspect) land in each other's top-K
    candidate set because every non-hub record is equidistant from the hub
    cluster; this pair must still be rejected hub_suspect."""
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    pair = _find(report["pairs"], "spotify:h1", "name:dup|artist")
    assert pair["verdict"] == VERDICT_REJECTED
    assert pair["reason"] == REASON_HUB_SUSPECT
    assert pair["accepted"] is None


def test_scan_reports_named_thresholds(tmp_path):
    store = _build_synthetic_store(tmp_path)

    json_path = scan(store, now=lambda: datetime(2026, 9, 6, tzinfo=UTC))
    report = json.loads(json_path.read_text(encoding="utf-8"))

    thresholds = report["thresholds"]
    assert thresholds["fingerprint_ber_threshold"] == 0.25
    assert thresholds["fingerprint_min_overlap_frames"] == 240
    assert "cosine_distance_fingerprint_embedding" in thresholds
    assert "cosine_distance_embedding_only" in thresholds
    assert "hub_guard_neighbor_count" in thresholds
    assert "hub_guard_distance_eps" in thresholds
