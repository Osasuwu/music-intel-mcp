"""Offline embedding-space near-duplicate batch merge (#140).

AC4: ``near-dup apply <report>`` consumes a *reviewed* scan report and writes
``{loser, winner, tier}`` lines to ``aliases.jsonl`` beside ``audio_analysis/``
for ``accepted: true`` pairs only. It never merges, deletes, or rewrites
anything under ``audio_analysis/`` -- that directory must be byte-identical
before and after ``apply`` runs.
"""

from __future__ import annotations

import json

from music_intel_mcp.near_dup import apply, scan
from music_intel_mcp.store import UserStore


def _write(store: UserStore, track_id: str, embedding: list[float]) -> None:
    store.write_audio_analysis(track_id=track_id, embedding=embedding, tags={})


def _build_store_with_scan_report(tmp_path):
    store = UserStore(root=tmp_path / "root")
    _write(store, "spotify:dup", [1.0, 0.0, 0.0, 0.0])
    _write(store, "name:dup|artist", [1.0, 0.0, 0.0, 0.0])
    dup_fp = list(range(300))
    store.write_fingerprint(track_id="spotify:dup", fingerprint=dup_fp, duration_s=30.0)
    store.write_fingerprint(track_id="name:dup|artist", fingerprint=dup_fp, duration_s=30.0)

    json_path = scan(store)
    return store, json_path


def _mark_accepted(json_path, key_a: str, key_b: str, accepted: bool) -> None:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    wanted = sorted((key_a, key_b))
    for pair in report["pairs"]:
        if pair["keys"] == wanted:
            pair["accepted"] = accepted
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _read_aliases(store: UserStore) -> list[dict]:
    aliases_path = store.root / "aliases.jsonl"
    if not aliases_path.exists():
        return []
    lines = aliases_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_apply_writes_alias_for_accepted_pair(tmp_path):
    store, json_path = _build_store_with_scan_report(tmp_path)
    _mark_accepted(json_path, "spotify:dup", "name:dup|artist", True)

    aliases_path = apply(json_path)

    assert aliases_path == store.root / "aliases.jsonl"
    aliases = _read_aliases(store)
    assert len(aliases) == 1
    # spotify: outranks name: in the winner-prefix-rank table.
    assert aliases[0] == {
        "loser": "name:dup|artist",
        "winner": "spotify:dup",
        "tier": "fingerprint+embedding",
    }


def test_apply_ignores_pairs_not_explicitly_accepted(tmp_path):
    store, json_path = _build_store_with_scan_report(tmp_path)
    # accepted stays None (the scan default) -- never touched by a reviewer.

    apply(json_path)

    assert _read_aliases(store) == []


def test_apply_ignores_explicitly_rejected_pairs(tmp_path):
    store, json_path = _build_store_with_scan_report(tmp_path)
    _mark_accepted(json_path, "spotify:dup", "name:dup|artist", False)

    apply(json_path)

    assert _read_aliases(store) == []


def test_apply_is_idempotent(tmp_path):
    store, json_path = _build_store_with_scan_report(tmp_path)
    _mark_accepted(json_path, "spotify:dup", "name:dup|artist", True)

    apply(json_path)
    apply(json_path)

    assert len(_read_aliases(store)) == 1


def test_apply_never_touches_audio_analysis_files(tmp_path):
    store, json_path = _build_store_with_scan_report(tmp_path)
    _mark_accepted(json_path, "spotify:dup", "name:dup|artist", True)

    before = {p.name: p.read_bytes() for p in sorted(store.audio_analysis_dir.glob("*.json"))}

    apply(json_path)

    after = {p.name: p.read_bytes() for p in sorted(store.audio_analysis_dir.glob("*.json"))}
    assert before == after


def test_apply_winner_by_prefix_rank_mbid_beats_isrc(tmp_path):
    store = UserStore(root=tmp_path / "root")
    _write(store, "mbid:11111111-1111-1111-1111-111111111111", [1.0, 0.0, 0.0, 0.0])
    _write(store, "isrc:US1234567890", [1.0, 0.0, 0.0, 0.0])
    json_path = scan(store)
    _mark_accepted(
        json_path, "mbid:11111111-1111-1111-1111-111111111111", "isrc:US1234567890", True
    )

    apply(json_path)

    aliases = _read_aliases(store)
    assert len(aliases) == 1
    assert aliases[0]["winner"] == "mbid:11111111-1111-1111-1111-111111111111"
    assert aliases[0]["loser"] == "isrc:US1234567890"


def test_apply_tiebreak_same_prefix_lexicographically_smaller_wins(tmp_path):
    store = UserStore(root=tmp_path / "root")
    _write(store, "spotify:bbbbbbbbbbbbbbbbbbbbbb", [1.0, 0.0, 0.0, 0.0])
    _write(store, "spotify:aaaaaaaaaaaaaaaaaaaaaa", [1.0, 0.0, 0.0, 0.0])
    json_path = scan(store)
    _mark_accepted(
        json_path, "spotify:bbbbbbbbbbbbbbbbbbbbbb", "spotify:aaaaaaaaaaaaaaaaaaaaaa", True
    )

    apply(json_path)

    aliases = _read_aliases(store)
    assert len(aliases) == 1
    assert aliases[0]["winner"] == "spotify:aaaaaaaaaaaaaaaaaaaaaa"
    assert aliases[0]["loser"] == "spotify:bbbbbbbbbbbbbbbbbbbbbb"
