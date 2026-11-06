"""Metadata cross-walk alias generation from the ISRC index (#178).

Index-based, no live API at run time: every fixture here is an in-memory
dict-backed index/source (``InMemoryIsrcMbidIndex``, a plain callable for
the spotify->isrc leg) -- no network, no ``SpotifyApiIsrcSource`` HTTP path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from music_intel_mcp.identity import InMemoryIsrcMbidIndex
from music_intel_mcp.inference import EMBEDDING_SPACE_VERSION
from music_intel_mcp.metadata_crosswalk import run_metadata_crosswalk
from music_intel_mcp.models import ListenEvent, TrackRef
from music_intel_mcp.replay_queue import replay_queue_coverage
from music_intel_mcp.shared_store import canonical_track_id
from music_intel_mcp.store import AudioAnalysisRecord, UserStore
from music_intel_mcp.timbre import TimbreSource, select_timbre_records


def _event(spotify_id: str, *, name: str = "Track", artist: str = "Artist") -> ListenEvent:
    return ListenEvent(
        track=TrackRef(spotify_id=spotify_id, name=name, artist=artist),
        played_at=datetime(2026, 1, 1, tzinfo=UTC),
        source="test",
    )


def _write_history(root, events: list[ListenEvent]) -> None:
    path = root / "history.jsonl"
    path.write_text(
        "\n".join(e.model_dump_json() for e in events) + "\n",
        encoding="utf-8",
    )


def test_emits_alias_line_for_history_only_key_resolvable_via_isrc_index(tmp_path):
    """AC1: a CLI-invocable step takes a data root and emits
    ``spotify:<id> -> mbid:<uuid>`` alias lines for history keys resolvable
    through the ISRC index -- fixture index, no network."""
    _write_history(tmp_path, [_event("sp1")])
    isrc_index = InMemoryIsrcMbidIndex({"ISRC0001": "mbid-aaa"})

    result = run_metadata_crosswalk(
        tmp_path,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001"}.get,
    )

    assert result.aliases_written == [
        {"loser": "spotify:sp1", "winner": "mbid:mbid-aaa", "tier": "metadata_crosswalk"}
    ]
    aliases_path = tmp_path / "aliases.jsonl"
    lines = [json.loads(line) for line in aliases_path.read_text(encoding="utf-8").splitlines()]
    assert lines == [
        {"loser": "spotify:sp1", "winner": "mbid:mbid-aaa", "tier": "metadata_crosswalk"}
    ]


def test_no_isrc_or_no_mbid_match_emits_nothing(tmp_path):
    _write_history(tmp_path, [_event("sp-unknown")])
    isrc_index = InMemoryIsrcMbidIndex({})

    result = run_metadata_crosswalk(
        tmp_path, isrc_index=isrc_index, spotify_isrc_lookup=lambda _sid: None
    )

    assert result.aliases_written == []
    assert not (tmp_path / "aliases.jsonl").exists()


def test_idempotent_does_not_duplicate_existing_alias_line(tmp_path):
    """AC2: existing lines are never duplicated on re-run."""
    _write_history(tmp_path, [_event("sp1")])
    aliases_path = tmp_path / "aliases.jsonl"
    aliases_path.write_text(
        json.dumps(
            {"loser": "spotify:sp1", "winner": "mbid:mbid-aaa", "tier": "metadata_crosswalk"}
        )
        + "\n",
        encoding="utf-8",
    )
    isrc_index = InMemoryIsrcMbidIndex({"ISRC0001": "mbid-aaa"})

    result = run_metadata_crosswalk(
        tmp_path,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001"}.get,
    )

    assert result.aliases_written == []
    lines = aliases_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_ambiguous_isrc_is_reported_not_aliased(tmp_path):
    """AC4: an ISRC resolving to multiple MBIDs is reported, not aliased."""
    _write_history(tmp_path, [_event("sp1")])
    isrc_index = InMemoryIsrcMbidIndex({"ISRC0001": ["mbid-aaa", "mbid-bbb"]})

    result = run_metadata_crosswalk(
        tmp_path,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001"}.get,
    )

    assert result.aliases_written == []
    assert not (tmp_path / "aliases.jsonl").exists()
    assert len(result.ambiguous) == 1
    ambiguous = result.ambiguous[0]
    assert ambiguous.spotify_id == "sp1"
    assert ambiguous.isrc == "ISRC0001"
    assert ambiguous.mbids == ["mbid-aaa", "mbid-bbb"]


def test_has_audio_analysis_treats_aliased_history_key_as_mbid_record(tmp_path):
    """AC3: after the crosswalk step, ``has_audio_analysis`` treats the
    aliased history key as the ``mbid:`` record."""
    _write_history(tmp_path, [_event("sp1")])
    isrc_index = InMemoryIsrcMbidIndex({"ISRC0001": "mbid-aaa"})
    run_metadata_crosswalk(
        tmp_path,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001"}.get,
    )

    store = UserStore(root=tmp_path)
    store.audio_analysis_dir.mkdir(parents=True, exist_ok=True)
    (store.audio_analysis_dir / "mbid_mbid-aaa.json").write_text(
        json.dumps({"track_id": "mbid:mbid-aaa"}), encoding="utf-8"
    )

    assert store.has_audio_analysis("spotify:sp1") is True


def test_replay_queue_coverage_treats_aliased_history_key_as_analyzed(tmp_path):
    """AC3: the replay queue selector treats the aliased history key as the
    ``mbid:`` record -- composability check against the resolve_mbid seam."""
    events = [_event("sp1") for _ in range(5)]
    _write_history(tmp_path, events)
    isrc_index = InMemoryIsrcMbidIndex({"ISRC0001": "mbid-aaa"})
    run_metadata_crosswalk(
        tmp_path,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001"}.get,
    )

    store = UserStore(root=tmp_path)

    def resolve_mbid(track: TrackRef) -> str | None:
        from music_intel_mcp.shared_store import canonical_track_id

        resolved = store.resolve_track_key(canonical_track_id(track))
        return resolved.split(":", 1)[1] if resolved.startswith("mbid:") else None

    analyzed_ids = {"mbid:mbid-aaa"}
    stats = replay_queue_coverage(
        events,
        has_audio_analysis=lambda key: key in analyzed_ids,
        resolve_mbid=resolve_mbid,
    )

    assert stats.already_analyzed_count == 1
    assert stats.queued_count == 0


def test_timbre_selection_treats_aliased_history_key_as_mbid(tmp_path):
    """AC3 (#264: through ``select_timbre_records``): the (root ∪ pool) ∩
    history selection treats the aliased history key as the ``mbid:``
    record. Three history-only spotify keys alias to three pool-analyzed
    mbids; three more history tracks are analyzed under their own spotify
    keys. Unresolved membership would select only the latter three."""
    root = tmp_path / "user"
    root.mkdir()
    other_events = [_event(f"other{i}") for i in range(3)]
    _write_history(
        root,
        [_event("sp1"), _event("sp2"), _event("sp3"), *other_events],
    )
    isrc_index = InMemoryIsrcMbidIndex(
        {"ISRC0001": "mbid-aaa", "ISRC0002": "mbid-bbb", "ISRC0003": "mbid-ccc"}
    )
    run_metadata_crosswalk(
        root,
        isrc_index=isrc_index,
        spotify_isrc_lookup={"sp1": "ISRC0001", "sp2": "ISRC0002", "sp3": "ISRC0003"}.get,
    )
    store = UserStore(root=root)

    other_keys = [canonical_track_id(e.track) for e in other_events]
    records = [
        AudioAnalysisRecord(
            track_id=key, embedding=[0.0, 0.0], tags={}, model_version=EMBEDDING_SPACE_VERSION
        )
        for key in ["mbid:mbid-aaa", "mbid:mbid-bbb", "mbid:mbid-ccc", *other_keys]
    ]

    selection = select_timbre_records(
        store.load_history(),
        TimbreSource(records=records, resolve_key=store.resolve_track_key),
    )

    assert sorted(r.track_id for r in selection.records) == [
        "mbid:mbid-aaa",
        "mbid:mbid-bbb",
        "mbid:mbid-ccc",
        "spotify:other0",
        "spotify:other1",
        "spotify:other2",
    ]
    assert selection.coverage == 1.0
