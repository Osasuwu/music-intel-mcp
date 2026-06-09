"""Honest-empty analyzer + generated_from correctness, and the end-to-end
audio stage (#63) wired through `analyze`."""

from __future__ import annotations

from datetime import UTC, datetime

from music_intel_mcp.analyzer import analyze
from music_intel_mcp.audio import InMemoryAudioFeatureSource
from music_intel_mcp.models import (
    AudioParams,
    ListenEvent,
    MethodParams,
    RootProfile,
    SceneParams,
    TrackRef,
    ValidationParams,
)
from music_intel_mcp.scene import InMemoryTagSource
from music_intel_mcp.shared_store import AudioFeatures, InMemorySharedStore, TrackTag

FIXED_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)

# Relaxed thresholds open the temporal gate on a small fixture (see test_audio).
AUDIO_METHOD_PARAMS = MethodParams(
    audio=AudioParams(min_cluster_size=5, min_samples=2),
    validation=ValidationParams(
        N_THRESHOLD=5, T_THRESHOLD_DAYS=5, evidence_count_floor=5, confidence_floor=0.5
    ),
)
EARLY = datetime(2025, 1, 20, tzinfo=UTC)
LATE = datetime(2025, 5, 20, tzinfo=UTC)


def _load(history_sample_path):
    lines = history_sample_path.read_text("utf-8").splitlines()
    return [ListenEvent.model_validate_json(ln) for ln in lines if ln.strip()]


def test_analyze_empty_input_is_honest_empty():
    profile = analyze([], user_id="u", generated_at=FIXED_NOW)
    assert isinstance(profile, RootProfile)
    assert profile.roots == []
    assert profile.tendencies == []
    assert profile.epochs == []
    assert profile.generated_from.n_events == 0
    assert profile.generated_from.n_unique_tracks == 0
    assert profile.generated_from.history_range is None
    assert profile.model_maturity == "proxy"


def test_generated_from_counts_match_fixture(history_sample_path):
    events = _load(history_sample_path)
    profile = analyze(events, user_id="petr", generated_at=FIXED_NOW)
    gf = profile.generated_from
    assert gf.n_events == 5
    # 3 unique tracks: spotify AAA (x2), isrc USABC... (x2), mbid 1111... (x1)
    assert gf.n_unique_tracks == 3
    assert gf.data_sources == ["ifttt_csv", "lastfm"]
    assert gf.history_range is not None
    earliest, latest = gf.history_range
    assert earliest.year == 2025 and earliest.month == 1
    assert latest.month == 3
    assert gf.history_span_days == (latest - earliest).days
    # honest coverage: no enricher has run
    assert gf.coverage_per_category["audio"] == 0.0
    assert gf.coverage_per_category["temporal"] == 1.0


def test_snapshot_id_embeds_user_and_timestamp():
    profile = analyze([], user_id="petr", generated_at=FIXED_NOW)
    assert profile.snapshot_id.startswith("petr/")
    assert "2026-06-09" in profile.snapshot_id


# --------------------------------------------------------------------------- #
# Audio stage end-to-end (#63): events -> seed -> enrich -> cluster -> roots
# --------------------------------------------------------------------------- #


def _audio_events_and_source():
    """Two tight feature blobs over 12 MBID-bearing tracks; each track is played
    once in each history half (balanced -> high temporal stability)."""
    source: dict[str, AudioFeatures] = {}
    events: list[ListenEvent] = []
    for label, base in (("low", (78, 0.18, 0.22, 0.28)), ("high", (170, 0.88, 0.80, 0.83))):
        for i in range(6):
            mbid = f"{label}-{i}"
            bpm, energy, valence, dance = base
            source[mbid] = AudioFeatures(
                bpm=bpm + i * 0.4,
                energy=energy + i * 0.004,
                valence=valence + i * 0.004,
                danceability=dance + i * 0.004,
                acousticness=0.5,
                instrumentalness=0.1,
                source="synthetic",
            )
            track = TrackRef(mbid=mbid, name=f"{label} {i}", artist="A")
            for played_at in (EARLY, LATE):
                events.append(ListenEvent(track=track, played_at=played_at, source="lastfm"))
    return events, InMemoryAudioFeatureSource(source)


def test_analyze_runs_audio_stage_when_store_and_source_supplied():
    events, source = _audio_events_and_source()
    store = InMemorySharedStore()  # empty -> analyze seeds it from the events

    profile = analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=AUDIO_METHOD_PARAMS,
        shared_store=store,
        audio_source=source,
    )

    assert len(profile.roots) == 2
    assert {r.category for r in profile.roots} == {"audio"}
    assert {r.id for r in profile.roots} == {"r-audio-1", "r-audio-2"}
    # every track enriched -> full audio coverage, reflected in generated_from
    assert profile.generated_from.coverage_per_category["audio"] == 1.0
    # descriptor carries the four bands
    bands = {r.structural_descriptor["bpm_band"] for r in profile.roots}
    assert bands == {"low", "high"}


def test_analyze_without_audio_source_stays_honest_empty():
    events, _ = _audio_events_and_source()
    profile = analyze(events, user_id="petr", generated_at=FIXED_NOW)
    assert profile.roots == []
    assert profile.generated_from.coverage_per_category["audio"] == 0.0


# --------------------------------------------------------------------------- #
# Scene stage end-to-end (#64): events -> seed -> tag-enrich -> NMF -> roots
# --------------------------------------------------------------------------- #

SCENE_METHOD_PARAMS = MethodParams(
    scene=SceneParams(K_grid_explored=[2]),
    validation=ValidationParams(
        N_THRESHOLD=5, T_THRESHOLD_DAYS=5, evidence_count_floor=5, confidence_floor=0.5
    ),
)
_SCENE_TAGS = {
    "metal": ["metal", "thrash", "heavy"],
    "jazz": ["jazz", "bebop", "swing"],
}


def _scene_events_and_source():
    """Two coherent tag blobs over 12 tracks; each played in both history halves."""
    mapping: dict[tuple[str, str], list[TrackTag]] = {}
    events: list[ListenEvent] = []
    for label, tags in _SCENE_TAGS.items():
        artist = f"{label}-band"
        for i in range(6):
            name = f"{label} {i}"
            mapping[(artist, name)] = [TrackTag(tag=t, weight=1.0, source="lastfm") for t in tags]
            track = TrackRef(mbid=f"{label}-{i}", name=name, artist=artist)
            for played_at in (EARLY, LATE):
                events.append(ListenEvent(track=track, played_at=played_at, source="lastfm"))
    return events, InMemoryTagSource(mapping)


def test_analyze_runs_scene_stage_when_store_and_tag_source_supplied():
    events, tag_source = _scene_events_and_source()
    store = InMemorySharedStore()

    profile = analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=SCENE_METHOD_PARAMS,
        shared_store=store,
        tag_source=tag_source,
    )

    assert len(profile.roots) == 2
    assert {r.category for r in profile.roots} == {"scene"}
    assert {r.id for r in profile.roots} == {"r-scene-1", "r-scene-2"}
    assert profile.generated_from.coverage_per_category["scene"] == 1.0
    # the chosen K is recorded back into method_params for provenance
    assert profile.method_params.scene.K_selected == 2


def test_analyze_does_not_mutate_caller_method_params():
    events, tag_source = _scene_events_and_source()
    analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=SCENE_METHOD_PARAMS,
        shared_store=InMemorySharedStore(),
        tag_source=tag_source,
    )
    # recording K_selected on the deep copy must not touch the module-level params
    assert SCENE_METHOD_PARAMS.scene.K_selected is None


def test_analyze_without_tag_source_stays_honest_empty_scene():
    events, _ = _scene_events_and_source()
    profile = analyze(events, user_id="petr", generated_at=FIXED_NOW)
    assert profile.roots == []
    assert profile.generated_from.coverage_per_category["scene"] == 0.0


def test_analyze_runs_audio_and_scene_together():
    """Both enrichers wired → both categories surface in one profile, on the same
    seeded store."""
    audio_events, audio_source = _audio_events_and_source()
    scene_events, tag_source = _scene_events_and_source()
    params = MethodParams(
        audio=AudioParams(min_cluster_size=5, min_samples=2),
        scene=SceneParams(K_grid_explored=[2]),
        validation=ValidationParams(
            N_THRESHOLD=5, T_THRESHOLD_DAYS=5, evidence_count_floor=5, confidence_floor=0.5
        ),
    )
    store = InMemorySharedStore()

    profile = analyze(
        audio_events + scene_events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=params,
        shared_store=store,
        audio_source=audio_source,
        tag_source=tag_source,
    )

    categories = {r.category for r in profile.roots}
    assert categories == {"audio", "scene"}
    # 24 tracks total; each enricher covers only its own 12 → honest 0.5 each
    assert profile.generated_from.coverage_per_category["audio"] == 0.5
    assert profile.generated_from.coverage_per_category["scene"] == 0.5
