"""Honest-empty analyzer + generated_from correctness, and the end-to-end
audio stage (#63) wired through `analyze`."""

from __future__ import annotations

from datetime import UTC, datetime

from music_intel_mcp.analyzer import _group_and_seed, analyze
from music_intel_mcp.audio import InMemoryAudioFeatureSource
from music_intel_mcp.identity import (
    IdentityResolver,
    InMemoryIsrcMbidIndex,
    InMemorySpotifyIsrcSource,
)
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


def test_analyze_surfaces_audio_enrichment_diagnostics(history_sample_path):
    """#87 AC-E: the enrichment buckets (enriched / missing_features / no_mbid)
    reach the persisted RootProfile so full-library MBID coverage and
    P(features|MBID) are measurable off a stored profile — not lost inside the
    analyzer. Mixed fixture: 1 MBID track the source covers (enriched), 1 MBID
    track it does not (missing_features), 1 name-only track (no_mbid)."""
    events = [
        ListenEvent(track=TrackRef(mbid="has", name="h", artist="A"), played_at=EARLY, source="s"),
        ListenEvent(track=TrackRef(mbid="gap", name="g", artist="A"), played_at=EARLY, source="s"),
        ListenEvent(track=TrackRef(name="n", artist="A"), played_at=EARLY, source="s"),
    ]
    source = InMemoryAudioFeatureSource(
        {"has": AudioFeatures(bpm=120, energy=0.5, valence=0.5, danceability=0.5, source="syn")}
    )

    profile = analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        shared_store=InMemorySharedStore(),
        audio_source=source,
    )

    diag = profile.generated_from.enrichment_diagnostics["audio"]
    assert diag == {"enriched": 1, "already_present": 0, "missing_features": 1, "no_mbid": 1}
    # MBID coverage and P(features|MBID) derive from the surfaced buckets.
    total = sum(diag.values())
    assert (total - diag["no_mbid"]) / total == 2 / 3  # 2 of 3 tracks carry an MBID


def test_analyze_without_enricher_reports_no_diagnostics():
    """Honest-empty: no audio source wired -> no audio diagnostics fabricated."""
    profile = analyze(
        [ListenEvent(track=TrackRef(mbid="m", name="n", artist="A"), played_at=EARLY, source="s")],
        user_id="petr",
        generated_at=FIXED_NOW,
    )
    assert "audio" not in profile.generated_from.enrichment_diagnostics


# --------------------------------------------------------------------------- #
# P9 resolver -> analyze bridge (#87 AC-C): events carry only a spotify_id; the
# MBID the enricher joins on lives behind the identity waterfall. Without the
# bridge audio stays 0%; with it, the resolved MBID unblocks enrichment.
# --------------------------------------------------------------------------- #


def _spotify_only_audio_events_and_sources():
    """The `_audio_events_and_source` fixture, re-expressed as the real history
    shape: every event carries ONLY a spotify_id. The MBID (which the audio
    source is keyed by) is reachable only through spotify_id -> ISRC -> MBID, so
    a resolver is the sole path to enrichment."""
    features: dict[str, AudioFeatures] = {}
    spotify_to_isrc: dict[str, str] = {}
    isrc_to_mbid: dict[str, str] = {}
    events: list[ListenEvent] = []
    for label, base in (("low", (78, 0.18, 0.22, 0.28)), ("high", (170, 0.88, 0.80, 0.83))):
        for i in range(6):
            mbid, sid, isrc = f"{label}-{i}", f"S-{label}-{i}", f"I-{label}-{i}"
            spotify_to_isrc[sid] = isrc
            isrc_to_mbid[isrc] = mbid
            bpm, energy, valence, dance = base
            features[mbid] = AudioFeatures(
                bpm=bpm + i * 0.4,
                energy=energy + i * 0.004,
                valence=valence + i * 0.004,
                danceability=dance + i * 0.004,
                acousticness=0.5,
                instrumentalness=0.1,
                source="synthetic",
            )
            track = TrackRef(spotify_id=sid, name=f"{label} {i}", artist="A")
            for played_at in (EARLY, LATE):
                events.append(
                    ListenEvent(track=track, played_at=played_at, source="spotify_extended")
                )
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex(isrc_to_mbid),
        spotify_source=InMemorySpotifyIsrcSource(spotify_to_isrc),
    )
    return events, InMemoryAudioFeatureSource(features), resolver


def test_analyze_bridges_resolver_output_into_audio_enrichment():
    """With a resolver wired, spotify-only events are walked to their MBID and
    enriched — audio roots surface where the raw ingest identity had none."""
    events, source, resolver = _spotify_only_audio_events_and_sources()
    store = InMemorySharedStore()

    profile = analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=AUDIO_METHOD_PARAMS,
        shared_store=store,
        audio_source=source,
        resolver=resolver,
    )

    assert len(profile.roots) == 2
    assert {r.category for r in profile.roots} == {"audio"}
    assert profile.generated_from.coverage_per_category["audio"] == 1.0
    assert {r.structural_descriptor["bpm_band"] for r in profile.roots} == {"low", "high"}


def test_analyze_spotify_only_without_resolver_stays_honest_empty():
    """The control: the same spotify-only events, no resolver -> the raw
    ingest identity carries no MBID -> audio is honest-empty. Proves the bridge
    is the thing that unblocks enrichment, not the fixture."""
    events, source, _ = _spotify_only_audio_events_and_sources()
    profile = analyze(
        events,
        user_id="petr",
        generated_at=FIXED_NOW,
        method_params=AUDIO_METHOD_PARAMS,
        shared_store=InMemorySharedStore(),
        audio_source=source,
    )
    assert profile.roots == []
    assert profile.generated_from.coverage_per_category["audio"] == 0.0


def test_group_and_seed_rekeys_and_merges_by_resolved_mbid():
    """F7 key-space rebuild: events with *different* raw identities that resolve
    to the same MBID collapse onto one ``mbid:``-keyed group with merged plays,
    and the seeded record carries the resolved MBID for the enricher to join."""
    resolver = IdentityResolver(
        InMemoryIsrcMbidIndex({"I-1": "M-1"}),
        spotify_source=InMemorySpotifyIsrcSource({"S-1": "I-1"}),
    )
    events = [
        ListenEvent(
            track=TrackRef(spotify_id="S-1", name="n", artist="a"),
            played_at=EARLY,
            source="spotify_extended",
        ),
        ListenEvent(
            track=TrackRef(isrc="I-1", name="n", artist="a"),
            played_at=LATE,
            source="lastfm",
        ),
    ]
    store = InMemorySharedStore()

    plays, unique_ids = _group_and_seed(events, store, FIXED_NOW, resolver)

    assert unique_ids == ["mbid:M-1"]  # both raw ids rebuilt onto the MBID
    assert plays["mbid:M-1"] == [EARLY, LATE]  # plays merged, not two groups
    seeded = store.get_tracks(["mbid:M-1"])["mbid:M-1"]
    assert seeded.mbid == "M-1"


def test_group_and_seed_without_resolver_keeps_raw_identity():
    """No resolver -> the raw ingest identity is preserved unchanged (the
    honest-empty V0 path); a spotify-only event stays ``spotify:``-keyed."""
    events = [
        ListenEvent(
            track=TrackRef(spotify_id="S-1", name="n", artist="a"),
            played_at=EARLY,
            source="spotify_extended",
        )
    ]
    plays, unique_ids = _group_and_seed(events, InMemorySharedStore(), FIXED_NOW)
    assert unique_ids == ["spotify:S-1"]
    assert plays["spotify:S-1"] == [EARLY]


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
