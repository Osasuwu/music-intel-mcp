"""V0 invariant gates (#74 — the automatable subset of #66 criteria 4 & 5).

End-to-end guardrails that must hold for *every* ``RootProfile`` ``analyze()``
emits, regardless of what (if anything) was derived. They assert behaviour that
is already locked in code and decisions, through the public ``analyze()`` seam —
so a regression in any pillar trips here instead of silently shipping:

- **honest-empty** — no enrichment → a valid, well-formed, round-trippable
  artifact with empty sections; roots are never fabricated (CONTEXT.md
  §Invariants).
- **transparent-rejection** — a candidate that *forms* but fails a hard floor
  lands in ``quality_log[]`` with a named ``failed_test``; nothing is silently
  dropped (decision ``bce66b6e``).
- **inert category_weights** — weights are recorded-but-unused in V0; identical
  whether or not roots were derived, and equal to the locked default (decision
  ``7b3adb41``).
- **schema canonised** — the emitted artifact's top-level shape matches the
  committed schema example (decision ``7b3adb41``).

Empirical threshold calibration and the real-data run stay in #66 (owner-gated).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from music_intel_mcp.analyzer import analyze
from music_intel_mcp.audio import InMemoryAudioFeatureSource
from music_intel_mcp.models import (
    AudioParams,
    ListenEvent,
    MethodParams,
    RootProfile,
    TrackRef,
    ValidationParams,
)
from music_intel_mcp.shared_store import AudioFeatures, InMemorySharedStore

FIXED_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)
EARLY = datetime(2025, 1, 20, tzinfo=UTC)
LATE = datetime(2025, 5, 20, tzinfo=UTC)

_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "schemas" / "root_profile.v0.example.json"

# Relaxed floors that let the small synthetic fixture below promote to roots —
# the "derivation happened" arm of the inert-weights comparison.
DERIVE_PARAMS = MethodParams(
    audio=AudioParams(min_cluster_size=5, min_samples=2),
    validation=ValidationParams(
        N_THRESHOLD=5, T_THRESHOLD_DAYS=5, evidence_count_floor=5, confidence_floor=0.5
    ),
)


def _example() -> dict:
    return json.loads(_EXAMPLE_PATH.read_text("utf-8"))


def _example_top_level_keys() -> set[str]:
    # `_comment` is schema documentation, not a model field (extra="forbid").
    return {k for k in _example() if not k.startswith("_comment")}


def _audio_events_and_source() -> tuple[list[ListenEvent], InMemoryAudioFeatureSource]:
    """Two tight feature blobs over 12 MBID-bearing tracks; each track played in
    both history halves. Mirrors tests/test_analyzer.py so the derivation path is
    a known-good one — these gates only assert invariants *around* it."""
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


# --------------------------------------------------------------------------- #
# honest-empty + schema canonised
# --------------------------------------------------------------------------- #


def test_honest_empty_is_valid_well_formed_and_canonical():
    profile = analyze([], user_id="u", generated_at=FIXED_NOW)

    # honest-empty: every derived section empty, nothing fabricated.
    assert profile.roots == []
    assert profile.tendencies == []
    assert profile.epochs == []
    assert profile.quality_log == []

    # ...yet still a complete, round-trippable artifact (extra="forbid" means
    # re-parsing the dump would reject any stray/missing field).
    assert RootProfile.model_validate_json(profile.model_dump_json()) == profile

    # proxy outputs are labelled as such; schema version pinned.
    assert profile.model_maturity == "proxy"
    assert profile.schema_version == "v0"

    # top-level shape matches the committed schema example.
    assert set(profile.model_dump().keys()) == _example_top_level_keys()


def test_derived_profile_matches_canonical_top_level_shape():
    events, source = _audio_events_and_source()
    profile = analyze(
        events,
        user_id="u",
        generated_at=FIXED_NOW,
        method_params=DERIVE_PARAMS,
        shared_store=InMemorySharedStore(),
        audio_source=source,
    )
    assert profile.roots, "fixture sanity: this arm must derive roots"
    # a populated profile carries the exact same top-level shape as an empty one.
    assert set(profile.model_dump().keys()) == _example_top_level_keys()


# --------------------------------------------------------------------------- #
# transparent-rejection
# --------------------------------------------------------------------------- #


def test_formed_but_rejected_candidates_surface_in_quality_log():
    """An impossibly high evidence-count floor lets the audio clusters *form*
    but trips the calibration gate (G) on every one — so they must appear in
    quality_log[], not vanish. This is the transparent-rejection invariant
    exercised end-to-end through analyze()."""
    events, source = _audio_events_and_source()
    params = MethodParams(
        audio=AudioParams(min_cluster_size=5, min_samples=2),
        validation=ValidationParams(evidence_count_floor=10_000),
    )
    profile = analyze(
        events,
        user_id="u",
        generated_at=FIXED_NOW,
        method_params=params,
        shared_store=InMemorySharedStore(),
        audio_source=source,
    )

    # honest-empty user-facing sections...
    assert profile.roots == []
    assert profile.tendencies == []
    # ...but the rejections are transparent, never silently dropped.
    assert profile.quality_log, "formed-but-rejected candidates must be logged"
    for entry in profile.quality_log:
        assert entry.category == "audio"
        assert entry.failed_test == "calibration"  # G fires before any other gate
        assert entry.details["floor"] == 10_000

    # still a schema-valid artifact even when everything was rejected.
    assert set(profile.model_dump().keys()) == _example_top_level_keys()


# --------------------------------------------------------------------------- #
# inert category_weights
# --------------------------------------------------------------------------- #


def test_category_weights_are_inert_regardless_of_derivation():
    """V0 records category_weights but never uses them: the emitted weights must
    not depend on what was derived. Compare an empty run against a roots-bearing
    run — identical weights prove they are inert (decision 7b3adb41)."""
    empty = analyze([], user_id="u", generated_at=FIXED_NOW)

    events, source = _audio_events_and_source()
    derived = analyze(
        events,
        user_id="u",
        generated_at=FIXED_NOW,
        method_params=DERIVE_PARAMS,
        shared_store=InMemorySharedStore(),
        audio_source=source,
    )
    assert derived.roots, "fixture sanity: this arm must derive roots"

    # identical whether or not roots were produced → recorded, not used.
    assert empty.category_weights == derived.category_weights
    # and equal to the locked V0 default (active categories 1.0, rest inert).
    assert derived.category_weights.audio == 1.0
    assert derived.category_weights.temporal == 1.0
    assert derived.category_weights.scene == 1.0
    assert derived.category_weights.timbre is None
    assert derived.category_weights.career_phase is None
    # matches the committed schema example exactly.
    assert derived.category_weights.model_dump() == _example()["category_weights"]
