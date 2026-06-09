"""Honest-empty analyzer + generated_from correctness."""

from __future__ import annotations

from datetime import UTC, datetime

from music_intel_mcp.analyzer import analyze
from music_intel_mcp.models import ListenEvent, RootProfile

FIXED_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


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
