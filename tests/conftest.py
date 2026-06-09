"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def history_sample_path() -> Path:
    return FIXTURES / "history_sample.jsonl"


@pytest.fixture
def canonical_profile_dict() -> dict:
    """The canonical RootProfile example, with the leading ``_comment`` key
    stripped so it validates against the (extra-forbidding) models."""
    raw = json.loads((REPO_ROOT / "schemas" / "root_profile.v0.example.json").read_text("utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}
