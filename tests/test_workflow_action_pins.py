"""Every external `uses:` in .github/workflows is pinned to a 40-hex commit SHA (#233)."""

import re
from pathlib import Path

import pytest

_root = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "workflows").is_dir())
WORKFLOWS = sorted((_root / ".github" / "workflows").glob("*.yml"))

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)", re.MULTILINE)
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def unpinned(text: str) -> list[str]:
    refs = (m.group("ref").strip("'\"") for m in USES.finditer(text))
    return [
        r
        for r in refs
        if not r.startswith("./") and not r.startswith("docker://") and not PINNED.match(r)
    ]


def test_workflows_found():
    assert WORKFLOWS


def test_detector_flags_tag_and_short_sha():
    assert unpinned("- uses: actions/checkout@v7\n") == ["actions/checkout@v7"]
    assert unpinned("  uses: a/b@" + "a" * 39 + "\n") == ["a/b@" + "a" * 39]


def test_detector_accepts_sha_local_and_docker():
    text = f"- uses: a/b@{'a' * 40} # v1\n  uses: ./.github/actions/x\n  uses: docker://alpine:3\n"
    assert unpinned(text) == []


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_action_pinned_to_sha(path):
    assert unpinned(path.read_text(encoding="utf-8")) == []
