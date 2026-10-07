"""Decision rules of .github/scripts/code_gate_verdict.py — the "Review evidence" gate.

Every function under test is pure: plain dicts in, a value out. The fixtures
are shaped like real GitHub API payloads (workflow runs, pull requests, PR
files) and every expected value is a literal worked out from the locked design,
not read back from the module.
"""

import importlib.util
from pathlib import Path

import pytest

_root = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "scripts").is_dir())
_spec = importlib.util.spec_from_file_location(
    "code_gate_verdict", _root / ".github" / "scripts" / "code_gate_verdict.py"
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

SHA = "a" * 40
OTHER_SHA = "b" * 40
PR = 7


# --- classify_paths -------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # music-intel-mcp: docs/domain/ is cosmetic (product docs)
        "docs/domain/agent-play-exclusion.md",
        "docs/domain/analyze-cli.md",
        "docs/domain/audio-root-pipeline.md",
        "docs/domain/nested/deep.md",
        # Images anywhere: cosmetic
        "docs/img/arch.png",
        "docs/domain/deep/nested/shot.webp",
        "assets/logo.png",
        "src/pkg/icon.jpg",
        "anim.gif",
        "pic.webp",
        # Root cosmetic
        "README.md",
        "SECURITY.md",
        "LICENSE",
        "LICENSE-APACHE",
        "THIRD_PARTY_LICENSES",
    ],
)
def test_cosmetic_paths(path):
    assert gate.classify_paths([path]) == {"code": [], "cosmetic": [path]}


@pytest.mark.parametrize(
    "path",
    [
        # Behavior-carrying docs: code
        "docs/reference/github-repo-setup.md",
        "docs/COLLECTOR_SETUP.md",
        "docs/notes.txt",
        "docs/diagram.svg",
        # Root .md: AGENTS/CONTEXT/INVARIANTS are code (affect domain/behavior)
        "AGENTS.md",
        "INVARIANTS.md",
        "CONTEXT.md",
        # Non-root README: code
        "sub/README.md",
        # Config files: code
        "notes.txt",
        "logo.svg",
        # Agent/gate behavior: code
        ".claude/hooks/secret-scanner.py",
        ".claude/marketplace/.claude-plugin/marketplace.json",
        ".claude/settings.json",
        # Gate machinery: code
        ".github/workflows/pytest.yml",
        ".github/scripts/code_gate_verdict.py",
        # Product code: code
        "src/music_intel_mcp/app.py",
        "tests/test_inference.py",
        "native/wasapi_loopback_helper/helper.cpp",
        "schemas/findings.json",
        # Config files: code
        ".env.example",
        "pyproject.toml",
    ],
)
def test_code_paths(path):
    assert gate.classify_paths([path]) == {"code": [path], "cosmetic": []}


def test_classify_preserves_input_order_in_each_bucket():
    got = gate.classify_paths(["b.py", "docs/domain/a.md", "a.py", "docs/reference/b.md"])
    assert got == {
        "code": ["b.py", "a.py", "docs/reference/b.md"],
        "cosmetic": ["docs/domain/a.md"],
    }


# --- changed_paths --------------------------------------------------------


def test_rename_is_judged_on_both_paths():
    files = [{"filename": "docs/new.md", "previous_filename": "src/old.py", "status": "renamed"}]
    assert gate.changed_paths(files) == ["docs/new.md", "src/old.py"]


def test_rename_of_code_into_docs_is_not_all_cosmetic():
    # Rename into docs/domain/ (cosmetic) and docs/ (code) both matter
    files = [
        {"filename": "docs/domain/new.md", "previous_filename": "src/old.py", "status": "renamed"}
    ]
    classified = gate.classify_paths(gate.changed_paths(files))
    assert classified["code"] == ["src/old.py"]
    assert classified["cosmetic"] == ["docs/domain/new.md"]


def test_deleted_file_keeps_its_path():
    files = [{"filename": "src/gone.py", "status": "removed"}]
    assert gate.changed_paths(files) == ["src/gone.py"]


# --- validate_findings ----------------------------------------------------


def test_clean_findings_are_valid():
    assert gate.validate_findings({"blocking": False, "findings": []}) == []


def test_blocking_findings_are_valid():
    obj = {"blocking": True, "findings": [{"class": "regression", "file": "a.py"}]}
    assert gate.validate_findings(obj) == []


@pytest.mark.parametrize(
    "obj",
    [
        {"blocking": True, "findings": []},
        {"blocking": False, "findings": [{"class": "regression", "file": "a.py"}]},
        {"blocking": True, "findings": [{"class": "unknown", "file": "a.py"}]},
        {"blocking": True, "findings": [{"class": "regression"}]},
        {"blocking": "no", "findings": []},
        {"findings": []},
        {"blocking": False},
        {"status": "unreviewed"},
        [],
        None,
    ],
)
def test_invalid_findings_are_rejected(obj):
    assert gate.validate_findings(obj) != []


def test_all_finding_classes_are_accepted():
    classes = [
        "regression",
        "exception-handling",
        "intent-vs-logic",
        "breaking-contract",
        "concurrency",
        "requirement-semantics",
        "design-modularity",
        "performance",
    ]
    findings = [{"class": c, "file": "a.py"} for c in classes]
    assert gate.validate_findings({"blocking": True, "findings": findings}) == []


# --- check_clean_tree -----------------------------------------------------


def test_clean_tree_apart_from_findings_file():
    porcelain = "?? .review/findings.json\n"
    assert gate.check_clean_tree(porcelain, ".review/findings.json") == []


def test_modified_tracked_file_is_dirty():
    porcelain = " M .github/workflows/pytest.yml\n?? .review/findings.json\n"
    assert gate.check_clean_tree(porcelain, ".review/findings.json") == [
        ".github/workflows/pytest.yml"
    ]


def test_stray_untracked_file_is_dirty():
    porcelain = "?? .review/findings.json\n?? scratch.sh\n"
    assert gate.check_clean_tree(porcelain, ".review/findings.json") == ["scratch.sh"]


# --- build_evidence -------------------------------------------------------


def test_build_evidence_stamps_sha_and_base():
    got = gate.build_evidence(
        {"blocking": True, "findings": [{"class": "regression", "file": "a.py"}]},
        sha=SHA,
        base_ref="main",
    )
    assert got == {
        "schema": 1,
        "status": "reviewed",
        "sha": SHA,
        "base_ref": "main",
        "blocking": True,
        "findings": [{"class": "regression", "file": "a.py"}],
    }


def test_skipped_evidence_shape():
    assert gate.build_skipped_evidence(sha=SHA, base_ref="main", reason="cosmetic") == {
        "schema": 1,
        "status": "skipped",
        "sha": SHA,
        "base_ref": "main",
        "reason": "cosmetic",
    }
