"""Decision rules of .github/scripts/code_gate_verdict.py — the "Review evidence" gate.

Issue #226, ported from Osasuwu/jarvis#1964.

Every function under test is pure: plain dicts in, a value out. The fixtures
are shaped like real GitHub API payloads (workflow runs, pull requests, PR
files) and every expected value is a literal worked out from the locked design,
not read back from the module.
"""

import base64
import http.client
import importlib.util
import json
import urllib.error
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
        "LICENSE.md",
        "LICENSE.txt",
        "LICENSE-APACHE",
        "LICENSE-MIT",
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
        "CLAUDE.md",  # behaviour-carrying markdown is reviewed by path
        "SOUL.md",
        # docs/domain/ is cosmetic only for markdown and raster images
        "docs/domain/x.py",
        "docs/domain/x.svg",
        # Non-root README: code
        "sub/README.md",
        # Config files: code
        "notes.txt",
        "logo.svg",
        # Agent/gate behavior: code
        ".claude/hooks/secret-scanner.py",
        ".claude/marketplace/.claude-plugin/marketplace.json",
        ".claude/settings.json",
        ".claude/skills/x/SKILL.md",
        ".claude/agents/planner.md",
        # Gate machinery: code
        ".github/workflows/pytest.yml",
        ".github/scripts/code_gate_verdict.py",
        # Product code: code
        "src/music_intel_mcp/app.py",
        "tests/test_inference.py",
        "native/wasapi_loopback_helper/helper.cpp",
        "schemas/findings.json",
        "supabase/migrations/20260609000000_shared_metadata.sql",  # schema is code
        "scripts/install.ps1",  # PowerShell is code
        # Config files: code
        ".env.example",
        "pyproject.toml",
        "LICENSE.py",  # LICENSE is matched by exact name, not as a prefix
        "LICENSE-check.sh",
        "docs/domain/../AGENTS.md",  # a `..` component never reads as cosmetic
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


# --- is_gate_machinery ----------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/code-review.yml",
        ".github/workflows/code-gate-verdict.yml",
        ".github/scripts/code_gate_verdict.py",
        ".github/actions/anything/action.yml",
        # music-intel-mcp: agent/gate behavior files are judged by a human too
        ".claude/hooks/secret-scanner.py",
        ".claude/marketplace/.claude-plugin/marketplace.json",
        ".claude/settings.json",
        # Any hook runs in the reviewer's harness, so the whole directory is machinery
        ".claude/hooks/other.py",
    ],
)
def test_gate_machinery_paths(path):
    assert gate.is_gate_machinery([path]) is True


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/pytest.yml",
        ".github/scripts/unblock_ready.py",
        "tests/test_code_gate_verdict.py",
        "docs/reference/github-repo-setup.md",
        ".claude/skills/x/SKILL.md",
    ],
)
def test_non_machinery_paths(path):
    assert gate.is_gate_machinery([path]) is False


def test_machinery_on_the_old_side_of_a_rename_counts():
    files = [
        {
            "filename": "docs/x.md",
            "previous_filename": ".github/scripts/code_gate_verdict.py",
            "status": "renamed",
        }
    ]
    assert gate.is_gate_machinery(gate.changed_paths(files)) is True


# --- is_untrusted_author / is_draft ---------------------------------------


def _pr(**over):
    pr = {
        "number": PR,
        "draft": False,
        "changed_files": 1,
        "user": {"login": "Osasuwu", "type": "User"},
        "base": {"ref": "main", "repo": {"full_name": "Osasuwu/music-intel-mcp"}},
        "head": {"sha": SHA, "repo": {"full_name": "Osasuwu/music-intel-mcp"}},
    }
    pr.update(over)
    return pr


def test_same_repo_human_is_trusted():
    assert gate.is_untrusted_author(_pr()) is False


def test_fork_head_is_untrusted():
    pr = _pr(head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    assert gate.is_untrusted_author(pr) is True


def test_deleted_fork_head_is_untrusted():
    assert gate.is_untrusted_author(_pr(head={"sha": SHA, "repo": None})) is True


def test_dependabot_is_untrusted_even_from_the_same_repo():
    pr = _pr(user={"login": "dependabot[bot]", "type": "Bot"})
    assert gate.is_untrusted_author(pr) is True


def test_draft_flag():
    assert gate.is_draft(_pr(draft=True)) is True
    assert gate.is_draft(_pr(draft=False)) is False


# --- validate_findings ----------------------------------------------------


def test_clean_findings_are_valid():
    assert gate.validate_findings({"blocking": False, "findings": []}) == []


def test_blocking_findings_are_valid():
    obj = {"blocking": True, "findings": [{"class": "regression", "file": "a.py"}]}
    assert gate.validate_findings(obj) == []


@pytest.mark.parametrize(
    "obj",
    [
        {"blocking": True, "findings": []},  # blocking with nothing to block on
        {"blocking": False, "findings": [{"class": "regression", "file": "a.py"}]},
        {"blocking": True, "findings": [{"class": "nitpick", "file": "a.py"}]},  # unknown class
        {"blocking": True, "findings": [{"class": "regression"}]},  # no file
        {"blocking": "no", "findings": []},  # not a bool
        {"findings": []},
        {"blocking": False},
        {"status": "unreviewed"},  # the seed the review job writes before the reviewer runs
        [],
        None,
    ],
)
def test_invalid_findings_are_rejected(obj):
    assert gate.validate_findings(obj) != []


def test_all_eight_classes_are_accepted():
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
    porcelain = "?? code-review-findings.json\n"
    assert gate.check_clean_tree(porcelain, "code-review-findings.json") == []


def test_modified_tracked_file_is_dirty():
    porcelain = " M .github/workflows/pytest.yml\n?? code-review-findings.json\n"
    assert gate.check_clean_tree(porcelain, "code-review-findings.json") == [
        ".github/workflows/pytest.yml"
    ]


def test_stray_untracked_file_is_dirty():
    porcelain = "?? code-review-findings.json\n?? scratch.sh\n"
    assert gate.check_clean_tree(porcelain, "code-review-findings.json") == ["scratch.sh"]


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


# --- run_provenance (real run-API shape) ---------------------------------


def _pr_run(**over):
    run = {
        "id": 1001,
        "name": "Code review",
        "path": ".github/workflows/code-review.yml",
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_branch": "feat/x",
        "head_sha": SHA,
        "display_title": f"Code review PR #{PR} @ {SHA}",
        "pull_requests": [
            {
                "number": PR,
                "base": {"ref": "main", "sha": OTHER_SHA},
                "head": {"ref": "feat/x", "sha": SHA},
            }
        ],
    }
    run.update(over)
    return run


def _dispatch_run(**over):
    """A dispatch run as `gather_entries` hands it on: the history fact is set."""
    run = {
        "id": 2002,
        "name": "Code review",
        "path": ".github/workflows/code-review.yml",
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "success",
        "head_branch": "main",
        "head_sha": OTHER_SHA,  # the default branch tip, not the PR head
        "display_title": f"Code review PR #{PR} @ {SHA}",
        "pull_requests": [],
        "_in_default_history": True,
    }
    run.update(over)
    return run


def _q(run):
    return gate.run_provenance(run, pr_number=PR, head_sha=SHA, default_branch="main")


def _listed(number, base):
    return {"number": number, "base": {"ref": base}, "head": {"sha": SHA}}


def test_pull_request_run_bound_to_pr_qualifies():
    assert _q(_pr_run()) == "bound"


def test_pull_request_run_for_another_pr_is_excluded():
    assert _q(_pr_run(pull_requests=[_listed(8, "main")])) == ""


def test_pull_request_run_with_no_pull_requests_is_excluded():
    """A fork's pull_request run carries an empty pull_requests[]."""
    assert _q(_pr_run(pull_requests=[])) == ""


def test_pull_request_run_into_a_non_default_base_is_foreign():
    assert _q(_pr_run(pull_requests=[_listed(PR, "release")])) == "foreign-base"


@pytest.mark.parametrize("order", ["foreign-first", "foreign-last"])
def test_pull_request_run_also_listing_a_pr_into_another_base_is_foreign(order):
    """like-current-song#241: the run lists every open PR on its head, and may have run the other
    base's copy of the workflow — whichever order the list comes in."""
    listed = [_listed(PR, "main"), _listed(8, "release")]
    if order == "foreign-first":
        listed.reverse()
    assert _q(_pr_run(pull_requests=listed)) == "foreign-base"


def test_pull_request_run_also_listing_another_pr_into_the_default_base_is_bound():
    run = _pr_run(pull_requests=[_listed(8, "main"), _listed(PR, "main")])
    assert _q(run) == "bound"


def test_foreign_pull_request_run_for_an_older_head_sha_is_excluded():
    run = _pr_run(head_sha=OTHER_SHA, pull_requests=[_listed(PR, "release")])
    assert _q(run) == ""


def test_pull_request_run_for_an_older_head_sha_is_excluded():
    assert _q(_pr_run(head_sha=OTHER_SHA)) == ""


def test_dispatch_run_from_default_branch_history_qualifies():
    assert _q(_dispatch_run()) == "bound"


@pytest.mark.parametrize("fact", [False, None, "true", 1], ids=["false", "absent", "str", "int"])
def test_dispatch_run_off_default_branch_history_is_excluded(fact):
    """like-current-song#241: a tag or branch named like the default one does not put a commit in
    its history; only an exact True set by gather_entries does."""
    run = _dispatch_run()
    if fact is None:
        del run["_in_default_history"]
    else:
        run["_in_default_history"] = fact
    assert _q(run) == ""


def test_dispatch_run_bound_by_history_not_by_head_branch():
    """head_branch is spoofable (a tag named `main`), so it decides nothing."""
    assert _q(_dispatch_run(head_branch="feat/x")) == "bound"
    assert _q(_dispatch_run(head_branch="main", _in_default_history=False)) == ""


def test_dispatch_run_titled_for_another_sha_is_excluded():
    assert _q(_dispatch_run(display_title=f"Code review PR #{PR} @ {OTHER_SHA}")) == ""


def test_dispatch_run_titled_for_another_pr_is_excluded():
    assert _q(_dispatch_run(display_title=f"Code review PR #8 @ {SHA}")) == ""


def test_dispatch_run_with_unparseable_title_is_excluded():
    assert _q(_dispatch_run(display_title="Code review")) == ""


def test_run_of_another_workflow_is_excluded():
    assert _q(_pr_run(path=".github/workflows/pytest.yml")) == ""


def test_run_path_with_ref_suffix_is_still_the_review_workflow():
    assert _q(_pr_run(path=".github/workflows/code-review.yml@refs/heads/main")) == "bound"


def test_other_event_types_are_excluded():
    assert _q(_pr_run(event="push")) == ""


# --- evaluate_evidence ----------------------------------------------------


def _artifact(**over):
    art = {
        "schema": 1,
        "status": "reviewed",
        "sha": SHA,
        "base_ref": "main",
        "blocking": False,
        "findings": [],
    }
    art.update(over)
    return art


def _entry(run, artifact="default", state="ok"):
    return {
        "run": run,
        "artifact": _artifact() if artifact == "default" else artifact,
        "state": state,
    }


def _ev(entries):
    return gate.evaluate_evidence(
        entries, pr_number=PR, head_sha=SHA, base_ref="main", default_branch="main"
    )


def test_clean_run_is_green():
    got = _ev([_entry(_pr_run())])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_blocking_run_is_red():
    art = _artifact(blocking=True, findings=[{"class": "regression", "file": "a.py"}])
    got = _ev([_entry(_pr_run(), art)])
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_no_runs_is_red():
    got = _ev([])
    assert (got.green, got.code) == (False, "evidence-none")


def test_unbound_runs_do_not_count():
    got = _ev([_entry(_pr_run(head_sha=OTHER_SHA))])
    assert (got.green, got.code) == (False, "evidence-none")


def test_skip_only_evidence_is_red():
    skipped = {
        "schema": 1,
        "status": "skipped",
        "sha": SHA,
        "base_ref": "main",
        "reason": "cosmetic",
    }
    got = _ev([_entry(_pr_run(), skipped)])
    assert (got.green, got.code) == (False, "evidence-none")


def test_skipped_artifact_is_ignored_next_to_a_clean_one():
    skipped = {
        "schema": 1,
        "status": "skipped",
        "sha": SHA,
        "base_ref": "main",
        "reason": "cosmetic",
    }
    got = _ev([_entry(_pr_run(id=1), skipped), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_success_run_missing_its_artifact_is_red():
    got = _ev([_entry(_pr_run(), None, state="missing")])
    assert (got.green, got.code) == (False, "evidence-missing")


def test_expired_artifact_is_red_with_its_own_code():
    got = _ev([_entry(_pr_run(), None, state="expired")])
    assert (got.green, got.code) == (False, "evidence-expired")


def test_sticky_worst_blocking_beats_a_later_clean_run():
    blocking = _artifact(blocking=True, findings=[{"class": "concurrency", "file": "a.py"}])
    got = _ev([_entry(_pr_run(id=1), blocking), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-blocking")


# A non-blocking red is a failure to produce evidence, not a verdict on the code:
# the remediation its message names ("re-dispatch the review") must be able to
# turn the check green, so a clean run for the SHA supersedes it.
_SUPERSEDABLE_REDS = {
    "evidence-missing": lambda run: _entry(run, None, state="missing"),
    "evidence-expired": lambda run: _entry(run, None, state="expired"),
    "evidence-invalid": lambda run: _entry(run, _artifact(blocking=True, findings=[])),
    "evidence-sha-mismatch": lambda run: _entry(run, _artifact(sha=OTHER_SHA)),
    "evidence-base-mismatch": lambda run: _entry(run, _artifact(base_ref="release")),
    # like-current-song#241: clean evidence from a run that also lists a PR into another base.
    "evidence-foreign-base": lambda run: _entry(
        {**run, "pull_requests": [_listed(PR, "main"), _listed(8, "release")]}
    ),
}


@pytest.mark.parametrize("code", sorted(_SUPERSEDABLE_REDS))
def test_clean_redispatch_supersedes_a_non_blocking_red(code):
    red = _SUPERSEDABLE_REDS[code](_pr_run(id=1))
    assert _ev([red]).code == code  # the red alone is what the fixture claims
    got = _ev([red, _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_blocking_stays_sticky_next_to_supersedable_reds_and_a_clean_run():
    blocking = _artifact(blocking=True, findings=[{"class": "regression", "file": "a.py"}])
    got = _ev(
        [
            _entry(_pr_run(id=1), blocking),
            _entry(_pr_run(id=2), None, state="missing"),
            _entry(_dispatch_run(id=3)),
        ]
    )
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_blocking_evidence_from_a_foreign_base_run_still_sticks():
    """A foreign run cannot clear the PR, but its blocking verdict is only stricter."""
    foreign = _pr_run(id=1, pull_requests=[_listed(PR, "main"), _listed(8, "release")])
    got = _ev([_entry(foreign, _artifact(**_BLOCKING)), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_foreign_base_verdict_says_why():
    """The summary the PR author reads names the cause (like-current-song#241)."""
    foreign = _pr_run(pull_requests=[_listed(PR, "main"), _listed(8, "release")])
    got = _ev([_entry(foreign)])
    assert got.message == (
        "A review run for this commit also lists an open PR into another base branch, "
        "so it may have run that branch's copy of the review workflow; it is evidence "
        "for no PR. Close or retarget that PR, or have a maintainer re-dispatch the review."
    )


def test_foreign_base_outranks_other_supersedable_reds():
    foreign = _pr_run(id=1, pull_requests=[_listed(PR, "main"), _listed(8, "release")])
    got = _ev([_entry(foreign), _entry(_pr_run(id=2), _artifact(sha=OTHER_SHA))])
    assert (got.green, got.code) == (False, "evidence-foreign-base")


def test_worst_supersedable_red_reported_when_no_clean_run():
    got = _ev(
        [
            _entry(_pr_run(id=1), None, state="missing"),
            _entry(_pr_run(id=2), _artifact(sha=OTHER_SHA)),
        ]
    )
    assert (got.green, got.code) == (False, "evidence-sha-mismatch")


def test_artifact_for_another_sha_is_red():
    got = _ev([_entry(_pr_run(), _artifact(sha=OTHER_SHA))])
    assert (got.green, got.code) == (False, "evidence-sha-mismatch")


def test_artifact_for_another_base_is_red():
    got = _ev([_entry(_pr_run(), _artifact(base_ref="release"))])
    assert (got.green, got.code) == (False, "evidence-base-mismatch")


def test_malformed_artifact_is_red():
    got = _ev([_entry(_pr_run(), _artifact(blocking=True, findings=[]))])
    assert (got.green, got.code) == (False, "evidence-invalid")


def test_failed_run_is_ignored_when_a_success_run_has_clean_evidence():
    failed = _pr_run(id=1, conclusion="failure")
    got = _ev([_entry(failed, None, state="missing"), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_failed_run_alone_is_not_evidence():
    got = _ev([_entry(_pr_run(conclusion="failure"), None, state="missing")])
    assert (got.green, got.code) == (False, "evidence-none")


def test_cancelled_run_is_ignored():
    got = _ev(
        [
            _entry(_pr_run(id=1, conclusion="cancelled"), None, state="missing"),
            _entry(_dispatch_run(id=2)),
        ]
    )
    assert (got.green, got.code) == (True, "evidence-clean")


@pytest.mark.parametrize("status", ["queued", "in_progress", "waiting", "pending", "requested"])
def test_unfinished_bound_run_is_red_pending(status):
    run = _pr_run(status=status, conclusion=None)
    got = _ev([_entry(run, None, state="missing"), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-pending")


def test_action_required_run_is_red():
    run = _pr_run(status="completed", conclusion="action_required")
    got = _ev([_entry(run, None, state="missing")])
    assert (got.green, got.code) == (False, "evidence-pending")


# Each run attempt uploads its own artifact, so one run can yield several entries;
# a run's `conclusion` is its latest attempt's.

_BLOCKING = {"blocking": True, "findings": [{"class": "regression", "file": "a.py"}]}


def test_blocking_attempt_is_not_lifted_by_a_clean_re_run_of_the_same_run():
    run = _pr_run(id=1, run_attempt=2)
    got = _ev([_entry(run, _artifact(**_BLOCKING)), _entry(run)])
    assert (got.green, got.code) == (False, "evidence-blocking")


@pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
def test_blocking_from_a_run_whose_re_run_did_not_succeed_still_counts(conclusion):
    failed = _pr_run(id=1, run_attempt=2, conclusion=conclusion)
    got = _ev([_entry(failed, _artifact(**_BLOCKING)), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_clean_evidence_from_a_run_that_did_not_succeed_is_not_evidence():
    got = _ev([_entry(_pr_run(id=1, run_attempt=2, conclusion="failure"))])
    assert (got.green, got.code) == (False, "evidence-none")


@pytest.mark.parametrize(
    "art",
    [
        _artifact(sha=OTHER_SHA),
        _artifact(blocking=True, findings=[]),
        {"schema": 1, "status": "skipped", "sha": SHA, "base_ref": "main", "reason": "draft"},
    ],
    ids=["sha-mismatch", "invalid", "skipped"],
)
def test_non_blocking_evidence_from_a_failed_run_is_ignored(art):
    failed = _pr_run(id=1, conclusion="failure")
    got = _ev([_entry(failed, art), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_unreadable_artifact_alone_is_red():
    got = _ev([_entry(_pr_run(), None, state="unreadable")])
    assert (got.green, got.code) == (False, "evidence-unreadable")


def test_unreadable_artifact_is_not_lifted_by_a_clean_run():
    """It might be the blocking one; a clean run cannot prove otherwise."""
    got = _ev([_entry(_pr_run(id=1), None, state="unreadable"), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-unreadable")


def test_unreadable_artifact_of_a_failed_run_still_counts():
    failed = _pr_run(id=1, conclusion="failure")
    got = _ev([_entry(failed, None, state="unreadable"), _entry(_dispatch_run(id=2))])
    assert (got.green, got.code) == (False, "evidence-unreadable")


def test_blocking_outranks_unreadable():
    got = _ev(
        [
            _entry(_pr_run(id=1), None, state="unreadable"),
            _entry(_pr_run(id=2), _artifact(**_BLOCKING)),
        ]
    )
    assert (got.green, got.code) == (False, "evidence-blocking")


# --- evaluate_pr: the whole rule ------------------------------------------


def _files(*names):
    return [{"filename": n, "status": "modified"} for n in names]


def _eval(pr, files, entries):
    return gate.evaluate_pr(pr, files, entries, default_branch="main")


def test_draft_is_red_with_the_literal_reason_draft_even_with_clean_evidence():
    got = _eval(_pr(draft=True, changed_files=1), _files("src/a.py"), [_entry(_pr_run())])
    assert (got.green, got.code) == (False, "draft")


def test_draft_beats_an_all_cosmetic_change_set():
    got = _eval(_pr(draft=True, changed_files=1), _files("docs/domain/a.md"), [])
    assert (got.green, got.code) == (False, "draft")


def test_all_cosmetic_change_set_is_green_without_evidence():
    got = _eval(_pr(changed_files=2), _files("docs/domain/a.md", "docs/domain/b.png"), [])
    assert (got.green, got.code) == (True, "cosmetic")


def test_one_code_file_among_cosmetic_ones_needs_evidence():
    got = _eval(_pr(changed_files=2), _files("docs/domain/a.md", "src/a.py"), [])
    assert (got.green, got.code) == (False, "evidence-none")


def test_code_change_with_clean_evidence_is_green():
    got = _eval(_pr(changed_files=1), _files("src/a.py"), [_entry(_pr_run())])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_code_change_with_blocking_evidence_is_red():
    blocking = _artifact(blocking=True, findings=[{"class": "regression", "file": "src/a.py"}])
    got = _eval(_pr(changed_files=1), _files("src/a.py"), [_entry(_pr_run(), blocking)])
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_changed_files_count_mismatch_is_red():
    got = _eval(_pr(changed_files=3001), _files("docs/domain/a.md"), [])
    assert (got.green, got.code) == (False, "files-incomplete")


def test_changed_files_count_mismatch_is_red_even_with_clean_evidence():
    got = _eval(_pr(changed_files=2), _files("src/a.py"), [_entry(_pr_run())])
    assert (got.green, got.code) == (False, "files-incomplete")


def test_empty_change_set_is_red():
    got = _eval(_pr(changed_files=0), [], [])
    assert (got.green, got.code) == (False, "files-empty")


def test_gate_machinery_with_clean_evidence_is_red():
    got = _eval(
        _pr(changed_files=1),
        _files(".github/workflows/code-review.yml"),
        [_entry(_pr_run())],
    )
    assert (got.green, got.code) == (False, "gate-machinery")


def test_dependabot_pr_without_evidence_is_red_with_dispatch_remediation():
    pr = _pr(changed_files=1, user={"login": "dependabot[bot]", "type": "Bot"})
    got = _eval(pr, _files("requirements.txt"), [])
    assert (got.green, got.code) == (False, "untrusted-needs-dispatch")


def test_fork_pr_without_evidence_is_red_with_dispatch_remediation():
    pr = _pr(changed_files=1, head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    got = _eval(pr, _files("src/a.py"), [])
    assert (got.green, got.code) == (False, "untrusted-needs-dispatch")


def test_fork_pr_with_a_dispatched_clean_review_is_green():
    pr = _pr(changed_files=1, head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    got = _eval(pr, _files("src/a.py"), [_entry(_dispatch_run())])
    assert (got.green, got.code) == (True, "evidence-clean")


def test_fork_pr_with_a_blocking_dispatched_review_stays_blocking_not_remediation():
    pr = _pr(changed_files=1, head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    blocking = _artifact(blocking=True, findings=[{"class": "regression", "file": "a.py"}])
    got = _eval(pr, _files("src/a.py"), [_entry(_dispatch_run(), blocking)])
    assert (got.green, got.code) == (False, "evidence-blocking")


def test_fork_pr_all_cosmetic_is_green():
    pr = _pr(changed_files=1, head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    got = _eval(pr, _files("docs/domain/a.md"), [])
    assert (got.green, got.code) == (True, "cosmetic")


# --- review_skip_reason ---------------------------------------------------


def _skip(pr, files, event="pull_request"):
    return gate.review_skip_reason(pr, files, event)


def test_skip_reason_draft():
    assert _skip(_pr(draft=True), _files("src/a.py")) == "draft"


def test_skip_reason_cosmetic():
    assert _skip(_pr(changed_files=1), _files("docs/domain/a.md")) == "cosmetic"


def test_skip_reason_files_incomplete():
    assert _skip(_pr(changed_files=3001), _files("src/a.py")) == "files-incomplete"


def test_skip_reason_untrusted_on_pull_request_event():
    pr = _pr(changed_files=1, user={"login": "dependabot[bot]", "type": "Bot"})
    assert _skip(pr, _files("requirements.txt")) == "untrusted"


def test_untrusted_pr_is_reviewed_when_a_maintainer_dispatches():
    pr = _pr(changed_files=1, head={"sha": SHA, "repo": {"full_name": "stranger/music-intel-mcp"}})
    assert _skip(pr, _files("src/a.py"), event="workflow_dispatch") == ""


def test_trusted_code_change_is_reviewed():
    assert _skip(_pr(changed_files=1), _files("src/a.py")) == ""


# --- resolve_run_target ---------------------------------------------------


def _resolve(event, title, head_sha, pull_requests):
    return gate.resolve_run_target(event, title, head_sha, pull_requests, "main")


def test_resolve_dispatch_target_from_the_run_title():
    got = _resolve("workflow_dispatch", f"Code review PR #{PR} @ {SHA}", OTHER_SHA, [])
    assert got == (7, SHA)


def test_resolve_pull_request_target_from_pull_requests():
    got = _resolve("pull_request", "x", SHA, [_listed(7, "main")])
    assert got == (7, SHA)


@pytest.mark.parametrize("order", ["default-first", "default-last"])
def test_resolve_picks_the_pr_into_the_default_branch(order):
    """like-current-song#230: the list holds every open PR on the head, in no promised order."""
    listed = [_listed(7, "main"), _listed(8, "release")]
    if order == "default-last":
        listed.reverse()
    assert _resolve("pull_request", "x", SHA, listed) == (7, SHA)


def test_resolve_run_listing_only_prs_into_other_bases_is_unbound():
    assert _resolve("pull_request", "x", SHA, [_listed(8, "release")]) is None


def test_resolve_command_binds_the_pr_into_the_repos_default_branch(monkeypatch, tmp_path):
    """The workflow's DEFAULT_BRANCH decides, not a hard-coded `main`."""
    out = tmp_path / "out"
    env = {
        "EVENT_NAME": "workflow_run",
        "RUN_EVENT": "pull_request",
        "RUN_HEAD_SHA": SHA,
        "RUN_PULL_REQUESTS": json.dumps([_listed(8, "main"), _listed(7, "trunk")]),
        "DEFAULT_BRANCH": "trunk",
        "GITHUB_OUTPUT": str(out),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert gate.cmd_resolve(None) == 0
    assert out.read_text(encoding="utf-8") == f"pr_number=7\nhead_sha={SHA}\n"


def test_resolve_fork_pull_request_run_is_unbound():
    assert _resolve("pull_request", "x", SHA, []) is None


def test_resolve_unparseable_dispatch_title_is_unbound():
    assert _resolve("workflow_dispatch", "Code review", SHA, []) is None


def test_resolve_push_run_is_unbound_even_with_pull_requests():
    """Only `pull_request` and `workflow_dispatch` runs carry review evidence."""
    assert _resolve("push", "x", SHA, [_listed(7, "main")]) is None


# --- read_evidence_zip ----------------------------------------------------


def _zip(name, payload):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, payload)
    return buf.getvalue()


def test_zip_with_evidence_is_parsed():
    blob = _zip("review-evidence.json", '{"status": "skipped"}')
    assert gate.read_evidence_zip(blob) == {"status": "skipped"}


def test_zip_without_the_evidence_file_is_malformed():
    assert gate.read_evidence_zip(_zip("other.json", "{}")) == {"status": "malformed"}


def _corrupt(blob):
    """Flip one byte of the stored payload: the zip still opens, its CRC no longer matches."""
    i = blob.index(b'"status"')
    return blob[:i] + b"X" + blob[i + 1 :]


@pytest.mark.parametrize(
    "blob",
    [
        b"not a zip",
        _zip("review-evidence.json", '{"status": "skipped"}')[:-30],
        _corrupt(_zip("review-evidence.json", '{"status": "skipped"}')),
    ],
    ids=["not-a-zip", "cut-off", "crc-mismatch"],
)
def test_unreadable_zip_raises_instead_of_reading_as_malformed(blob):
    """upload-artifact never writes a bad zip, so one means a broken download: its
    verdict is unknown, which is not the same as malformed (a clean run lifts that)."""
    with pytest.raises(gate.UnreadableArtifact):
        gate.read_evidence_zip(blob)


def test_zip_with_invalid_json_is_malformed():
    assert gate.read_evidence_zip(_zip("review-evidence.json", "{nope")) == {"status": "malformed"}


# --- load_entries (fake API) ----------------------------------------------


class FakeApi:
    """GETs answered by path prefix; downloads by artifact id from `blobs`, where
    an exception instance is raised instead of returned."""

    def __init__(self, responses=None, blobs=None):
        self.responses = responses or {}
        self.blobs = blobs or {}
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method != "GET":
            return {}
        for prefix, value in self.responses.items():
            if path.startswith(prefix):
                if isinstance(value, BaseException):
                    raise value
                return value
        raise AssertionError(f"unexpected API call {method} {path}")

    def download(self, path):
        self.calls.append(("DOWNLOAD", path, None))
        blob = self.blobs[int(path.split("/")[-2])]
        if isinstance(blob, BaseException):
            raise blob
        return blob


ARTIFACTS_PATH = "repos/o/r/actions/runs/1001/artifacts"


def _evidence_zip(sha=SHA, blocking=False):
    findings = [{"class": "regression", "file": "a.py"}] if blocking else []
    return _zip(
        "review-evidence.json",
        json.dumps(
            {
                "schema": 1,
                "status": "reviewed",
                "sha": sha,
                "base_ref": "main",
                "blocking": blocking,
                "findings": findings,
            }
        ),
    )


def _art(id, name, expired=False):
    return {"id": id, "name": name, "expired": expired}


def _downloads(api):
    return sorted(int(path.split("/")[-2]) for m, path, _ in api.calls if m == "DOWNLOAD")


def test_load_entries_reads_every_live_attempt():
    """Expired attempt 3 is skipped: its verdict is gone. That is the known
    retention bound on blocking stickiness (Osasuwu/like-current-song#236), not
    the intended end state."""
    api = FakeApi(
        {
            ARTIFACTS_PATH: {
                "artifacts": [
                    _art(1, "review-evidence-1"),
                    _art(2, "review-evidence-2"),
                    _art(3, "review-evidence-3", expired=True),
                ]
            }
        },
        blobs={1: _evidence_zip(blocking=True), 2: _evidence_zip()},
    )
    got = gate.load_entries(api, "o/r", _pr_run())
    assert [(e["state"], e["artifact"]["blocking"]) for e in got] == [("ok", True), ("ok", False)]
    assert _downloads(api) == [1, 2]


@pytest.mark.parametrize(
    "name",
    [
        "claude-execution-output-1",
        "review-evidence-x",
        "review-evidence",  # the pre-port-round-2 single, overwritable artifact
        "review-evidence-0",
        "review-evidence-1-extra",
        "my-review-evidence-1",
    ],
)
def test_load_entries_ignores_artifacts_not_named_review_evidence_attempt(name):
    api = FakeApi(
        {ARTIFACTS_PATH: {"artifacts": [_art(9, name), _art(1, "review-evidence-1")]}},
        blobs={1: _evidence_zip()},
    )
    got = gate.load_entries(api, "o/r", _pr_run())
    assert [e["state"] for e in got] == ["ok"]
    assert _downloads(api) == [1]


def test_load_entries_all_expired_is_one_expired_entry():
    api = FakeApi(
        {
            ARTIFACTS_PATH: {
                "artifacts": [
                    _art(1, "review-evidence-1", expired=True),
                    _art(2, "review-evidence-2", expired=True),
                ]
            }
        }
    )
    got = gate.load_entries(api, "o/r", _pr_run())
    assert [(e["state"], e["artifact"]) for e in got] == [("expired", None)]
    assert _downloads(api) == []


@pytest.mark.parametrize(
    "artifacts", [[], [_art(4, "claude-execution-output-1")]], ids=["none", "only-other"]
)
def test_load_entries_without_evidence_is_one_missing_entry(artifacts):
    api = FakeApi({ARTIFACTS_PATH: {"artifacts": artifacts}})
    got = gate.load_entries(api, "o/r", _pr_run())
    assert [(e["state"], e["artifact"]) for e in got] == [("missing", None)]


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.HTTPError("u", 500, "boom", {}, None),
        urllib.error.URLError("dns"),
        TimeoutError("read timed out"),
        ConnectionResetError("reset"),
        http.client.IncompleteRead(b"PK"),
    ],
    ids=["http", "url", "timeout", "reset", "truncated"],
)
def test_load_entries_failed_download_is_unreadable_not_missing(exc):
    api = FakeApi(
        {
            ARTIFACTS_PATH: {
                "artifacts": [_art(1, "review-evidence-1"), _art(2, "review-evidence-2")]
            }
        },
        blobs={1: exc, 2: _evidence_zip()},
    )
    got = gate.load_entries(api, "o/r", _pr_run())
    assert sorted(e["state"] for e in got) == ["ok", "unreadable"]
    (unreadable,) = [e for e in got if e["state"] == "unreadable"]
    assert unreadable["artifact"] is None


def test_cut_off_blocking_download_is_not_lifted_by_a_clean_attempt():
    """`HTTPResponse.read(amt)` hands back a short body without raising. Attempt 1
    blocking but cut off, attempt 2 clean: the SHA must not go green."""
    api = FakeApi(
        {
            ARTIFACTS_PATH: {
                "artifacts": [_art(1, "review-evidence-1"), _art(2, "review-evidence-2")]
            }
        },
        blobs={1: _evidence_zip(blocking=True)[:-30], 2: _evidence_zip()},
    )
    got = gate.load_entries(api, "o/r", _pr_run())
    assert sorted(e["state"] for e in got) == ["ok", "unreadable"]
    verdict = _ev(got)
    assert (verdict.green, verdict.code) == (False, "evidence-unreadable")


class _Resp:
    def __init__(self, body, content_length=None):
        self._body = body
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def read(self, amt):
        return self._body[:amt]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def __init__(self, outcome):
        self._outcome = outcome

    def open(self, req, timeout=None):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def test_download_rejects_a_cut_off_body_served_directly(monkeypatch):
    monkeypatch.setattr(gate.urllib.request, "build_opener", lambda *h: _Opener(_Resp(b"PK-z", 6)))
    with pytest.raises(gate.UnreadableArtifact):
        gate.Api("tok").download("repos/o/r/actions/artifacts/1/zip")


def test_download_rejects_a_cut_off_body_behind_the_redirect(monkeypatch):
    redirect = urllib.error.HTTPError(
        "https://api.github.com/x", 302, "Found", {"Location": "https://blob.example/z"}, None
    )
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req)
        return _Resp(b"PK-z", 6)

    monkeypatch.setattr(gate.urllib.request, "build_opener", lambda *h: _Opener(redirect))
    monkeypatch.setattr(gate.urllib.request, "urlopen", urlopen)
    with pytest.raises(gate.UnreadableArtifact):
        gate.Api("tok").download("repos/o/r/actions/artifacts/1/zip")
    assert [r.full_url for r in seen] == ["https://blob.example/z"]
    assert seen[0].get_header("Authorization") is None


def test_read_capped_returns_a_whole_body():
    assert gate.read_capped(_Resp(b"PK-zip", content_length=6)) == b"PK-zip"
    assert gate.read_capped(_Resp(b"PK-zip")) == b"PK-zip"


def test_read_capped_raises_on_a_body_shorter_than_its_content_length():
    with pytest.raises(gate.UnreadableArtifact):
        gate.read_capped(_Resp(b"PK-z", content_length=6))


def test_read_capped_raises_on_an_oversized_body_instead_of_truncating_it():
    with pytest.raises(gate.UnreadableArtifact):
        gate.read_capped(_Resp(b"x" * 11), limit=10)
    assert gate.read_capped(_Resp(b"x" * 10), limit=10) == b"x" * 10


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out"])
def test_load_entries_reads_a_run_whose_latest_attempt_did_not_succeed(conclusion):
    """An earlier attempt's blocking artifact outlives a failed or cancelled re-run."""
    api = FakeApi(
        {ARTIFACTS_PATH: {"artifacts": [_art(1, "review-evidence-1")]}},
        blobs={1: _evidence_zip(blocking=True)},
    )
    got = gate.load_entries(api, "o/r", _pr_run(conclusion=conclusion))
    assert [(e["state"], e["artifact"]["blocking"]) for e in got] == [("ok", True)]


@pytest.mark.parametrize(
    "status,conclusion",
    [("in_progress", None), ("queued", None), ("completed", "action_required")],
)
def test_load_entries_makes_no_call_for_an_unfinished_or_unapproved_run(status, conclusion):
    api = FakeApi()
    got = gate.load_entries(api, "o/r", _pr_run(status=status, conclusion=conclusion))
    assert [(e["state"], e["artifact"]) for e in got] == [("missing", None)]
    assert api.calls == []


def test_gather_then_evaluate_keeps_blocking_from_an_earlier_attempt():
    """End to end over the API shape: attempt 1 blocking, the re-run (attempt 2)
    clean and the run's conclusion success — the SHA stays blocked."""
    runs_path = "repos/o/r/actions/workflows/code-review.yml/runs"
    api = FakeApi(
        {
            runs_path: {"workflow_runs": [_pr_run(run_attempt=2)]},
            ARTIFACTS_PATH: {
                "artifacts": [_art(1, "review-evidence-1"), _art(2, "review-evidence-2")]
            },
        },
        blobs={1: _evidence_zip(blocking=True), 2: _evidence_zip()},
    )
    entries = gate.gather_entries(api, "o/r", PR, SHA, "main")
    assert len(entries) == 2
    got = _ev(entries)
    assert (got.green, got.code) == (False, "evidence-blocking")


RUNS_PATH = "repos/o/r/actions/workflows/code-review.yml/runs"
COMPARE_PATH = f"repos/o/r/compare/main...{OTHER_SHA}"
DISPATCH_ARTIFACTS = "repos/o/r/actions/runs/2002/artifacts"


def _payload_dispatch(**over):
    """A dispatch run as the runs API returns it: no history fact."""
    run = _dispatch_run(**over)
    if "_in_default_history" not in over:
        del run["_in_default_history"]
    return run


def _gather(runs, compare):
    """Only the two listings the runs API is really asked are answered: the PR's head
    SHA (a dispatch run's head is the default branch's commit, so it is not there) and
    the default branch's dispatch runs. A wrong query is an unexpected call."""
    api = FakeApi(
        {
            COMPARE_PATH: compare,
            f"{RUNS_PATH}?head_sha={SHA}&": {"workflow_runs": []},
            f"{RUNS_PATH}?event=workflow_dispatch&branch=main&": {"workflow_runs": runs},
            DISPATCH_ARTIFACTS: {"artifacts": [_art(1, "review-evidence-1")]},
        },
        blobs={1: _evidence_zip()},
    )
    return api, gate.gather_entries(api, "o/r", PR, SHA, "main")


def _compares(api):
    return [path for _, path, _ in api.calls if path.startswith("repos/o/r/compare/")]


@pytest.mark.parametrize("status", ["identical", "behind"])
def test_dispatch_run_in_default_history_is_evidence(status):
    """'behind': the default branch moved on past the run's head, and the run still
    counts (like-current-song#241)."""
    api, entries = _gather([_payload_dispatch()], {"status": status})
    assert _compares(api) == [f"{COMPARE_PATH}?per_page=1"]
    got = _ev(entries)
    assert (got.green, got.code) == (True, "evidence-clean")


@pytest.mark.parametrize("status", ["ahead", "diverged"])
def test_dispatch_run_off_default_history_is_not_loaded(status):
    """A tag or branch named `main` puts the run on `branch=main`, not in its history."""
    api, entries = _gather([_payload_dispatch()], {"status": status})
    assert _compares(api) == [f"{COMPARE_PATH}?per_page=1"]
    assert entries == []
    assert _downloads(api) == []


def test_spoofed_history_key_in_the_run_payload_is_ignored():
    _, entries = _gather([_payload_dispatch(_in_default_history=True)], {"status": "diverged"})
    assert entries == []


def test_history_is_checked_only_for_dispatch_runs_titled_for_this_pr_and_sha():
    runs = [
        _payload_dispatch(id=2003, display_title=f"Code review PR #8 @ {SHA}"),
        _payload_dispatch(id=2004, display_title=f"Code review PR #{PR} @ {OTHER_SHA}"),
        _pr_run(),
    ]
    api = FakeApi(
        {
            RUNS_PATH: {"workflow_runs": runs},
            ARTIFACTS_PATH: {"artifacts": [_art(1, "review-evidence-1")]},
        },
        blobs={1: _evidence_zip()},
    )
    entries = gate.gather_entries(api, "o/r", PR, SHA, "main")
    assert _compares(api) == []
    assert [e["run"]["id"] for e in entries] == [1001]


def test_gather_loads_a_foreign_base_run_so_its_blocking_verdict_sticks():
    foreign = _pr_run(pull_requests=[_listed(PR, "main"), _listed(8, "release")])
    api = FakeApi(
        {
            RUNS_PATH: {"workflow_runs": [foreign]},
            ARTIFACTS_PATH: {"artifacts": [_art(1, "review-evidence-1")]},
        },
        blobs={1: _evidence_zip(blocking=True)},
    )
    got = _ev(gate.gather_entries(api, "o/r", PR, SHA, "main"))
    assert (got.green, got.code) == (False, "evidence-blocking")


@pytest.mark.parametrize("code", [404, 422])
def test_unknown_or_unrelated_commit_is_not_in_default_history(code):
    err = urllib.error.HTTPError(COMPARE_PATH, code, "x", {}, None)
    assert gate.in_default_history(FakeApi({COMPARE_PATH: err}), "o/r", "main", OTHER_SHA) is False


def test_compare_server_error_is_raised_not_read_as_off_history():
    err = urllib.error.HTTPError(COMPARE_PATH, 502, "x", {}, None)
    with pytest.raises(urllib.error.HTTPError) as caught:
        gate.in_default_history(FakeApi({COMPARE_PATH: err}), "o/r", "main", OTHER_SHA)
    assert caught.value.code == 502


# --- fetch_pr_snapshot ----------------------------------------------------


def test_snapshot_returns_pr_and_files():
    pr = _pr(changed_files=1)
    api = FakeApi({"repos/o/r/pulls/7/files": [{"filename": "a.py"}], "repos/o/r/pulls/7": pr})
    got_pr, got_files = gate.fetch_pr_snapshot(api, "o/r", 7)
    assert got_pr["head"]["sha"] == SHA
    assert got_files == [{"filename": "a.py"}]


def test_snapshot_fails_when_the_head_moves_mid_read():
    heads = iter([SHA, OTHER_SHA])

    class Moving(FakeApi):
        def __call__(self, method, path, body=None):
            if path == "repos/o/r/pulls/7":
                return _pr(
                    head={"sha": next(heads), "repo": {"full_name": "Osasuwu/music-intel-mcp"}}
                )
            return []

    with pytest.raises(RuntimeError, match="head moved"):
        gate.fetch_pr_snapshot(Moving(), "o/r", 7)


# --- post_check (fake API) ------------------------------------------------


CHECKS_PATH = f"repos/o/r/commits/{SHA}/check-runs"
RED = gate.Verdict(False, "evidence-none", "no review")
GREEN = gate.Verdict(True, "cosmetic", "cosmetic")


def test_post_check_patches_this_apps_existing_check_run():
    runs = [{"id": 9, "app": {"id": 123}}, {"id": 8, "app": {"id": 15368}}]
    api = FakeApi({CHECKS_PATH: {"check_runs": runs}})
    assert gate.post_check(api, "o/r", SHA, RED, 123) == "patched"
    method, path, body = api.calls[-1]
    assert (method, path) == ("PATCH", "repos/o/r/check-runs/9")
    assert body["conclusion"] == "failure"
    assert body["output"]["title"] == "verify-verdict: evidence-none"


def test_post_check_creates_only_when_the_app_has_none():
    api = FakeApi({CHECKS_PATH: {"check_runs": [{"id": 8, "app": {"id": 15368}}]}})
    assert gate.post_check(api, "o/r", SHA, GREEN, 123) == "created"
    method, path, body = api.calls[-1]
    assert (method, path) == ("POST", "repos/o/r/check-runs")
    assert (body["name"], body["head_sha"], body["conclusion"]) == (
        "verify-verdict",
        SHA,
        "success",
    )


# --- cmd_verdict (fake API) -----------------------------------------------


def _run_verdict(monkeypatch, read_responses, blobs=None):
    """cmd_verdict over a read API answering `read_responses` and a gate API whose
    `verify-verdict` for the SHA is currently green (id 9, app 123)."""
    read = FakeApi(read_responses, blobs)
    write = FakeApi({CHECKS_PATH: {"check_runs": [{"id": 9, "app": {"id": 123}}]}})
    apis = {"read-token": read, "gate-token": write}
    monkeypatch.setattr(gate, "Api", lambda token: apis[token])
    env = {
        "GITHUB_REPOSITORY": "o/r",
        "HEAD_SHA": SHA,
        "GH_TOKEN": "read-token",
        "GATE_TOKEN": "gate-token",
        "GATE_APP_ID": "123",
        "PR_NUMBER": str(PR),
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return write


def _patched(write):
    return [(path, body) for m, path, body in write.calls if m == "PATCH"]


OPEN_PULLS = "repos/o/r/pulls?state=open&base=main"


def _snapshot_responses(pr=None, **extra):
    pr = pr or _pr(changed_files=1)
    responses = {**extra}
    responses.setdefault(OPEN_PULLS, [pr])
    responses.update(
        {
            f"repos/o/r/pulls/{PR}/files": [{"filename": "a.py", "status": "modified"}],
            f"repos/o/r/pulls/{PR}": pr,
            "repos/o/r": {"default_branch": "main"},
        }
    )
    return responses


@pytest.mark.parametrize(
    "where,responses",
    [
        (
            "artifacts-listing",
            _snapshot_responses(
                **{
                    "repos/o/r/actions/workflows/code-review.yml/runs": {
                        "workflow_runs": [_pr_run()]
                    },
                    ARTIFACTS_PATH: ConnectionResetError("reset"),
                }
            ),
        ),
        ("pr-snapshot", {f"repos/o/r/pulls/{PR}": urllib.error.URLError("dns")}),
    ],
)
def test_a_crashing_verdict_replaces_a_green_check_with_a_red_one(monkeypatch, where, responses):
    """A crash must not leave the SHA's earlier green standing: it posts red, then fails the job."""
    write = _run_verdict(monkeypatch, responses)
    with pytest.raises(OSError):
        gate.cmd_verdict(None)
    ((path, body),) = _patched(write)
    assert path == "repos/o/r/check-runs/9"
    assert body["conclusion"] == "failure"
    assert body["output"]["title"] == "verify-verdict: verdict-error"


def test_a_completed_verdict_posts_its_result(monkeypatch):
    write = _run_verdict(
        monkeypatch,
        _snapshot_responses(
            **{
                "repos/o/r/actions/workflows/code-review.yml/runs": {"workflow_runs": [_pr_run()]},
                ARTIFACTS_PATH: {"artifacts": [_art(1, "review-evidence-1")]},
            }
        ),
        blobs={1: _evidence_zip(blocking=True)},
    )
    assert gate.cmd_verdict(None) == 0
    ((_, body),) = _patched(write)
    assert body["output"]["title"] == "verify-verdict: evidence-blocking"


def test_a_pr_into_another_base_posts_no_verdict(monkeypatch):
    """like-current-song#230: the check belongs to the commit, so a verdict for a PR into `release`
    would land on a PR into the default branch that shares the head."""
    pr = _pr(changed_files=1, base={"ref": "release", "repo": {"full_name": "o/r"}})
    write = _run_verdict(monkeypatch, _snapshot_responses(pr=pr))
    assert gate.cmd_verdict(None) == 0
    assert write.calls == []


def test_a_head_that_moved_past_the_event_sha_posts_no_verdict(monkeypatch):
    """The newer SHA has its own evaluation; a verdict here would land on a stale commit."""
    pr = _pr(head={"sha": OTHER_SHA, "repo": {"full_name": "o/r"}})
    write = _run_verdict(monkeypatch, _snapshot_responses(pr=pr))
    assert gate.cmd_verdict(None) == 0
    assert write.calls == []


def test_two_open_prs_on_one_head_post_shared_head(monkeypatch):
    responses = _snapshot_responses(
        **{OPEN_PULLS: [_pr(changed_files=1), _pr(number=8, changed_files=1)]}
    )
    write = _run_verdict(monkeypatch, responses)
    assert gate.cmd_verdict(None) == 0
    ((_, body),) = _patched(write)
    assert (body["conclusion"], body["output"]["title"]) == (
        "failure",
        "verify-verdict: shared-head",
    )


def test_an_open_pr_listing_too_long_to_read_whole_fails_closed(monkeypatch):
    page = [_pr(number=100 + i, head={"sha": OTHER_SHA}) for i in range(100)]
    write = _run_verdict(monkeypatch, _snapshot_responses(**{OPEN_PULLS: page}))
    with pytest.raises(RuntimeError, match="too many open PRs"):
        gate.cmd_verdict(None)
    ((_, body),) = _patched(write)
    assert body["output"]["title"] == "verify-verdict: verdict-error"


# --- shared_head_verdict --------------------------------------------------


def _open(number, sha=SHA, base="main"):
    return _pr(number=number, head={"sha": sha}, base={"ref": base})


def test_shared_head_names_every_other_pr_on_the_commit():
    got = gate.shared_head_verdict(_open(PR), [_open(9), _open(PR), _open(8)])
    assert (got.green, got.code) == (False, "shared-head")
    assert got.message.startswith("Open PR(s) #8, #9 into the same base have this same head commit")


@pytest.mark.parametrize(
    "others",
    [[], [_open(8, sha=OTHER_SHA)], [_open(8, base="release")]],
    ids=["alone", "other-head", "other-base"],
)
def test_no_shared_head_without_another_pr_on_the_same_commit_and_base(others):
    assert gate.shared_head_verdict(_open(PR), [_open(PR), *others]) is None


# --- safe_relpath / build_diff --------------------------------------------


@pytest.mark.parametrize(
    "path", ["../x", "/etc/passwd", "a/../b", "a\\b", "C:/x", "a//b", "", "./a"]
)
def test_unsafe_relpaths_are_rejected(path):
    assert gate.safe_relpath(path) is None


def test_ordinary_relpath_is_kept():
    assert gate.safe_relpath("src/pkg/a.py") == "src/pkg/a.py"


def test_build_diff_labels_added_removed_and_renamed_files():
    files = [
        {"filename": "new.py", "status": "added", "patch": "@@ -0,0 +1 @@\n+x"},
        {"filename": "gone.py", "status": "removed", "patch": "@@ -1 +0,0 @@\n-x"},
        {
            "filename": "b.py",
            "previous_filename": "a.py",
            "status": "renamed",
            "patch": "@@ -1 +1 @@\n-x\n+y",
        },
        {"filename": "img.png", "status": "added"},
    ]
    assert gate.build_diff(files) == (
        "diff --git a/new.py b/new.py\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        "diff --git a/a.py b/b.py\n--- a/a.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )


# --- write_head_data ------------------------------------------------------


def test_head_files_are_written_inert_under_a_pr_suffix(tmp_path):
    """A PR-controlled `.claude/settings.json` must not land under a name the
    reviewer's harness would load as config."""
    blob = {"size": 2, "content": base64.b64encode(b"{}").decode()}
    api = FakeApi({"repos/o/r/git/blobs/s1": blob})
    files = [{"filename": ".claude/settings.json", "status": "modified", "sha": "s1"}]
    gate.write_head_data(api, "o/r", files, str(tmp_path))
    assert (tmp_path / ".claude" / "settings.json.pr").read_bytes() == b"{}"
    assert not (tmp_path / ".claude" / "settings.json").exists()
