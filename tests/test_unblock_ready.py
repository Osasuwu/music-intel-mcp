"""Label decisions of .github/scripts/unblock_ready.py (issues: closed / unlabeled)."""

import importlib.util
from pathlib import Path

_root = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "scripts").is_dir())
_spec = importlib.util.spec_from_file_location(
    "unblock_ready", _root / ".github" / "scripts" / "unblock_ready.py"
)
unblock_ready = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unblock_ready)


def _issue(labels=(), open_blockers=0, state="open"):
    return {
        "state": state,
        "labels": [{"name": n} for n in labels],
        "issue_dependencies_summary": {
            "blocked_by": open_blockers,
            "total_blocked_by": open_blockers + 1,
        },
    }


def test_last_blocker_closed_promotes_to_ready():
    assert unblock_ready.plan(_issue(["task"])) == (["status:ready"], [])


def test_remaining_open_blocker_keeps_issue_unready():
    assert unblock_ready.plan(_issue(["task"], open_blockers=1)) == ([], [])


def test_closed_issue_is_left_alone():
    assert unblock_ready.plan(_issue(["task"], state="closed")) == ([], [])


def test_issue_past_ready_is_not_overwritten():
    assert unblock_ready.plan(_issue(["task", "status:in-progress"])) == ([], [])


def test_blocked_label_is_swapped_for_ready():
    assert unblock_ready.plan(_issue(["status:blocked"])) == (["status:ready"], ["status:blocked"])


def test_owner_queue_issue_still_gets_ready():
    assert unblock_ready.plan(_issue(["status:owner-queue"])) == (["status:ready"], [])


def test_already_ready_is_a_no_op():
    assert unblock_ready.plan(_issue(["status:ready"])) == ([], [])


def test_missing_summary_is_treated_as_blocked():
    assert unblock_ready.plan({"state": "open", "labels": []}) == ([], [])


def test_dependent_in_same_repo_is_processed():
    dep = {"repository_url": "https://api.github.com/repos/Owner/Repo"}
    assert unblock_ready.dependent_repo(dep, "owner/repo") == "owner/repo"


def test_dependent_in_other_repo_is_skipped():
    dep = {"repository_url": "https://api.github.com/repos/other/repo"}
    assert unblock_ready.dependent_repo(dep, "owner/repo") is None


def test_dependent_without_repository_url_is_skipped():
    assert unblock_ready.dependent_repo({"number": 7}, "owner/repo") is None


def test_open_needs_label_keeps_issue_unready():
    assert unblock_ready.plan(_issue(["needs-safety-review"])) == ([], [])


def test_needs_label_does_not_drop_blocked():
    assert unblock_ready.plan(_issue(["status:blocked", "needs-triage"])) == ([], [])


def test_issue_with_past_blocker_is_reevaluated():
    assert unblock_ready.was_blocked(_issue()) is True


def test_never_blocked_issue_is_not_promoted_on_unlabel():
    issue = {"issue_dependencies_summary": {"blocked_by": 0, "total_blocked_by": 0}}
    assert unblock_ready.was_blocked(issue) is False
    assert unblock_ready.was_blocked({}) is False
