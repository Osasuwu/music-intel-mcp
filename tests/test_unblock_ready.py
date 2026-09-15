"""Label decisions of .github/scripts/unblock_ready.py (issue and PR lifecycle events)."""

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


def test_needs_label_with_owner_queue_keeps_issue_unready():
    assert unblock_ready.plan(_issue(["status:owner-queue", "needs-triage"])) == ([], [])


def test_issue_with_past_blocker_is_reevaluated():
    assert unblock_ready.was_blocked(_issue()) is True


def test_never_blocked_issue_is_not_promoted_on_unlabel():
    issue = {"issue_dependencies_summary": {"blocked_by": 0, "total_blocked_by": 0}}
    assert unblock_ready.was_blocked(issue) is False
    assert unblock_ready.was_blocked({}) is False


def test_close_strips_status_labels_but_keeps_hardware():
    issue = _issue(
        ["task", "status:in-progress", "status:owner-queue", "status:hardware-done"],
        state="closed",
    )
    assert unblock_ready.plan_close(issue) == ([], ["status:in-progress", "status:owner-queue"])


def test_close_cleanup_skips_issue_reopened_in_the_meantime():
    assert unblock_ready.plan_close(_issue(["status:in-progress"])) == ([], [])


def test_reopen_drops_in_flight_and_restores_ready():
    issue = _issue(["task", "status:in-progress", "status:review"])
    expected = (["status:ready"], ["status:in-progress", "status:review"])
    assert unblock_ready.plan_reopen(issue) == expected


def test_reopen_with_open_blocker_only_drops_in_flight():
    issue = _issue(["status:in-progress"], open_blockers=1)
    assert unblock_ready.plan_reopen(issue) == ([], ["status:in-progress"])


def test_reopen_with_needs_label_is_not_ready():
    assert unblock_ready.plan_reopen(_issue(["needs-grill"])) == ([], [])


def test_reopen_hardware_issue_is_left_to_hardware_lifecycle():
    assert unblock_ready.plan_reopen(_issue(["status:hardware-done"])) == ([], [])


def test_reopen_keeps_existing_ready():
    assert unblock_ready.plan_reopen(_issue(["status:ready"])) == ([], [])


def test_pr_up_moves_ready_issue_to_review():
    expected = (["status:review"], ["status:ready"])
    assert unblock_ready.plan_review(_issue(["status:ready"])) == expected


def test_pr_up_replaces_in_progress_with_review():
    issue = _issue(["status:in-progress", "status:owner-queue"])
    assert unblock_ready.plan_review(issue) == (["status:review"], ["status:in-progress"])


def test_pr_up_on_issue_already_in_review_is_a_no_op():
    assert unblock_ready.plan_review(_issue(["status:review"])) == ([], [])


def test_pr_up_leaves_closed_issue_alone():
    assert unblock_ready.plan_review(_issue(["status:ready"], state="closed")) == ([], [])


def test_pr_up_leaves_hardware_issue_to_hardware_lifecycle():
    assert unblock_ready.plan_review(_issue(["status:hardware-testing"])) == ([], [])


def _linked(number, repo="owner/repo", open_prs=0):
    return {
        "number": number,
        "repository": {"nameWithOwner": repo},
        "closedByPullRequestsReferences": {"nodes": [{"number": 9}] * open_prs},
    }


def test_pr_issues_skip_other_repos():
    nodes = [_linked(1, "Owner/Repo"), _linked(2, "other/repo")]
    assert unblock_ready.pr_issue_numbers(nodes, "owner/repo", dropped=False) == [1]


def test_dropped_pr_skips_issue_another_open_pr_still_closes():
    nodes = [_linked(1), _linked(2, open_prs=1)]
    assert unblock_ready.pr_issue_numbers(nodes, "owner/repo", dropped=True) == [1]
    assert unblock_ready.pr_issue_numbers(nodes, "owner/repo", dropped=False) == [1, 2]
