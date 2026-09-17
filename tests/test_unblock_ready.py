"""Label decisions of .github/scripts/unblock_ready.py (issue and PR lifecycle events)."""

import importlib.util
import io
import urllib.error
from pathlib import Path

import pytest

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


def test_removing_a_label_a_racing_run_already_removed_is_not_an_error(monkeypatch):
    calls = []

    def fake_api(method, path, body=None):
        calls.append((method, path))
        if method == "DELETE":
            raise urllib.error.HTTPError(path, 404, "Label does not exist", {}, None)
        return None

    monkeypatch.setattr(unblock_ready, "_api", fake_api)
    issue = dict(_issue(["status:ready"], state="closed"), number=7)
    unblock_ready.apply("owner/repo", issue, unblock_ready.plan_close)
    assert calls == [("DELETE", "repos/owner/repo/issues/7/labels/status%3Aready")]


def test_a_delete_failing_for_any_other_reason_still_raises(monkeypatch):
    def fake_api(method, path, body=None):
        raise urllib.error.HTTPError(path, 500, "boom", {}, None)

    monkeypatch.setattr(unblock_ready, "_api", fake_api)
    issue = dict(_issue(["status:ready"], state="closed"), number=7)
    with pytest.raises(urllib.error.HTTPError):
        unblock_ready.apply("owner/repo", issue, unblock_ready.plan_close)


def _http_error(status, body="", headers=None):
    return urllib.error.HTTPError(
        "https://api.github.com/x", status, body, headers or {}, io.BytesIO(body.encode())
    )


def _api_over(monkeypatch, responses):
    """Drive `_api` against a scripted sequence, with a sleep that never blocks."""
    slept, queue = [], list(responses)

    def fake_request(method, path, body=None):
        outcome = queue.pop(0)
        if isinstance(outcome, urllib.error.HTTPError):
            raise outcome
        return outcome

    monkeypatch.setattr(unblock_ready, "_request", fake_request)
    return slept, lambda: queue


def test_a_throttled_call_backs_off_and_then_succeeds(monkeypatch):
    slept, _ = _api_over(
        monkeypatch,
        [_http_error(403, "You have exceeded a secondary rate limit"), {"ok": True}],
    )
    result = unblock_ready._api("DELETE", "repos/o/r/issues/1/labels/x", sleep=slept.append)
    assert result == {"ok": True}
    assert slept == [1]


def test_a_403_that_is_not_a_throttle_raises_without_retrying(monkeypatch):
    slept, remaining = _api_over(monkeypatch, [_http_error(403, "Resource not accessible")])
    with pytest.raises(urllib.error.HTTPError):
        unblock_ready._api("DELETE", "repos/o/r/issues/1/labels/x", sleep=slept.append)
    assert slept == []
    assert remaining() == []


def test_retry_after_sets_the_delay_and_is_capped(monkeypatch):
    slept, _ = _api_over(monkeypatch, [_http_error(429, "slow down", {"Retry-After": "999"}), None])
    unblock_ready._api("POST", "repos/o/r/issues/1/labels", sleep=slept.append)
    assert slept == [unblock_ready.MAX_BACKOFF_SECONDS]


def test_an_exhausted_primary_limit_waits_for_its_reset(monkeypatch):
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1000"}
    slept, _ = _api_over(monkeypatch, [_http_error(403, "rate limit", headers), None])
    monkeypatch.setattr(unblock_ready.time, "time", lambda: 970)
    unblock_ready._api("POST", "repos/o/r/issues/1/labels", sleep=slept.append)
    assert slept == [30]


def test_a_throttle_that_never_clears_gives_up_after_the_attempt_cap(monkeypatch):
    throttled = [_http_error(429, "slow down") for _ in range(unblock_ready.MAX_ATTEMPTS)]
    slept, remaining = _api_over(monkeypatch, throttled)
    with pytest.raises(urllib.error.HTTPError):
        unblock_ready._api("DELETE", "repos/o/r/issues/1/labels/x", sleep=slept.append)
    assert len(slept) == unblock_ready.MAX_ATTEMPTS - 1
    assert remaining() == []


def test_a_404_is_left_for_the_caller_to_tolerate(monkeypatch):
    slept, _ = _api_over(monkeypatch, [_http_error(404, "Label does not exist")])
    with pytest.raises(urllib.error.HTTPError) as caught:
        unblock_ready._api("DELETE", "repos/o/r/issues/1/labels/x", sleep=slept.append)
    assert caught.value.code == 404
    assert slept == []


def test_a_failing_call_logs_its_response_body(monkeypatch, capsys):
    slept, _ = _api_over(monkeypatch, [_http_error(403, "Resource not accessible by integration")])
    with pytest.raises(urllib.error.HTTPError):
        unblock_ready._api("DELETE", "repos/o/r/issues/1/labels/x", sleep=slept.append)
    assert "Resource not accessible by integration" in capsys.readouterr().out


def test_a_closed_issue_resolves_to_itself_plus_what_it_unblocks(monkeypatch):
    def fake_api(method, path, body=None):
        assert path.startswith("repos/owner/repo/issues/5/dependencies/blocking")
        return [
            {"number": 6, "repository_url": "https://api.github.com/repos/owner/repo"},
            {"number": 7, "repository_url": "https://api.github.com/repos/other/repo"},
        ]

    monkeypatch.setattr(unblock_ready, "_api", fake_api)
    assert unblock_ready.resolve_issue("owner/repo", "5", "closed") == [
        {"issue": 5, "mode": "close"},
        {"issue": 6, "mode": "promote"},
    ]


def test_reopen_and_unlabel_resolve_to_the_issue_alone():
    assert unblock_ready.resolve_issue("owner/repo", "5", "reopened") == [
        {"issue": 5, "mode": "reopen"}
    ]
    assert unblock_ready.resolve_issue("owner/repo", "5", "unlabeled") == [
        {"issue": 5, "mode": "unlabel"}
    ]


def _fake_graphql(monkeypatch, nodes):
    def fake_api(method, path, body=None):
        assert (method, path) == ("POST", "graphql")
        pr = {"closingIssuesReferences": {"nodes": nodes}}
        return {"data": {"repository": {"pullRequest": pr}}}

    monkeypatch.setattr(unblock_ready, "_api", fake_api)


def test_a_pr_going_up_resolves_its_issues_to_review(monkeypatch):
    _fake_graphql(monkeypatch, [_linked(1), _linked(2, "other/repo")])
    targets = unblock_ready.resolve_pr("owner/repo", "3", "opened", merged=False)
    assert targets == [{"issue": 1, "mode": "review"}]


def test_a_dropped_pr_resolves_its_issues_back_as_a_reopen(monkeypatch):
    _fake_graphql(monkeypatch, [_linked(1)])
    targets = unblock_ready.resolve_pr("owner/repo", "3", "closed", merged=False)
    assert targets == [{"issue": 1, "mode": "reopen"}]


def test_a_merged_pr_resolves_to_nothing():
    assert unblock_ready.resolve_pr("owner/repo", "3", "closed", merged=True) == []


def test_every_mode_the_resolver_emits_has_a_planner(monkeypatch):
    _fake_graphql(monkeypatch, [_linked(1)])
    emitted = {t["mode"] for t in unblock_ready.resolve_pr("owner/repo", "3", "opened", False)}
    emitted |= {t["mode"] for t in unblock_ready.resolve_pr("owner/repo", "3", "closed", False)}
    dep = [{"number": 6, "repository_url": "https://api.github.com/repos/owner/repo"}]
    monkeypatch.setattr(unblock_ready, "_api", lambda *a, **k: dep)
    for action in ("closed", "reopened", "unlabeled"):
        emitted |= {t["mode"] for t in unblock_ready.resolve_issue("owner/repo", "5", action)}
    assert emitted == set(unblock_ready.PLANNERS)


def _sweep_api(pages, calls):
    """Stub `_api` for a sweep: one status label, then `pages` of results.

    `pages` maps a 1-based page number to the list that page returns; a page
    with no entry returns []. Every DELETE is recorded in `calls`.
    """

    def fake_api(method, path, body=None):
        calls.append((method, path))
        if method == "DELETE":
            # The real API drops the label, so that issue leaves the result set.
            number = int(path.split("/issues/", 1)[1].split("/", 1)[0])
            for items in pages.values():
                items[:] = [i for i in items if i["number"] != number]
            return None
        if "/labels?" in path:
            return [{"name": "status:review"}]
        page = int(path.rsplit("page=", 1)[1])
        return pages.get(page, [])

    return fake_api


def _closed(number, is_pr=False):
    item = {"number": number, "state": "closed", "labels": [{"name": "status:review"}]}
    if is_pr:
        item["pull_request"] = {"url": f"https://api.github.com/repos/owner/repo/pulls/{number}"}
    return item


def test_sweep_leaves_pull_requests_alone(monkeypatch):
    # `GET /issues` lists PRs too, and a label DELETE on a PR needs
    # `pull-requests: write` — the sweep job only has `issues: write`, so
    # touching one dies with 403 "Resource not accessible by integration".
    calls = []
    pages = {1: [_closed(1833, is_pr=True), _closed(1840)]}
    monkeypatch.setattr(unblock_ready, "_api", _sweep_api(pages, calls))
    unblock_ready.sweep_closed("owner/repo")
    deletes = [path for method, path in calls if method == "DELETE"]
    assert deletes == ["repos/owner/repo/issues/1840/labels/status%3Areview"]


def test_sweep_pages_past_a_full_page_of_pull_requests(monkeypatch):
    # Page 1 never shrinks — nothing on it is ours to relabel — so re-asking
    # for it would spin forever. The issues behind it must still get swept.
    calls = []
    pages = {1: [_closed(n, is_pr=True) for n in range(100)], 2: [_closed(1840)]}
    monkeypatch.setattr(unblock_ready, "_api", _sweep_api(pages, calls))
    unblock_ready.sweep_closed("owner/repo")
    deletes = [path for method, path in calls if method == "DELETE"]
    assert deletes == ["repos/owner/repo/issues/1840/labels/status%3Areview"]


def test_an_unlabel_promotes_only_an_issue_that_was_once_blocked():
    assert unblock_ready.plan_unlabeled(_issue(["task"])) == (["status:ready"], [])
    never = {"state": "open", "labels": [], "issue_dependencies_summary": {"blocked_by": 0}}
    assert unblock_ready.plan_unlabeled(never) == ([], [])
