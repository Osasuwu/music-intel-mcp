"""Label decisions of .github/scripts/unblock_ready.py (issue and PR lifecycle events)."""

import importlib.util
import urllib.error
from pathlib import Path

import pytest

from ._github_fake import FakeApi, FakeTransport, http_error, load_client

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


def test_issue_with_past_blocker_is_reevaluated():
    assert unblock_ready.was_blocked(_issue()) is True


def test_never_blocked_issue_is_not_promoted_on_unlabel():
    issue = {"issue_dependencies_summary": {"blocked_by": 0, "total_blocked_by": 0}}
    assert unblock_ready.was_blocked(issue) is False
    assert unblock_ready.was_blocked({}) is False


def test_close_strips_status_labels_but_keeps_hardware():
    issue = _issue(
        ["task", "status:in-progress", "status:rework-in-progress", "status:hardware-done"],
        state="closed",
    )
    assert unblock_ready.plan_close(issue) == (
        [],
        ["status:in-progress", "status:rework-in-progress"],
    )


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
    issue = _issue(["status:in-progress"])
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


def _main(monkeypatch, api, mode, **env):
    """Run main() in `mode` against `api`; answer the tokens make_client was given."""
    tokens = []
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "job-token")
    monkeypatch.setenv("MODE", mode)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    def make_client(token):
        tokens.append(token)
        return api

    unblock_ready.main(make_client=make_client)
    return tokens


def _deletes(api):
    return [path for method, path, _ in api.calls if method == "DELETE"]


def test_removing_a_label_a_racing_run_already_removed_is_not_an_error(capsys):
    api = FakeApi(writes={"repos/owner/repo/issues/7/labels/": http_error(404, "Not Found")})
    issue = dict(_issue(["status:ready"], state="closed"), number=7)
    unblock_ready.apply(api, "owner/repo", issue, unblock_ready.plan_close)
    assert api.calls == [("DELETE", "repos/owner/repo/issues/7/labels/status%3Aready", None)]
    assert "#7: status:ready already gone\n" in capsys.readouterr().out


def test_a_delete_failing_for_any_other_reason_still_raises():
    api = FakeApi(writes={"repos/owner/repo/issues/7/labels/": http_error(500, "boom")})
    issue = dict(_issue(["status:ready"], state="closed"), number=7)
    with pytest.raises(urllib.error.HTTPError) as caught:
        unblock_ready.apply(api, "owner/repo", issue, unblock_ready.plan_close)
    assert caught.value.code == 500


def _same_repo(number):
    return {"number": number, "repository_url": "https://api.github.com/repos/owner/repo"}


def _blocking_api(*pages):
    """#5's dependencies/blocking listing, answering page N with `pages[N - 1]`."""

    def page(method, path, body):
        return pages[int(path.rsplit("page=", 1)[1]) - 1]

    return FakeApi(responses={"repos/owner/repo/issues/5/dependencies/blocking": page})


def test_a_closed_issue_resolves_to_itself_plus_what_it_unblocks():
    other = {"number": 7, "repository_url": "https://api.github.com/repos/other/repo"}
    api = _blocking_api([_same_repo(6), other])
    assert unblock_ready.resolve_issue(api, "owner/repo", "5", "closed") == [
        {"issue": 5, "mode": "close"},
        {"issue": 6, "mode": "promote"},
    ]
    assert api.calls == [
        ("GET", "repos/owner/repo/issues/5/dependencies/blocking?per_page=100&page=1", None)
    ]


def _closed_issue_env(tmp_path):
    out = tmp_path / "github_output"
    env = {"EVENT_NAME": "issues", "EVENT_ACTION": "closed", "ISSUE_NUMBER": "5"}
    return out, dict(env, GITHUB_OUTPUT=str(out))


def test_main_resolves_a_closed_issue_into_its_targets(monkeypatch, tmp_path):
    out, env = _closed_issue_env(tmp_path)
    tokens = _main(monkeypatch, _blocking_api([_same_repo(6)]), "resolve", **env)
    assert tokens == ["job-token"]
    assert out.read_text(encoding="utf-8") == (
        'targets=[{"issue": 5, "mode": "close"}, {"issue": 6, "mode": "promote"}]\n'
    )


def test_a_blocking_listing_past_ten_full_pages_raises_and_writes_no_targets(monkeypatch, tmp_path):
    # Past its cap the listing may hold the one dependent this close frees: emitting
    # the targets read so far would leave that issue unpromoted with a green run.
    out, env = _closed_issue_env(tmp_path)
    full = [_same_repo(n) for n in range(100)]
    api = _blocking_api(*[full] * 10)
    with pytest.raises(load_client().ListingTruncated):
        _main(monkeypatch, api, "resolve", **env)
    assert not out.exists()
    assert len(api.calls) == 10


def test_reopen_and_unlabel_resolve_to_the_issue_alone():
    api = FakeApi()
    assert unblock_ready.resolve_issue(api, "owner/repo", "5", "reopened") == [
        {"issue": 5, "mode": "reopen"}
    ]
    assert unblock_ready.resolve_issue(api, "owner/repo", "5", "unlabeled") == [
        {"issue": 5, "mode": "unlabel"}
    ]
    assert api.calls == []


def _graphql_api(nodes, errors=None):
    answer = {
        "data": {"repository": {"pullRequest": {"closingIssuesReferences": {"nodes": nodes}}}}
    }
    if errors:
        answer["errors"] = errors
    return FakeApi(writes={"graphql": answer})


def test_a_pr_going_up_resolves_its_issues_to_review():
    api = _graphql_api([_linked(1), _linked(2, "other/repo")])
    targets = unblock_ready.resolve_pr(api, "owner/repo", "3", "opened", merged=False)
    assert targets == [{"issue": 1, "mode": "review"}]
    variables = {"owner": "owner", "name": "repo", "pr": 3}
    assert api.calls == [
        ("POST", "graphql", {"query": unblock_ready.LINKED_ISSUES_QUERY, "variables": variables})
    ]


def test_a_dropped_pr_resolves_its_issues_back_as_a_reopen():
    api = _graphql_api([_linked(1)])
    targets = unblock_ready.resolve_pr(api, "owner/repo", "3", "closed", merged=False)
    assert targets == [{"issue": 1, "mode": "reopen"}]


def test_a_graphql_errors_answer_raises_in_resolve_pr():
    # The answer still carries partial data; relabelling from it would act on a
    # lookup GitHub itself reported as failed.
    api = _graphql_api([_linked(1)], errors=[{"message": "Something went wrong"}])
    with pytest.raises(RuntimeError, match="GraphQL lookup failed"):
        unblock_ready.resolve_pr(api, "owner/repo", "3", "opened", merged=False)


def test_a_merged_pr_resolves_to_nothing():
    api = FakeApi()
    assert unblock_ready.resolve_pr(api, "owner/repo", "3", "closed", merged=True) == []
    assert api.calls == []


def test_every_mode_the_resolver_emits_has_a_planner():
    api = _graphql_api([_linked(1)])
    api.responses = _blocking_api([_same_repo(6)]).responses
    emitted = {t["mode"] for t in unblock_ready.resolve_pr(api, "owner/repo", "3", "opened", False)}
    emitted |= {
        t["mode"] for t in unblock_ready.resolve_pr(api, "owner/repo", "3", "closed", False)
    }
    for action in ("closed", "reopened", "unlabeled"):
        emitted |= {t["mode"] for t in unblock_ready.resolve_issue(api, "owner/repo", "5", action)}
    assert emitted == set(unblock_ready.PLANNERS)


CLOSED_LISTING = "repos/owner/repo/issues?state=closed&labels=status%3Areview"


def _sweep_api(*pages):
    """A sweep's API: one status label, then the closed `status:review` listing,
    answering page N with `pages[N - 1]` and an empty page past them.

    The listing does not shrink after a DELETE, as GitHub's label index keeps
    serving a cleared issue for seconds; it refuses a second read of a page, so a
    sweep that re-reads a page it changed fails here instead of spinning."""
    reads = set()

    def listing(method, path, body):
        if path in reads:
            raise AssertionError(f"closed listing page re-read: {path}")
        reads.add(path)
        number = int(path.rsplit("page=", 1)[1])
        return pages[number - 1] if number <= len(pages) else []

    return FakeApi(
        responses={"repos/owner/repo/labels": [{"name": "status:review"}], CLOSED_LISTING: listing}
    )


def _closed(number, is_pr=False):
    item = {"number": number, "state": "closed", "labels": [{"name": "status:review"}]}
    if is_pr:
        item["pull_request"] = {"url": f"https://api.github.com/repos/owner/repo/pulls/{number}"}
    return item


def test_sweep_leaves_pull_requests_alone():
    # `GET /issues` lists PRs too, and a label DELETE on a PR needs
    # `pull-requests: write` — the sweep job only has `issues: write`, so
    # touching one dies with 403 "Resource not accessible by integration".
    api = _sweep_api([_closed(1833, is_pr=True), _closed(1840)])
    unblock_ready.sweep_closed(api, "owner/repo")
    assert _deletes(api) == ["repos/owner/repo/issues/1840/labels/status%3Areview"]


def test_sweep_pages_past_a_full_page_of_pull_requests():
    # Nothing on page 1 is ours to relabel; the issues behind it must still get swept.
    api = _sweep_api([_closed(n, is_pr=True) for n in range(100)], [_closed(1840)])
    unblock_ready.sweep_closed(api, "owner/repo")
    assert _deletes(api) == ["repos/owner/repo/issues/1840/labels/status%3Areview"]


def test_sweep_reads_each_closed_issues_page_once():
    # The listing is read whole before any DELETE, so its pages cannot shift under
    # the sweep and a page it changed is never asked for again. #1841 is a stale
    # entry the label index still serves after the DELETE that cleared it.
    stale = {"number": 1841, "state": "closed", "labels": []}
    api = _sweep_api([_closed(1840), stale])
    unblock_ready.sweep_closed(api, "owner/repo")
    assert _deletes(api) == ["repos/owner/repo/issues/1840/labels/status%3Areview"]
    assert sum(1 for _, path, _ in api.calls if "state=closed" in path) == 1


def test_a_truncated_closed_listing_is_swept_as_far_as_it_was_read_and_then_raises():
    # A rerun converges: every issue stripped here leaves the listing.
    items = [_closed(0, is_pr=True)] + [_closed(n) for n in range(1, 1000)]
    api = _sweep_api(*[items[start : start + 100] for start in range(0, 1000, 100)])
    with pytest.raises(load_client().ListingTruncated):
        unblock_ready.sweep_closed(api, "owner/repo")
    deletes = _deletes(api)
    assert len(deletes) == 999
    assert deletes[0] == "repos/owner/repo/issues/1/labels/status%3Areview"
    assert deletes[-1] == "repos/owner/repo/issues/999/labels/status%3Areview"


def test_a_truncated_labels_listing_raises_before_any_closed_listing_is_read_or_written():
    # A sweep over part of the labels would skip every `status:*` label past the cap
    # with a green run; it must raise before reading a closed listing or writing.
    pages = [[{"name": f"status:s{page}-{n}"} for n in range(100)] for page in range(10)]

    def labels(method, path, body):
        return pages[int(path.rsplit("page=", 1)[1]) - 1]

    closed = [{"number": 9, "state": "closed", "labels": [{"name": "status:review"}]}]
    api = FakeApi(
        responses={
            "repos/owner/repo/labels": labels,
            "repos/owner/repo/issues?state=closed": closed,
        }
    )
    with pytest.raises(
        load_client().ListingTruncated, match=r"^listing repos/owner/repo/labels truncated: "
    ):
        unblock_ready.sweep_closed(api, "owner/repo")
    assert api.calls == [
        ("GET", f"repos/owner/repo/labels?per_page=100&page={n}", None) for n in range(1, 11)
    ]


def test_main_builds_its_own_client_with_the_job_token_and_the_unblock_ready_agent(monkeypatch):
    """The production `_make_client`, not an injected one: `main()` runs with only the
    transport patched, so a wrong token, user agent or constructor call goes red here."""
    gh = load_client()
    issue = dict(_issue(["status:ready"], state="closed"), number=5)
    transport = FakeTransport([issue, b""])
    monkeypatch.setattr(gh, "urllib_transport", transport)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "job-token")
    monkeypatch.setenv("MODE", "apply")
    monkeypatch.setenv("TARGET_ISSUE", "5")
    monkeypatch.setenv("TARGET_MODE", "close")
    unblock_ready.main()
    assert [
        (r.method, r.url, r.headers["authorization"], r.headers["user-agent"])
        for r in transport.requests
    ] == [
        (
            "GET",
            "https://api.github.com/repos/owner/repo/issues/5",
            "Bearer job-token",
            "unblock-ready",
        ),
        (
            "DELETE",
            "https://api.github.com/repos/owner/repo/issues/5/labels/status%3Aready",
            "Bearer job-token",
            "unblock-ready",
        ),
    ]


def test_an_unlabel_promotes_only_an_issue_that_was_once_blocked():
    assert unblock_ready.plan_unlabeled(_issue(["task"])) == (["status:ready"], [])
    never = {"state": "open", "labels": [], "issue_dependencies_summary": {"blocked_by": 0}}
    assert unblock_ready.plan_unlabeled(never) == ([], [])


# --- reconcile: the scheduled pass for blockers closed in another repo (#1981) ---


def _listed(number, labels=("task",), open_blockers=0, total_blockers=1, is_pr=False):
    """An item as `GET /issues?state=open` returns it."""
    item = {
        "number": number,
        "state": "open",
        "labels": [{"name": n} for n in labels],
        "issue_dependencies_summary": {
            "blocked_by": open_blockers,
            "total_blocked_by": total_blockers,
        },
    }
    if is_pr:
        item["pull_request"] = {"url": f"https://api.github.com/repos/owner/repo/pulls/{number}"}
    return item


def test_reconcile_promotes_an_issue_whose_every_blocker_has_closed():
    assert unblock_ready.plan_reconcile(_listed(1)) == (["status:ready"], [])


@pytest.mark.parametrize(
    "issue",
    [
        _listed(1, open_blockers=1),
        _listed(1, labels=("task", "status:in-progress")),
        _listed(1, labels=("task", "status:hardware-pending")),
        _listed(1, labels=("task", "needs-grill")),
        _listed(1, total_blockers=0),
        _listed(1, is_pr=True),
    ],
    ids=["open-blocker", "status-label", "hardware-status", "needs-label", "never-blocked", "pr"],
)
def test_reconcile_adds_nothing_to_an_issue_that_is_not_freshly_unblocked(issue):
    assert unblock_ready.plan_reconcile(issue) == ([], [])


def test_reconcile_leaves_an_issue_that_already_carries_ready_alone():
    assert unblock_ready.plan_reconcile(_listed(1, labels=("status:ready",))) == ([], [])


def _reconcile_api(*pages):
    """A reconcile's API: the open-issue listing answers page N with `pages[N - 1]`
    and an empty page past them; a single issue's GET answers from the same data,
    as the real endpoint does."""
    by_number = {i["number"]: i for items in pages for i in items}

    def listing(method, path, body):
        number = int(path.rsplit("page=", 1)[1])
        return pages[number - 1] if number <= len(pages) else []

    def one(method, path, body):
        return by_number[int(path.rsplit("/issues/", 1)[1])]

    return FakeApi(
        responses={"repos/owner/repo/issues?state=open": listing, "repos/owner/repo/issues/": one}
    )


def _writes(api):
    return [(m, p, b) for m, p, b in api.calls if m != "GET"]


def test_reconcile_labels_each_freshly_unblocked_issue_and_nothing_else():
    api = _reconcile_api(
        [
            _listed(10),
            _listed(11, open_blockers=1),
            _listed(12, labels=("task", "status:review")),
            _listed(13, is_pr=True),
            _listed(14, total_blockers=0),
        ]
    )
    unblock_ready.reconcile(api, "owner/repo")
    assert _writes(api) == [
        ("POST", "repos/owner/repo/issues/10/labels", {"labels": ["status:ready"]})
    ]


def test_reconcile_pages_through_a_full_page_of_open_issues():
    api = _reconcile_api([_listed(n, total_blockers=0) for n in range(100)], [_listed(500)])
    unblock_ready.reconcile(api, "owner/repo")
    assert [(m, p) for m, p, _ in _writes(api)] == [("POST", "repos/owner/repo/issues/500/labels")]


def test_reconcile_labels_what_it_read_and_then_raises_on_a_truncated_listing():
    full = [_listed(n, total_blockers=0) for n in range(100)]
    pages = [full] * 9 + [[_listed(900)] + full[1:]]
    api = _reconcile_api(*pages)
    with pytest.raises(load_client().ListingTruncated):
        unblock_ready.reconcile(api, "owner/repo")
    assert [(m, p) for m, p, _ in _writes(api)] == [("POST", "repos/owner/repo/issues/900/labels")]


def test_reconcile_replans_from_a_fresh_read_before_writing():
    # The listing is a snapshot: by the time the pass reaches #10 a PR event may
    # have moved it to review. The write must follow the labels as they are now.
    api = _reconcile_api([_listed(10)])
    api.responses["repos/owner/repo/issues/10"] = _listed(10, labels=("task", "status:review"))
    unblock_ready.reconcile(api, "owner/repo")
    assert _writes(api) == []


def test_main_runs_the_reconcile_mode(monkeypatch):
    api = _reconcile_api([_listed(10)])
    tokens = _main(monkeypatch, api, "reconcile")
    assert tokens == ["job-token"]
    assert [(m, p) for m, p, _ in _writes(api)] == [("POST", "repos/owner/repo/issues/10/labels")]


def test_main_applies_the_target_mode_to_a_fresh_read_of_the_issue(monkeypatch):
    api = FakeApi(responses={"repos/owner/repo/issues/7": dict(_closed(7))})
    _main(monkeypatch, api, "apply", TARGET_ISSUE="7", TARGET_MODE="close")
    assert api.calls == [
        ("GET", "repos/owner/repo/issues/7", None),
        ("DELETE", "repos/owner/repo/issues/7/labels/status%3Areview", None),
    ]
