"""Keep `status:*` labels in step with the issue and PR lifecycle.

Runs from `.github/workflows/unblock-ready.yml`. Native blocked_by edges record
a block but never flip the status label, and closing, reopening or opening a PR
never touches an issue's status either, so without this labels drift.

Issue events:
- `closed`: strip the closed issue's `status:*` labels (a closed issue has no
  work status; `status:hardware-*` belongs to the hardware lifecycle), then
  promote every issue it was blocking whose last open blocker it was.
- `reopened`: drop stale in-flight statuses and put the issue back to ready
  when nothing holds it (open blocker, `needs-*`, another status).
- `unlabeled` (a `needs-*` label): an issue skipped for an open `needs-*`
  question is re-evaluated once that question is answered.

PR events (issues the PR closes via `Closes #N`, same repo only):
- opened / reopened / ready for review / edited: the issues move to review.
- closed without merge: the issues go back as if reopened, unless another
  open PR still closes them. A merge needs nothing here: it closes the issue.

`sweep` (manual dispatch): one-off cleanup of `status:*` on closed issues.

Stdlib only: the job needs no dependency install.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

READY = "status:ready"
REVIEW = "status:review"
# Status labels that may coexist with ready; any other `status:*` means the
# issue already moved past ready and must not be overwritten.
COEXISTS_WITH_READY = {READY, "status:owner-queue"}
# Statuses of work under way; stale once the issue is reopened or its PR dropped.
IN_FLIGHT = {"status:in-progress", REVIEW, "status:rework-in-progress"}
# Set by a hardware lifecycle workflow on close/reopen; not ours to touch.
HARDWARE_PREFIX = "status:hardware-"
# `needs-grill`, `needs-triage`, `needs-safety-review`, ...: an open question
# that must be answered before the issue is ready.
NEEDS_PREFIX = "needs-"

LINKED_ISSUES_QUERY = """
query($owner: String!, $name: String!, $pr: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $pr) {
      closingIssuesReferences(first: 50) {
        nodes {
          number
          repository { nameWithOwner }
          closedByPullRequestsReferences(first: 1, includeClosedPrs: false) { nodes { number } }
        }
      }
    }
  }
}
"""


def _names(issue):
    return {label["name"] for label in issue.get("labels", [])}


def _held(labels):
    """True if a label other than an open blocker keeps the issue from ready."""
    if any(n.startswith("status:") and n not in COEXISTS_WITH_READY for n in labels):
        return True
    return any(n.startswith(NEEDS_PREFIX) for n in labels)


def _unblocked(issue):
    # `blocked_by` counts only OPEN blockers (`total_blocked_by` counts all).
    return (issue.get("issue_dependencies_summary") or {}).get("blocked_by", 1) == 0


def plan(issue):
    """Return (labels_to_add, labels_to_remove) for an issue the closed one was blocking."""
    labels = _names(issue)
    if issue.get("state") != "open" or not _unblocked(issue) or _held(labels):
        return [], []
    return ([] if READY in labels else [READY]), []


def plan_close(issue):
    """Strip work statuses from a closed issue."""
    if issue.get("state") != "closed":
        return [], []
    status = {n for n in _names(issue) if n.startswith("status:")}
    return [], sorted(n for n in status if not n.startswith(HARDWARE_PREFIX))


def plan_reopen(issue):
    """Drop stale in-flight statuses and restore ready when nothing holds the issue."""
    if issue.get("state") != "open":
        return [], []
    labels = _names(issue)
    rest = labels - IN_FLIGHT
    add = [READY] if READY not in rest and _unblocked(issue) and not _held(rest) else []
    return add, sorted(labels & IN_FLIGHT)


def plan_review(issue):
    """A PR that closes the issue is up: ready / in-progress give way to review."""
    labels = _names(issue)
    if issue.get("state") != "open" or any(n.startswith(HARDWARE_PREFIX) for n in labels):
        return [], []
    stale = labels & (IN_FLIGHT | {READY}) - {REVIEW}
    return ([] if REVIEW in labels else [REVIEW]), sorted(stale)


def pr_issue_numbers(nodes, repo, dropped):
    """Same-repo issue numbers a PR closes.

    `dropped` (PR closed unmerged): skip issues another open PR still closes,
    their work is still in flight.
    """
    numbers = []
    for node in nodes:
        if node["repository"]["nameWithOwner"].lower() != repo.lower():
            continue
        if dropped and node["closedByPullRequestsReferences"]["nodes"]:
            continue
        numbers.append(node["number"])
    return numbers


def was_blocked(issue):
    """True if the issue ever had a native blocker.

    Only such issues are this workflow's to promote: an issue that never had a
    blocker gets its status from triage, not from a `needs-*` label going away.
    """
    return (issue.get("issue_dependencies_summary") or {}).get("total_blocked_by", 0) > 0


def dependent_repo(dep, repo):
    """Return `repo` if the dependent lives in it, else None.

    Dependencies can cross repos, and `number` alone is ambiguous: jarvis#1162
    blocking redrobot#1412 must not relabel jarvis#1412. GITHUB_TOKEN can only
    write to its own repo, so cross-repo dependents are skipped.
    """
    url = dep.get("repository_url") or ""
    if "/repos/" not in url:
        return None
    return repo if url.split("/repos/", 1)[1].lower() == repo.lower() else None


def _api(method, path, body=None):
    req = urllib.request.Request(
        f"https://api.github.com/{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else None


def _remove_label(repo, num, name):
    """Drop a label, tolerating a concurrent run that dropped it first.

    Two runs on the same issue (a PR event and an issue event, say) can read the
    same labels; the second DELETE then 404s on a label that is already gone.
    That is the intended end state, not a failure.
    """
    try:
        _api("DELETE", f"repos/{repo}/issues/{num}/labels/{urllib.parse.quote(name)}")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        print(f"#{num}: {name} already gone")
        return False


def apply(repo, issue, planner=plan):
    add, remove = planner(issue)
    num = issue["number"]
    if add:
        _api("POST", f"repos/{repo}/issues/{num}/labels", {"labels": add})
    for name in remove:
        _remove_label(repo, num, name)
    print(f"#{num}: add={add} remove={remove}")


def promote_dependents(repo, closed):
    page = 1
    while True:
        batch = _api(
            "GET", f"repos/{repo}/issues/{closed}/dependencies/blocking?per_page=100&page={page}"
        )
        for dep in batch:
            if dependent_repo(dep, repo) is None:
                print(f"skip {dep.get('html_url', dep.get('number'))}: outside {repo}")
                continue
            # Re-fetch: the dependency summary is the source of truth for open blockers.
            apply(repo, _api("GET", f"repos/{repo}/issues/{dep['number']}"))
        if len(batch) < 100:
            break
        page += 1


def sync_pr_issues(repo, pr, action, merged):
    if action == "closed" and merged:
        print(f"PR #{pr} merged: its issues close on their own")
        return
    dropped = action == "closed"
    owner, name = repo.split("/")
    variables = {"owner": owner, "name": name, "pr": int(pr)}
    data = _api("POST", "graphql", {"query": LINKED_ISSUES_QUERY, "variables": variables})
    nodes = data["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["nodes"]
    for number in pr_issue_numbers(nodes, repo, dropped):
        issue = _api("GET", f"repos/{repo}/issues/{number}")
        apply(repo, issue, plan_reopen if dropped else plan_review)


def sweep_closed(repo):
    """Strip `status:*` from every closed issue, one status label at a time."""
    names, page = [], 1
    while True:
        batch = _api("GET", f"repos/{repo}/labels?per_page=100&page={page}")
        names += [label["name"] for label in batch]
        if len(batch) < 100:
            break
        page += 1
    for name in names:
        if not name.startswith("status:") or name.startswith(HARDWARE_PREFIX):
            continue
        query = f"repos/{repo}/issues?state=closed&per_page=100&labels={urllib.parse.quote(name)}"
        # Always page 1: every pass removes the label, so the result set shrinks.
        while batch := _api("GET", query):
            for issue in batch:
                apply(repo, issue, plan_close)


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    event = os.environ.get("EVENT_NAME", "issues")
    action = os.environ.get("EVENT_ACTION", "closed")
    if event == "workflow_dispatch":
        sweep_closed(repo)
        return
    if event == "pull_request":
        sync_pr_issues(repo, os.environ["PR_NUMBER"], action, os.environ.get("PR_MERGED") == "true")
        return
    number = os.environ["ISSUE_NUMBER"]
    issue = _api("GET", f"repos/{repo}/issues/{number}")
    if action == "closed":
        apply(repo, issue, plan_close)
        promote_dependents(repo, number)
    elif action == "reopened":
        apply(repo, issue, plan_reopen)
    elif action == "unlabeled":
        if was_blocked(issue):
            apply(repo, issue)
        else:
            print(f"#{number}: never blocked, status is triage's call")
    else:
        raise SystemExit(f"unexpected EVENT_ACTION={action!r}")


if __name__ == "__main__":
    main()
