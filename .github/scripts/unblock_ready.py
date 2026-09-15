"""Promote issues to `status:ready` when their last open blocker closes.

Runs from `.github/workflows/unblock-ready.yml`. Native blocked_by edges record
a block but never flip the status label, so without this a slice stays unready
after its blocker ships (closed by PR or by hand).

- `closed`: evaluate every issue the closed one was blocking.
- `unlabeled` (a `needs-*` label): an issue skipped for an open `needs-*`
  question is re-evaluated once that question is answered.

Stdlib only: the job needs no dependency install.
"""

import json
import os
import urllib.parse
import urllib.request

READY = "status:ready"
BLOCKED = "status:blocked"
# Status labels that may coexist with ready; any other `status:*` means the
# issue already moved past ready and must not be overwritten.
COEXISTS_WITH_READY = {READY, BLOCKED, "status:owner-queue"}
# `needs-grill`, `needs-triage`, `needs-safety-review`, ...: an open question
# that must be answered before the issue is ready.
NEEDS_PREFIX = "needs-"


def plan(issue):
    """Return (labels_to_add, labels_to_remove) for an issue the closed one was blocking."""
    if issue.get("state") != "open":
        return [], []
    summary = issue.get("issue_dependencies_summary") or {}
    # `blocked_by` counts only OPEN blockers (`total_blocked_by` counts all).
    if summary.get("blocked_by", 1) != 0:
        return [], []
    labels = {label["name"] for label in issue.get("labels", [])}
    if any(n.startswith("status:") and n not in COEXISTS_WITH_READY for n in labels):
        return [], []
    if any(n.startswith(NEEDS_PREFIX) for n in labels):
        return [], []
    add = [] if READY in labels else [READY]
    remove = [BLOCKED] if BLOCKED in labels else []
    return add, remove


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


def apply(repo, issue):
    add, remove = plan(issue)
    num = issue["number"]
    if add:
        _api("POST", f"repos/{repo}/issues/{num}/labels", {"labels": add})
    for name in remove:
        _api("DELETE", f"repos/{repo}/issues/{num}/labels/{urllib.parse.quote(name)}")
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


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    number = os.environ["ISSUE_NUMBER"]
    action = os.environ.get("EVENT_ACTION", "closed")
    if action == "closed":
        promote_dependents(repo, number)
    elif action == "unlabeled":
        issue = _api("GET", f"repos/{repo}/issues/{number}")
        if was_blocked(issue):
            apply(repo, issue)
        else:
            print(f"#{number}: never blocked, status is triage's call")
    else:
        raise SystemExit(f"unexpected EVENT_ACTION={action!r}")


if __name__ == "__main__":
    main()
