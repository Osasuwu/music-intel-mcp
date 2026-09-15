"""Promote issues to `status:ready` when their last open blocker closes.

Runs from `.github/workflows/unblock-ready.yml` on `issues: closed`. Native
blocked_by edges record a block but never flip the status label, so without
this a slice stays unready after its blocker ships (closed by PR or by hand).
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
    add = [] if READY in labels else [READY]
    remove = [BLOCKED] if BLOCKED in labels else []
    return add, remove


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


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    closed = os.environ["ISSUE_NUMBER"]
    page = 1
    while True:
        batch = _api(
            "GET", f"repos/{repo}/issues/{closed}/dependencies/blocking?per_page=100&page={page}"
        )
        for dep in batch:
            # Re-fetch: the dependency summary is the source of truth for open blockers.
            issue = _api("GET", f"repos/{repo}/issues/{dep['number']}")
            add, remove = plan(issue)
            num = issue["number"]
            if add:
                _api("POST", f"repos/{repo}/issues/{num}/labels", {"labels": add})
            for name in remove:
                _api("DELETE", f"repos/{repo}/issues/{num}/labels/{urllib.parse.quote(name)}")
            print(f"#{num}: add={add} remove={remove}")
        if len(batch) < 100:
            break
        page += 1


if __name__ == "__main__":
    main()
