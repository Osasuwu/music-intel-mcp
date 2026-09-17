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

The workflow runs this script in three modes, set by `MODE`:
- `resolve`: work out which issues this event touches and with which planner,
  and emit them as the matrix the `sync` job fans out over. Reads only the
  event's own shape (linked issues, dependents) — never labels.
- `apply`: one issue, one planner. Reads the issue's labels itself, so the
  snapshot it plans from is taken inside its own concurrency group.
- `sweep` (manual dispatch): one-off cleanup of `status:*` on closed issues.

Splitting resolve from apply is what makes the per-issue concurrency group
possible: a PR event's target issue numbers are only known after the GraphQL
lookup, so the workflow cannot key a group on them until `resolve` has run.

Stdlib only: the job needs no dependency install.
"""

import json
import os
import time
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
# Throttle handling: a sweep's burst of mutations trips GitHub's secondary rate
# limit, which answers 403 with no machine-readable marker beyond this phrase.
SECONDARY_LIMIT_HINT = "secondary rate limit"
MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 60
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


def was_blocked(issue):
    """True if the issue ever had a native blocker.

    Only such issues are this workflow's to promote: an issue that never had a
    blocker gets its status from triage, not from a `needs-*` label going away.
    """
    return (issue.get("issue_dependencies_summary") or {}).get("total_blocked_by", 0) > 0


def plan_unlabeled(issue):
    """A `needs-*` question was answered: promote, but only a once-blocked issue."""
    if not was_blocked(issue):
        return [], []
    return plan(issue)


PLANNERS = {
    "close": plan_close,
    "reopen": plan_reopen,
    "review": plan_review,
    "promote": plan,
    "unlabel": plan_unlabeled,
}


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


def _throttle_delay(status, headers, payload, attempt, now=None):
    """Seconds to wait before retrying a throttled request, or None if it isn't one.

    A `sweep` pass fires a long burst of mutations, and GitHub answers a burst
    with the *secondary* rate limit — which shares its status code with a plain
    permission failure, so the code alone can't tell them apart. The headers
    can: `Retry-After` is what the secondary limiter sends, and an exhausted
    primary limit sends `x-ratelimit-remaining: 0` with a reset epoch. Absent
    both, a 403 whose body doesn't name the secondary limit is a real 403 and
    must surface rather than be retried into silence. A 429 is unambiguous.
    """
    headers = headers or {}
    retry_after = _positive_int(headers.get("Retry-After"))
    if retry_after is not None:
        return min(retry_after, MAX_BACKOFF_SECONDS)
    if headers.get("x-ratelimit-remaining") == "0":
        reset = _positive_int(headers.get("x-ratelimit-reset"))
        now = int((now or time.time)())
        return min(max(reset - now, 1), MAX_BACKOFF_SECONDS) if reset else MAX_BACKOFF_SECONDS
    if status == 429 or SECONDARY_LIMIT_HINT in (payload or "").lower():
        return min(2**attempt, MAX_BACKOFF_SECONDS)
    return None


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _request(method, path, body=None):
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


def _api(method, path, body=None, attempts=MAX_ATTEMPTS, sleep=time.sleep):
    """Call the REST API, backing off while GitHub throttles us.

    Every call this script makes is idempotent (add a label it may already
    carry, drop one that may already be gone), so a retry can only repeat work,
    never corrupt it. Anything that isn't a throttle propagates with its body
    logged: a bare status line leaves the operator unable to tell a rate limit
    from a permission problem, which is the whole reason this helper exists.
    """
    for attempt in range(attempts):
        try:
            return _request(method, path, body)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            payload = exc.read().decode("utf-8", "replace")
            delay = _throttle_delay(exc.code, exc.headers, payload, attempt)
            if delay is None or attempt == attempts - 1:
                print(f"{method} {path} -> HTTP {exc.code}: {' '.join(payload.split())[:500]}")
                raise
            print(f"{method} {path} -> HTTP {exc.code}, throttled; retrying in {delay}s")
            sleep(delay)


def _remove_label(repo, num, name):
    """Drop a label, tolerating a concurrent run that dropped it first.

    Same-issue runs are serialised by the workflow's per-issue concurrency
    group, but a `sweep` pass or a hand edit can still delete a label between
    this run's read and its DELETE. An already-absent label is the intended end
    state, not a failure.
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
    """Run the planner's verdict against the API. True when anything changed."""
    add, remove = planner(issue)
    num = issue["number"]
    if add:
        _api("POST", f"repos/{repo}/issues/{num}/labels", {"labels": add})
    for name in remove:
        _remove_label(repo, num, name)
    print(f"#{num}: add={add} remove={remove}")
    return bool(add or remove)


def blocking_dependents(repo, closed):
    """Same-repo issue numbers the closed issue was blocking."""
    numbers, page = [], 1
    while True:
        batch = _api(
            "GET", f"repos/{repo}/issues/{closed}/dependencies/blocking?per_page=100&page={page}"
        )
        for dep in batch:
            if dependent_repo(dep, repo) is None:
                print(f"skip {dep.get('html_url', dep.get('number'))}: outside {repo}")
                continue
            numbers.append(dep["number"])
        if len(batch) < 100:
            break
        page += 1
    return numbers


def resolve_pr(repo, pr, action, merged):
    """Matrix entries for a PR event: the issues it closes, and how to replan them."""
    if action == "closed" and merged:
        print(f"PR #{pr} merged: its issues close on their own")
        return []
    dropped = action == "closed"
    owner, name = repo.split("/")
    variables = {"owner": owner, "name": name, "pr": int(pr)}
    data = _api("POST", "graphql", {"query": LINKED_ISSUES_QUERY, "variables": variables})
    nodes = data["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["nodes"]
    mode = "reopen" if dropped else "review"
    return [{"issue": n, "mode": mode} for n in pr_issue_numbers(nodes, repo, dropped)]


def resolve_issue(repo, number, action):
    """Matrix entries for an issue event: the issue itself, plus what it unblocks."""
    number = int(number)
    if action == "closed":
        freed = blocking_dependents(repo, number)
        return [{"issue": number, "mode": "close"}] + [
            {"issue": n, "mode": "promote"} for n in freed
        ]
    if action == "reopened":
        return [{"issue": number, "mode": "reopen"}]
    if action == "unlabeled":
        return [{"issue": number, "mode": "unlabel"}]
    raise SystemExit(f"unexpected EVENT_ACTION={action!r}")


def resolve(repo):
    event = os.environ.get("EVENT_NAME", "issues")
    action = os.environ.get("EVENT_ACTION", "closed")
    if event == "pull_request":
        targets = resolve_pr(
            repo, os.environ["PR_NUMBER"], action, os.environ.get("PR_MERGED") == "true"
        )
    else:
        targets = resolve_issue(repo, os.environ["ISSUE_NUMBER"], action)
    print(f"targets: {targets}")
    # A matrix caps at 256 entries. Truncating would silently skip an issue, so
    # let an over-long fan-out fail the run visibly instead.
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"targets={json.dumps(targets)}\n")


def sweep_closed(repo):
    """Strip `status:*` from every closed issue, one status label at a time.

    `GET /issues` lists pull requests alongside issues, and a label DELETE on a
    PR needs `pull-requests: write`, which this workflow deliberately does not
    grant — GitHub answers `403 Resource not accessible by integration` and the
    sweep dies on the first such PR. A PR's labels aren't this script's business
    anyway (it tracks the issue lifecycle), so PRs are skipped.
    """
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
        page = 1
        while batch := _api("GET", f"{query}&page={page}"):
            changed = False
            for item in batch:
                if "pull_request" in item:
                    continue
                changed |= apply(repo, item, plan_close)
            if changed:
                # The page shrank by what we just relabelled, so page `page`
                # refills from behind: ask for it again rather than stepping on.
                continue
            # Nothing on this page was ours to change — skipped PRs, or issues
            # the label index is still serving for a few seconds after the
            # DELETE that cleared them. Either way the page will not shrink, so
            # re-asking for it would spin forever; move on instead.
            if len(batch) < 100:
                break
            page += 1


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    mode = os.environ.get("MODE", "resolve")
    if mode == "resolve":
        resolve(repo)
    elif mode == "apply":
        number = os.environ["TARGET_ISSUE"]
        # Read labels here, inside this issue's concurrency group, so no other
        # run can write between the read and the plan built from it.
        issue = _api("GET", f"repos/{repo}/issues/{number}")
        apply(repo, issue, PLANNERS[os.environ["TARGET_MODE"]])
    elif mode == "sweep":
        sweep_closed(repo)
    else:
        raise SystemExit(f"unexpected MODE={mode!r}")


if __name__ == "__main__":
    main()
