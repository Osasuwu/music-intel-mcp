"""The `verify-verdict` verdict: green only on review evidence bound to the evaluated commit.

Runs from `.github/workflows/code-gate-verdict.yml` (the base-pinned verdict) and,
for its review-side subcommands, from `.github/workflows/code-review.yml` (the
evidence producer). Ported from Osasuwu/jarvis#1964 (locked design); issue #226.

The rule ("Review evidence"). `verify-verdict` is green iff
  (a) at least one successful `code-review.yml` run bound to the evaluated head
      SHA carries a valid, non-blocking `review-evidence.json` artifact, no
      artifact of any bound run or attempt is blocking (sticky: each attempt
      uploads its own artifact, so neither a re-run nor a later clean run lifts
      it, only a new commit does — but only while the artifact is retained, see
      Osasuwu/like-current-song#236), none could not be downloaded, and no bound
      run is unfinished; an expired, missing, malformed or mismatched artifact
      is red only until a clean run for the SHA supersedes it (`status: skipped`
      artifacts are ignored), or
  (b) every changed file is cosmetic (see `is_cosmetic`).
The gate never reads the PR comment, a timestamp, a heading or a lineage: the
comment is for humans, the artifact is the machine-readable verdict.

Run provenance. A `pull_request` run counts only when `run.pull_requests[]`
contains the PR and every PR it lists is into the default branch (a fork's run
carries an empty list). `pull_requests[]` lists the open PRs on the run's head,
not the PR that triggered it, so a run listing any PR into another base may be
running that base's copy of the workflow: it is evidence for no PR, and the
verdict says so (like-current-song#241). A `workflow_dispatch` run counts only when its head
commit is in the default branch's history (`head_branch` is not enough: a tag
named like the branch reads the same, like-current-song#241) and its `display_title` — the
workflow's own `run-name`, which embeds `pr_number` and the `head_sha` input —
names this PR and this SHA. The artifact's own `sha` and `base_ref` must equal
the evaluated SHA and the PR's base: a mismatch is red.

One commit, one check. `verify-verdict` attaches to the head commit, not the PR
(like-current-song#230): a PR into another base posts nothing, and two open PRs into the default
branch on one head SHA make the check red until one of them moves or closes.

Gate machinery. A PR touching the review workflow, the verdict workflow, this
script or a local action is always red: its own review cannot be trusted, since
it is the thing under change. The sanctioned unblock is a human review-blind
admin-merge by the maintainer, backed by a fresh-session review of the final SHA
(see docs/ci-gates.md).

Every decision below is a pure function over plain dicts and lists; the API
access (`Api`, `gather_entries`, `post_check`) is a thin wrapper around them.
Stdlib only: the jobs need no dependency install.
"""

import argparse
import base64
import http.client
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from typing import NamedTuple

API_ROOT = "https://api.github.com"
REVIEW_WORKFLOW = ".github/workflows/code-review.yml"
CHECK_NAME = "verify-verdict"
EVIDENCE_ARTIFACT = "review-evidence"
# One evidence artifact per run attempt (`review-evidence-<run_attempt>`, never
# overwritten): a re-run cannot replace the evidence an earlier attempt left.
EVIDENCE_ARTIFACT_NAME = re.compile(rf"{EVIDENCE_ARTIFACT}-[1-9][0-9]*")
EVIDENCE_FILE = "review-evidence.json"
EVIDENCE_SCHEMA = 1
MAX_EVIDENCE_BYTES = 1 << 20
MAX_ARTIFACT_BYTES = MAX_EVIDENCE_BYTES * 8
DEPENDABOT = "dependabot[bot]"

FINDING_CLASSES = (
    "regression",
    "exception-handling",
    "intent-vs-logic",
    "breaking-contract",
    "concurrency",
    "requirement-semantics",
    "design-modularity",
    "performance",
)

# `run-name` of code-review.yml: `Code review PR #<n> @ <40-hex head sha>`.
RUN_TITLE = re.compile(r"^Code review PR #(\d+) @ ([0-9a-f]{40})$")

# The change set the gate itself is made of. A PR touching any of it is judged
# by a human, never by its own review.
# music-intel-mcp: gate machinery + anything affecting agent/gate behavior
GATE_MACHINERY_FILES = frozenset(
    {
        REVIEW_WORKFLOW,
        ".github/workflows/code-gate-verdict.yml",
        ".github/scripts/code_gate_verdict.py",
        # Agent/gate behavior files (must be reviewed by human if changed)
        ".claude/hooks/secret-scanner.py",
        ".claude/marketplace/.claude-plugin/marketplace.json",
        ".claude/settings.json",
    }
)
GATE_MACHINERY_PREFIXES = (".github/actions/", ".claude/hooks/")

_IMAGE_EXT = (".png", ".jpg", ".gif", ".webp")
_DOC_EXT = (".md",) + _IMAGE_EXT
# Root-level cosmetic files: docs/readme only. AGENTS.md, CONTEXT.md, INVARIANTS.md
# are code (affect domain model and gate behavior).
_ROOT_COSMETIC = frozenset(
    {
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_LICENSES",
        "LICENSE",
        "LICENSE.md",
        "LICENSE.txt",
        "LICENSE-APACHE",
        "LICENSE-MIT",
    }
)

# Runs that finished without producing a verdict: no evidence either way.
_IGNORED_CONCLUSIONS = frozenset(
    {"failure", "cancelled", "timed_out", "skipped", "neutral", "stale", "startup_failure"}
)

MAX_RUN_PAGES = 3
MAX_PR_PAGES = 10
MAX_FILE_PAGES = 30  # the PR files API serves at most 3000 files
MAX_HEAD_FILES = 200
MAX_HEAD_FILE_BYTES = 1 << 20
MAX_HEAD_TOTAL_BYTES = 20 << 20


class Verdict(NamedTuple):
    green: bool
    code: str
    message: str


# --- classification -------------------------------------------------------


def is_cosmetic(path):
    """One path against the allow-list. Anything not listed is code.

    music-intel-mcp rules:
    - Images anywhere: cosmetic
    - docs/domain/: product docs, cosmetic
    - docs/reference/: behavior-carrying, code
    - Root .md: README only (cosmetic); AGENTS.md, CONTEXT.md, INVARIANTS.md are code
    - .claude/*: code (affects gate/agent behavior)
    """
    if path.startswith("/") or "\\" in path or ".." in path.split("/"):
        return False
    if path.endswith(_IMAGE_EXT):
        return True
    if "/" not in path:
        return path in _ROOT_COSMETIC
    # docs/domain/: cosmetic. docs/ otherwise: code
    if path.startswith("docs/domain/"):
        return path.endswith(_DOC_EXT)
    return False


def classify_paths(paths):
    """Split paths into `code` and `cosmetic`, each in input order."""
    out = {"code": [], "cosmetic": []}
    for path in paths:
        out["cosmetic" if is_cosmetic(path) else "code"].append(path)
    return out


def changed_paths(files):
    """Every path a PR-files payload touches: a rename or move counts on both sides."""
    paths = []
    for f in files:
        for key in ("filename", "previous_filename"):
            name = f.get(key)
            if name and name not in paths:
                paths.append(name)
    return paths


def is_gate_machinery(paths):
    return any(p in GATE_MACHINERY_FILES or p.startswith(GATE_MACHINERY_PREFIXES) for p in paths)


def is_untrusted_author(pr):
    """A fork head or Dependabot: a `pull_request` run gets no secrets, so no review."""
    head_repo = (pr.get("head") or {}).get("repo") or {}
    base_repo = (pr.get("base") or {}).get("repo") or {}
    if not head_repo.get("full_name") or head_repo.get("full_name") != base_repo.get("full_name"):
        return True
    return (pr.get("user") or {}).get("login") == DEPENDABOT


def is_draft(pr):
    return pr.get("draft") is True


# --- review-side helpers --------------------------------------------------


def validate_findings(obj):
    """Problems with a findings object; empty list means valid."""
    if not isinstance(obj, dict):
        return ["findings file is not a JSON object"]
    errors = []
    blocking = obj.get("blocking")
    findings = obj.get("findings")
    if not isinstance(blocking, bool):
        errors.append("`blocking` must be a boolean")
    if not isinstance(findings, list):
        errors.append("`findings` must be a list")
        return errors
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            errors.append(f"findings[{i}] is not an object")
            continue
        if f.get("class") not in FINDING_CLASSES:
            errors.append(f"findings[{i}].class {f.get('class')!r} is not one of the 8 classes")
        if not isinstance(f.get("file"), str) or not f.get("file"):
            errors.append(f"findings[{i}].file must be a non-empty string")
    if isinstance(blocking, bool) and blocking != bool(findings):
        errors.append("`blocking` must be true exactly when `findings` is non-empty")
    return errors


def check_clean_tree(porcelain, findings_path):
    """Paths dirty in `git status --porcelain` output, apart from the findings file."""
    dirty = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip().strip('"')
        if path != findings_path:
            dirty.append(path)
    return dirty


def build_evidence(findings, sha, base_ref):
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "reviewed",
        "sha": sha,
        "base_ref": base_ref,
        "blocking": findings["blocking"],
        "findings": findings["findings"],
    }


def build_skipped_evidence(sha, base_ref, reason):
    return {
        "schema": EVIDENCE_SCHEMA,
        "status": "skipped",
        "sha": sha,
        "base_ref": base_ref,
        "reason": reason,
    }


def review_skip_reason(pr, files, event_name):
    """Why the review job runs no reviewer for this PR ('' = review it)."""
    if is_draft(pr):
        return "draft"
    if len(files) != pr.get("changed_files"):
        return "files-incomplete"
    if not classify_paths(changed_paths(files))["code"]:
        return "cosmetic"
    if event_name == "pull_request" and is_untrusted_author(pr):
        return "untrusted"
    return ""


def resolve_run_target(event, display_title, head_sha, pull_requests, default_branch):
    """(pr_number, head_sha) a completed review run was for, or None.

    A `pull_request` run resolves to the PR it lists into the default branch: the
    list holds every open PR on the run's head, in no promised order, and only a
    PR into the default branch has a `verify-verdict` to decide (like-current-song#230).
    """
    if event == "workflow_dispatch":
        m = RUN_TITLE.match(display_title or "")
        return (int(m.group(1)), m.group(2)) if m else None
    if event == "pull_request":
        for p in pull_requests or []:
            if (p.get("base") or {}).get("ref") == default_branch:
                return (p["number"], head_sha)
    return None


# --- run provenance and evidence ------------------------------------------


# Set on a dispatch run by `gather_entries` (never taken from the API payload):
# is the run's head commit the default branch's tip or one of its ancestors?
IN_DEFAULT_HISTORY = "_in_default_history"


def run_provenance(run, pr_number, head_sha, default_branch):
    """How this run binds to this PR at this SHA, through something it cannot forge.

    'bound': it is evidence. 'foreign-base': a `pull_request` run that lists this
    PR and also a PR into another base, so it may have run that base's copy of the
    workflow; it is evidence for no PR. '': it is not about this PR at this SHA.
    """
    if (run.get("path") or "").split("@")[0] != REVIEW_WORKFLOW:
        return ""
    event = run.get("event")
    if event == "pull_request":
        listed = run.get("pull_requests") or []
        if run.get("head_sha") != head_sha or not any(p.get("number") == pr_number for p in listed):
            return ""
        if any((p.get("base") or {}).get("ref") != default_branch for p in listed):
            return "foreign-base"
        return "bound"
    if event == "workflow_dispatch":
        m = RUN_TITLE.match(run.get("display_title") or "")
        if not m or int(m.group(1)) != pr_number or m.group(2) != head_sha:
            return ""
        return "bound" if run.get(IN_DEFAULT_HISTORY) is True else ""
    return ""


_WORST_FIRST = (
    "evidence-foreign-base",
    "evidence-sha-mismatch",
    "evidence-base-mismatch",
    "evidence-invalid",
    "evidence-blocking",
    "evidence-expired",
    "evidence-missing",
)

_EVIDENCE_MESSAGES = {
    "evidence-pending": (
        "A bound review run is not finished (or awaits approval). "
        "Wait for it, or approve/cancel it."
    ),
    "evidence-foreign-base": (
        "A review run for this commit also lists an open PR into another base branch, "
        "so it may have run that branch's copy of the review workflow; it is evidence "
        "for no PR. Close or retarget that PR, or have a maintainer re-dispatch the review."
    ),
    "evidence-sha-mismatch": (
        "A review artifact is stamped for a different commit than the one evaluated. "
        "Re-dispatch the review."
    ),
    "evidence-base-mismatch": (
        "A review artifact was produced against a different base branch. Re-dispatch the review."
    ),
    "evidence-invalid": "A review artifact is malformed. Re-dispatch the review.",
    "evidence-blocking": (
        "A review run reported blocking findings for this commit; that stays in force "
        "for the SHA, across re-runs. Fix them and push a new commit."
    ),
    "evidence-unreadable": (
        "A review artifact could not be downloaded, so it may hide a blocking verdict. "
        "Re-run the verdict job."
    ),
    "evidence-expired": (
        "A review artifact has expired (90-day retention). Re-dispatch the review."
    ),
    "evidence-missing": (
        "A successful review run has no review-evidence-<attempt> artifact. Re-dispatch the review."
    ),
    "evidence-none": (
        "No successful review run is bound to this commit. "
        "Push to trigger one, or re-dispatch the review."
    ),
    "evidence-clean": "A bound review run carries clean evidence for this commit.",
}


def _verdict(code):
    return Verdict(code == "evidence-clean", code, _EVIDENCE_MESSAGES[code])


def _judge_artifact(art, head_sha, base_ref):
    """'skipped', 'clean', or the red code for one artifact."""
    if not isinstance(art, dict):
        return "evidence-invalid"
    if art.get("status") == "skipped":
        return "skipped"
    if art.get("schema") != EVIDENCE_SCHEMA or art.get("status") != "reviewed":
        return "evidence-invalid"
    if validate_findings(art):
        return "evidence-invalid"
    if art.get("sha") != head_sha:
        return "evidence-sha-mismatch"
    if art.get("base_ref") != base_ref:
        return "evidence-base-mismatch"
    return "evidence-blocking" if art["blocking"] else "clean"


def evaluate_evidence(entries, pr_number, head_sha, base_ref, default_branch):
    """The evidence half of the rule. `entries`: {run, artifact, state}, one per
    evidence artifact of a candidate run (see `load_entries`)."""
    reds = set()
    clean = 0
    for e in entries:
        run = e["run"]
        provenance = run_provenance(run, pr_number, head_sha, default_branch)
        if not provenance:
            continue
        if run.get("status") != "completed" or run.get("conclusion") == "action_required":
            return _verdict("evidence-pending")
        state = e.get("state")
        if state == "unreadable":
            reds.add("evidence-unreadable")
            continue
        if state == "expired":
            result = "evidence-expired"
        elif state != "ok" or e.get("artifact") is None:
            result = "evidence-missing"
        else:
            result = _judge_artifact(e["artifact"], head_sha, base_ref)
        if provenance == "foreign-base":
            # Its evidence cannot clear this PR; a blocking verdict it carries can
            # only be stricter, so that one still sticks.
            reds.add(
                "evidence-blocking" if result == "evidence-blocking" else "evidence-foreign-base"
            )
            continue
        if run.get("conclusion") != "success":
            # The run's latest attempt failed or was cancelled. Blocking evidence an
            # earlier attempt left still counts; nothing else from the run does.
            if result == "evidence-blocking":
                reds.add(result)
            continue
        if result == "clean":
            clean += 1
        elif result != "skipped":
            reds.add(result)
    # Blocking is the one sticky red: a later clean run or attempt for the same SHA
    # cannot lift it, the fix is a new commit. An artifact that could not be read
    # might be a blocking one, so a clean run does not lift that either. Every other
    # red is a failure to produce evidence, not a verdict on the code, so a clean
    # run for the SHA supersedes it — that is what "re-dispatch the review" in its
    # message relies on.
    if "evidence-blocking" in reds:
        return _verdict("evidence-blocking")
    if "evidence-unreadable" in reds:
        return _verdict("evidence-unreadable")
    if clean:
        return _verdict("evidence-clean")
    for code in _WORST_FIRST:
        if code in reds:
            return _verdict(code)
    return _verdict("evidence-none")


def evaluate_pr(pr, files, entries, default_branch):
    """The whole rule for one PR at its current head."""
    if is_draft(pr):
        return Verdict(False, "draft", "draft")
    if len(files) != pr.get("changed_files"):
        return Verdict(
            False,
            "files-incomplete",
            "The changed-file list is incomplete (the API serves at most 3000 files); "
            "the PR cannot be classified.",
        )
    if not files:
        return Verdict(False, "files-empty", "The PR changes no files.")
    paths = changed_paths(files)
    if is_gate_machinery(paths):
        return Verdict(
            False,
            "gate-machinery",
            "This PR changes the gate itself, so its own review cannot be trusted. "
            "A human merges it by review-blind admin-merge, backed by a fresh-session "
            "/code-review posted with the final SHA.",
        )
    if not classify_paths(paths)["code"]:
        return Verdict(True, "cosmetic", "Every changed file is cosmetic; no review needed.")
    result = evaluate_evidence(
        entries,
        pr_number=pr["number"],
        head_sha=pr["head"]["sha"],
        base_ref=pr["base"]["ref"],
        default_branch=default_branch,
    )
    if result.code == "evidence-none" and is_untrusted_author(pr):
        return Verdict(
            False,
            "untrusted-needs-dispatch",
            "Fork and Dependabot PRs get no automatic review. A maintainer runs the "
            f"`{REVIEW_WORKFLOW}` workflow_dispatch with this `pr_number` and `head_sha` "
            "(environment `untrusted-review`).",
        )
    return result


# --- GitHub API wrapper ---------------------------------------------------


class Api:
    def __init__(self, token, sleep=time.sleep):
        self._token = token
        self._sleep = sleep

    def _headers(self):
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "code-gate-verdict",
        }

    def __call__(self, method, path, body=None, attempts=3):
        headers = self._headers()
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        for attempt in range(1, attempts + 1):
            req = urllib.request.Request(
                f"{API_ROOT}/{path}", data=data, method=method, headers=headers
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                if exc.code in (502, 503, 504) and attempt < attempts:
                    self._sleep(2**attempt)
                    continue
                raise

    def download(self, path):
        """A zip behind a redirect: the signed blob URL must not see our token."""

        class _Stop(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        req = urllib.request.Request(f"{API_ROOT}/{path}", headers=self._headers())
        try:
            with urllib.request.build_opener(_Stop).open(req, timeout=60) as resp:
                return read_capped(resp)
        except urllib.error.HTTPError as exc:
            location = (
                exc.headers.get("Location") if exc.code in (301, 302, 303, 307, 308) else None
            )
            if not location or not location.startswith("https://"):
                raise
        bare = urllib.request.Request(location, headers={"User-Agent": "code-gate-verdict"})
        with urllib.request.urlopen(bare, timeout=60) as resp:
            return read_capped(resp)


class UnreadableArtifact(Exception):
    """The artifact bytes are not the whole zip upload-artifact wrote: its verdict is unknown."""


def read_capped(resp, limit=MAX_ARTIFACT_BYTES):
    """The whole body, or raise. `HTTPResponse.read(amt)` returns a short body
    silently when the connection closes early, and a cut-off zip must not read as
    a merely malformed artifact (which a clean run lifts)."""
    body = resp.read(limit + 1)
    if len(body) > limit:
        raise UnreadableArtifact(f"artifact is over {limit} bytes")
    expected = (resp.headers.get("Content-Length") or "").strip()
    if expected.isdigit() and int(expected) != len(body):
        raise UnreadableArtifact(f"artifact body is {len(body)} of {expected} bytes")
    return body


def paged(api, path, key=None, per_page=100, max_pages=10):
    """All items of a paginated list endpoint (up to `max_pages`)."""
    joiner = "&" if "?" in path else "?"
    items = []
    for page in range(1, max_pages + 1):
        data = api("GET", f"{path}{joiner}per_page={per_page}&page={page}")
        batch = data[key] if key else data
        items.extend(batch)
        if len(batch) < per_page:
            break
    return items


def fetch_pr_snapshot(api, repo, number):
    """(pr, files) with a head check: a push landing mid-read is an error, not a mixed snapshot."""
    before = api("GET", f"repos/{repo}/pulls/{number}")
    files = paged(api, f"repos/{repo}/pulls/{number}/files", max_pages=MAX_FILE_PAGES)
    after = api("GET", f"repos/{repo}/pulls/{number}")
    if before["head"]["sha"] != after["head"]["sha"]:
        raise RuntimeError("PR head moved while its files were being read; re-evaluate")
    return after, files


def read_evidence_zip(blob):
    """The artifact object inside the downloaded zip.

    Bytes that are not a readable zip raise `UnreadableArtifact`: upload-artifact
    never writes one, so a cut-off or corrupt download is the cause, and its
    verdict is unknown. A readable zip whose evidence file is absent, oversized or
    not JSON becomes a malformed stub.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            try:
                info = z.getinfo(EVIDENCE_FILE)
            except KeyError:
                return {"status": "malformed"}
            if info.file_size > MAX_EVIDENCE_BYTES:
                return {"status": "malformed"}
            raw = z.read(info)
    except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError) as exc:
        raise UnreadableArtifact(f"artifact zip is unreadable: {exc}") from exc
    try:
        return json.loads(raw)
    except ValueError:
        return {"status": "malformed"}


def load_entries(api, repo, run):
    """A candidate run's entries: one per live evidence artifact (one per attempt).

    State per entry: ok / unreadable (the download failed), or a single missing /
    expired entry when the run has no live evidence artifact. A completed run is
    read whatever its conclusion: an earlier attempt's blocking artifact outlives
    a failed or cancelled re-run.
    """
    bare = {"run": run, "artifact": None, "state": "missing"}
    if run.get("status") != "completed" or run.get("conclusion") == "action_required":
        return [bare]
    arts = [
        a
        for a in paged(api, f"repos/{repo}/actions/runs/{run['id']}/artifacts", key="artifacts")
        if EVIDENCE_ARTIFACT_NAME.fullmatch(a.get("name") or "")
    ]
    live = [a for a in arts if not a.get("expired")]
    if not live:
        return [dict(bare, state="expired" if arts else "missing")]
    entries = []
    for a in live:
        try:
            artifact = read_evidence_zip(
                api.download(f"repos/{repo}/actions/artifacts/{a['id']}/zip")
            )
        except (OSError, http.client.HTTPException, UnreadableArtifact):
            # URLError/HTTPError, a timeout, a reset, a cut-off or corrupt body:
            # whatever the artifact said is unknown, so it must not read as
            # "missing" or "malformed", both of which a clean run lifts.
            entries.append(dict(bare, state="unreadable"))
            continue
        entries.append(dict(bare, artifact=artifact, state="ok"))
    return entries


def gather_entries(api, repo, pr_number, head_sha, default_branch):
    """Candidate runs for this PR/SHA, loaded only when they pass provenance."""
    base = f"repos/{repo}/actions/workflows/{REVIEW_WORKFLOW.rsplit('/', 1)[1]}/runs"
    queries = (
        f"head_sha={head_sha}",
        f"event=workflow_dispatch&branch={urllib.parse.quote(default_branch)}",
    )
    seen = {}
    for q in queries:
        for run in paged(api, f"{base}?{q}", key="workflow_runs", max_pages=MAX_RUN_PAGES):
            seen[run["id"]] = run
    entries = []
    for run in seen.values():
        # The history fact is always the compare's, whatever the payload carried: a
        # dispatch run that would be bound if in history gets it set here, and any
        # other run is never bound through it.
        if (
            run.get("event") == "workflow_dispatch"
            and run_provenance(
                {**run, IN_DEFAULT_HISTORY: True}, pr_number, head_sha, default_branch
            )
            == "bound"
        ):
            run[IN_DEFAULT_HISTORY] = in_default_history(
                api, repo, default_branch, run.get("head_sha") or ""
            )
        # A foreign-base run is loaded too: a blocking artifact it carries still sticks.
        if run_provenance(run, pr_number, head_sha, default_branch):
            entries.extend(load_entries(api, repo, run))
    return entries


def in_default_history(api, repo, default_branch, sha):
    """Is `sha` the default branch's tip or one of its ancestors? Holds after the
    branch moves on; false for a tag or any ref off that history."""
    try:
        cmp = api(
            "GET",
            f"repos/{repo}/compare/{urllib.parse.quote(default_branch)}...{sha}?per_page=1",
        )
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):  # unknown commit, or no common history
            return False
        raise
    return cmp.get("status") in ("behind", "identical")


def shared_head_verdict(pr, open_prs):
    """Red when another open PR into the same base has the same head SHA, else None.

    The check attaches to the commit, so two such PRs would overwrite each other's
    verdict and either could show the other's green (like-current-song#230). The conservative
    combination: the commit is red until one PR moves or closes.
    """
    others = sorted(
        p["number"]
        for p in open_prs
        if p["number"] != pr["number"]
        and p["head"]["sha"] == pr["head"]["sha"]
        and p["base"]["ref"] == pr["base"]["ref"]
    )
    if not others:
        return None
    listed = ", ".join(f"#{n}" for n in others)
    return Verdict(
        False,
        "shared-head",
        f"Open PR(s) {listed} into the same base have this same head commit, and the "
        "check belongs to the commit, so it cannot carry one verdict per PR. Close all "
        "but one, or push a distinct commit to each.",
    )


def post_check(api, repo, head_sha, verdict, app_id):
    """PATCH this App's `verify-verdict` check run for the SHA; create one only if none exists."""
    output = {"title": f"{CHECK_NAME}: {verdict.code}", "summary": verdict.message}
    conclusion = "success" if verdict.green else "failure"
    existing = api(
        "GET", f"repos/{repo}/commits/{head_sha}/check-runs?check_name={CHECK_NAME}&per_page=100"
    )
    mine = [
        c
        for c in existing.get("check_runs", [])
        if str((c.get("app") or {}).get("id")) == str(app_id)
    ]
    if mine:
        api(
            "PATCH",
            f"repos/{repo}/check-runs/{mine[0]['id']}",
            {"status": "completed", "conclusion": conclusion, "output": output},
        )
        return "patched"
    api(
        "POST",
        f"repos/{repo}/check-runs",
        {
            "name": CHECK_NAME,
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": output,
        },
    )
    return "created"


# --- head content for the reviewer ----------------------------------------


def safe_relpath(path):
    """A repo-relative path safe to join under a data directory, or None."""
    if not path or path.startswith("/") or "\\" in path or ":" in path.split("/")[0]:
        return None
    if any(part in ("", ".", "..") for part in path.split("/")):
        return None
    return path


def build_diff(files):
    """A unified diff assembled from the files API's per-file patches."""
    chunks = []
    for f in files:
        patch = f.get("patch")
        if not patch:
            continue
        new, old = f["filename"], f.get("previous_filename") or f["filename"]
        status = f.get("status")
        old_hdr = "/dev/null" if status == "added" else f"a/{old}"
        new_hdr = "/dev/null" if status == "removed" else f"b/{new}"
        chunks.append(f"diff --git a/{old} b/{new}\n--- {old_hdr}\n+++ {new_hdr}\n{patch}\n")
    return "".join(chunks)


def write_head_data(api, head_repo, files, out_dir):
    """Head-side content into `out_dir/<path>.pr` plus `changes.diff` and `changes.txt`.

    Nothing is written under a name the harness would treat as config (`.claude/...`):
    the `.pr` suffix and the foreign root keep PR-controlled files inert.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    total = 0
    fetched = 0
    for f in files:
        name = f["filename"]
        status = f.get("status")
        note = status
        rel = safe_relpath(name)
        if status == "removed":
            pass
        elif rel is None:
            note = f"{status} (unsafe path, not fetched)"
        elif name.endswith(_IMAGE_EXT):
            note = f"{status} (image, not fetched)"
        elif fetched >= MAX_HEAD_FILES or not f.get("sha"):
            note = f"{status} (not fetched: limit)"
        else:
            blob = api("GET", f"repos/{head_repo}/git/blobs/{f['sha']}")
            size = blob.get("size", 0)
            if size > MAX_HEAD_FILE_BYTES or total + size > MAX_HEAD_TOTAL_BYTES:
                note = f"{status} (not fetched: {size} bytes)"
            else:
                target = os.path.join(out_dir, *rel.split("/")) + ".pr"
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(base64.b64decode(blob["content"]))
                total += size
                fetched += 1
        prev = f.get("previous_filename")
        manifest.append(f"{note}\t{name}" + (f"\t(was {prev})" if prev else ""))
    with open(os.path.join(out_dir, "changes.diff"), "w", encoding="utf-8") as fh:
        fh.write(build_diff(files))
    with open(os.path.join(out_dir, "changes.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest) + "\n")


# --- CLI -------------------------------------------------------------------


def _output(name, value):
    value = str(value).replace("\r", " ").replace("\n", " ")
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    print(f"{name}={value}")


def _summary(text):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def cmd_prepare(args):
    """Review job: snapshot the PR, decide skip/review, stage head content."""
    repo = os.environ["GITHUB_REPOSITORY"]
    api = Api(os.environ["GH_TOKEN"])
    pr, files = fetch_pr_snapshot(api, repo, int(os.environ["PR_NUMBER"]))
    head_sha = pr["head"]["sha"]
    expected = os.environ.get("EXPECTED_HEAD_SHA", "")
    if expected and expected != head_sha:
        print(
            f"PR head is {head_sha}, this run was asked for {expected}: stale, aborting",
            file=sys.stderr,
        )
        return 1
    reason = review_skip_reason(pr, files, os.environ["EVENT_NAME"])
    _output("head_sha", head_sha)
    _output("base_ref", pr["base"]["ref"])
    _output("skip_reason", reason)
    if not reason:
        write_head_data(api, pr["head"]["repo"]["full_name"], files, args.head_dir)
    return 0


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def cmd_validate(args):
    errors = validate_findings(_read_json(args.findings))
    for e in errors:
        print(f"invalid findings: {e}", file=sys.stderr)
    return 1 if errors else 0


def cmd_check_tree(args):
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = check_clean_tree(out, args.findings)
    for path in dirty:
        print(f"workspace modified during review: {path}", file=sys.stderr)
    return 1 if dirty else 0


def _write_evidence(obj, out):
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def cmd_stamp(args):
    findings = _read_json(args.findings)
    if validate_findings(findings):
        print("refusing to stamp invalid findings", file=sys.stderr)
        return 1
    _write_evidence(
        build_evidence(findings, os.environ["HEAD_SHA"], os.environ["BASE_REF"]), args.out
    )
    return 0


def cmd_skip(args):
    _write_evidence(
        build_skipped_evidence(os.environ["HEAD_SHA"], os.environ["BASE_REF"], args.reason),
        args.out,
    )
    return 0


def cmd_resolve(_args):
    """Verdict workflow: which PR and SHA does this event concern?"""
    event = os.environ["EVENT_NAME"]
    if event == "pull_request_target":
        target = (int(os.environ["PR_NUMBER"]), os.environ["HEAD_SHA"])
    else:
        target = resolve_run_target(
            os.environ["RUN_EVENT"],
            os.environ.get("RUN_TITLE", ""),
            os.environ.get("RUN_HEAD_SHA", ""),
            json.loads(os.environ.get("RUN_PULL_REQUESTS") or "[]"),
            os.environ["DEFAULT_BRANCH"],
        )
    if target is None:
        print("event is not bound to a PR; nothing to evaluate")
        _output("pr_number", "")
        _output("head_sha", "")
        return 0
    _output("pr_number", target[0])
    _output("head_sha", target[1])
    return 0


def cmd_verdict(_args):
    repo = os.environ["GITHUB_REPOSITORY"]
    head_sha = os.environ["HEAD_SHA"]
    read = Api(os.environ["GH_TOKEN"])
    number = int(os.environ["PR_NUMBER"])
    failure = None
    try:
        pr, files = fetch_pr_snapshot(read, repo, number)
        if pr["head"]["sha"] != head_sha:
            print("PR head moved past the event's SHA; the newer SHA has its own evaluation")
            return 0
        default_branch = read("GET", f"repos/{repo}")["default_branch"]
        if pr["base"]["ref"] != default_branch:
            # The check belongs to the commit: a verdict for this PR would land on a
            # PR into the default branch that shares the head (like-current-song#230).
            print(f"PR is into {pr['base']['ref']}, not {default_branch}; no verdict to post")
            return 0
        open_prs = paged(
            read,
            f"repos/{repo}/pulls?state=open&base={urllib.parse.quote(default_branch)}",
            max_pages=MAX_PR_PAGES,
        )
        if len(open_prs) >= 100 * MAX_PR_PAGES:
            raise RuntimeError("too many open PRs to check for a shared head")
        verdict = shared_head_verdict(pr, open_prs) or evaluate_pr(
            pr, files, gather_entries(read, repo, number, head_sha, default_branch), default_branch
        )
    except Exception as exc:
        # Fail closed: a crash must not leave an earlier green check standing for
        # this SHA (attempt 1 clean, the re-run blocking, its listing reset).
        failure = exc
        verdict = Verdict(
            False,
            "verdict-error",
            f"The verdict could not be computed ({type(exc).__name__}: {str(exc)[:200]}). "
            "Re-run the verdict job.",
        )
    how = post_check(
        Api(os.environ["GATE_TOKEN"]), repo, head_sha, verdict, os.environ["GATE_APP_ID"]
    )
    color = "green" if verdict.green else "red"
    _summary(f"### {CHECK_NAME}: {color} — `{verdict.code}`\n\n{verdict.message}")
    print(f"{CHECK_NAME} {how}: green={verdict.green} code={verdict.code}")
    if failure is not None:
        raise failure
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--head-dir", default=".pr-head")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("validate")
    p.add_argument("--findings", required=True)
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("check-tree")
    p.add_argument("--findings", required=True)
    p.set_defaults(fn=cmd_check_tree)
    p = sub.add_parser("stamp")
    p.add_argument("--findings", required=True)
    p.add_argument("--out", default=EVIDENCE_FILE)
    p.set_defaults(fn=cmd_stamp)
    p = sub.add_parser("skip")
    p.add_argument("--reason", required=True)
    p.add_argument("--out", default=EVIDENCE_FILE)
    p.set_defaults(fn=cmd_skip)
    sub.add_parser("resolve").set_defaults(fn=cmd_resolve)
    sub.add_parser("verdict").set_defaults(fn=cmd_verdict)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
