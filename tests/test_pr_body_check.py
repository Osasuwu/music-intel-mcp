"""Behaviour of `require-linked-issue` in .github/workflows/pr-body-check.yml (#243).

Each test runs the job's real github-script body under node (tests/run_github_script.js)
with recording stubs for `core` and `github.rest.issues.get`, then asserts what the run
reported: the failure messages, the bypass notices and the issues it looked up. Nothing here
re-implements the rule in Python. Ported from Osasuwu/jarvis#2052 (jarvis#1967).

The tests need `node`; GitHub's ubuntu-latest runner ships it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = _ROOT / ".github" / "workflows" / "pr-body-check.yml"
HARNESS = Path(__file__).resolve().parent / "run_github_script.js"
REPO = {"owner": "Osasuwu", "repo": "music-intel-mcp"}
SELF = "Osasuwu/music-intel-mcp"

NO_LINK = (
    'PR body must include a linked issue reference (e.g. "Closes #123" to auto-close, '
    '"Closes owner/repo#123" or "Closes https://github.com/owner/repo/issues/123" for an '
    'issue in another repo, or "Refs #123" to link a partial-work PR without closing), a '
    "[no-issue] line for trivial fix-inline drive-bys (#428), a refactor: title prefix, or "
    "the priority:critical label for hotfixes."
)
NOT_ACCESSIBLE = (
    "does not exist or is not accessible (the workflow token reads public repos only, "
    "so an issue in a private repo cannot be verified)."
)
NOTICE_NO_ISSUE = "require-linked-issue bypassed: [no-issue] — fix-inline drive-by per #428."
BLOCKED = {"failed": [NO_LINK], "notices": [], "lookups": []}


def _script() -> str:
    doc = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    return next(
        step["with"]["script"]
        for step in doc["jobs"]["require-linked-issue"]["steps"]
        if "script" in (step.get("with") or {})
    )


def run(
    body: str = "",
    title: str = "feat: x",
    labels: tuple[str, ...] = (),
    author: str = "Osasuwu",
    issues: tuple[str, ...] = (f"{SELF}#42",),
) -> dict:
    node = shutil.which("node")
    assert node, "node is not on PATH; the test job runs on ubuntu-latest, which ships it"
    payload = {
        "pull_request": {
            "body": body,
            "title": title,
            "labels": [{"name": n} for n in labels],
            "user": {"login": author},
        }
    }
    proc = subprocess.run(
        [node, str(HARNESS)],
        input=json.dumps(
            {"script": _script(), "repo": REPO, "payload": payload, "issues": list(issues)}
        ),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# --- linkage -------------------------------------------------------------------------------


def test_every_github_closing_keyword_and_refs_links_the_issue():
    """GitHub's nine closing keywords, in any case and with `:`, plus refs/references."""
    keywords = [
        "close", "closes", "closed", "fix", "fixes", "fixed", "resolve", "resolves", "resolved",
        "ref", "refs", "reference", "references",
    ]  # fmt: skip
    bodies = [f"{k} #42" for k in keywords]
    bodies += ["CLOSES #42", "Fixed: #42", "Resolves:#42", "Some prose.\n\nRefs #42\n"]
    expected = {"failed": [], "notices": [], "lookups": [f"{SELF}#42"]}
    wrong = {b: r for b in bodies if (r := run(b)) != expected}
    assert wrong == {}


def test_cross_repo_and_url_references_look_up_the_named_repo():
    """#243 AC1: `owner/repo#N` and the issue URL link, and the existence check asks that repo."""
    for body in ("Closes Other/proj#7", "Fixes https://github.com/Other/proj/issues/7"):
        result = run(body, issues=("Other/proj#7",))
        assert result == {"failed": [], "notices": [], "lookups": ["Other/proj#7"]}, body


def test_cross_repo_reference_to_a_missing_issue_fails_naming_the_full_reference():
    """#243 AC2: the failure names `owner/repo#N`, and says what a private repo means."""
    assert run("Closes Other/proj#7", issues=(f"{SELF}#7",)) == {
        "failed": [f"Linked issue Other/proj#7 {NOT_ACCESSIBLE}"],
        "notices": [],
        "lookups": ["Other/proj#7"],
    }


def test_same_number_in_another_repo_is_not_the_local_issue():
    """#243 AC2/AC3: `Other/proj#42` must be looked up in Other/proj, never as local #42."""
    result = run("Closes Other/proj#42", issues=(f"{SELF}#42",))
    assert result["lookups"] == ["Other/proj#42"]
    assert result["failed"] == [f"Linked issue Other/proj#42 {NOT_ACCESSIBLE}"]


def test_repeated_reference_is_looked_up_once():
    result = run(f"Closes #42\n\nRefs #42, see also fixes {SELF}#42")
    assert result == {"failed": [], "notices": [], "lookups": [f"{SELF}#42"]}


def test_linked_issue_that_does_not_exist_fails():
    """#243 AC3: a missing same-repo issue still fails."""
    assert run("Closes #999") == {
        "failed": [f"Linked issue {SELF}#999 {NOT_ACCESSIBLE}"],
        "notices": [],
        "lookups": [f"{SELF}#999"],
    }


def test_missing_reference_failure_states_the_accepted_forms():
    """#243 AC5: the no-link message shows the cross-repo forms are allowed."""
    assert run("nothing linked here") == BLOCKED


def test_keyword_inside_a_longer_word_does_not_link():
    """`prefixes #12` / `suffixes #12` are not `fixes #12`; prose verbs are not keywords."""
    bodies = [
        "prefixes #12", "suffixes #12", "discloses #12", "unresolved #12",
        "This PR refers to #42 for background.", "This is close to done.", "",
    ]  # fmt: skip
    wrong = {b: r for b in bodies if (r := run(b, issues=(f"{SELF}#12",))) != BLOCKED}
    assert wrong == {}


# --- bypasses (#243 AC4: each one passes and names itself in a notice) ----------------------


def test_bot_author_bypasses_with_notice():
    assert run("bump foo", author="dependabot[bot]") == {
        "failed": [],
        "notices": [
            "require-linked-issue bypassed: bot author — "
            "dependabot[bot] cannot author linked issues."
        ],
        "lookups": [],
    }


def test_priority_critical_label_bypasses_with_notice():
    assert run("urgent prod fix", labels=("priority:critical",)) == {
        "failed": [],
        "notices": ["require-linked-issue bypassed: priority:critical — hotfix label present."],
        "lookups": [],
    }


def test_other_labels_do_not_bypass():
    assert run("trivial change", labels=("priority:medium", "documentation")) == BLOCKED


def test_no_issue_on_its_own_body_line_bypasses_with_notice():
    """The directive on a line of its own, any case, LF or CRLF."""
    bodies = ["Drive-by docs fix.\n\n[no-issue]\n", "Drive-by.\r\n\r\n  [NO-ISSUE]  \r\n"]
    expected = {"failed": [], "notices": [NOTICE_NO_ISSUE], "lookups": []}
    wrong = {b: r for b in bodies if (r := run(b)) != expected}
    assert wrong == {}


def test_no_issue_as_last_title_token_bypasses_with_notice():
    """The squash commit title carries the marker last, as the commit-msg hook expects."""
    assert run("", title="fix(tests): isolate env [no-issue]") == {
        "failed": [],
        "notices": [NOTICE_NO_ISSUE],
        "lookups": [],
    }


def test_no_issue_inside_prose_does_not_bypass():
    """The marker quoted or explained is not the directive."""
    cases = [
        ("This PR explains the [no-issue] marker.", "feat: x"),
        ("Use `[no-issue]` for drive-bys.", "feat: x"),
        ("[no-issue] inline doc fix", "feat: x"),
        ("", "docs: explain the [no-issue] escape"),
        ("There is no issue with this change.", "feat: x"),
    ]
    wrong = {c: r for c in cases if (r := run(c[0], title=c[1])) != BLOCKED}
    assert wrong == {}


def test_refactor_title_prefix_bypasses_with_notice():
    expected = {
        "failed": [],
        "notices": [
            "require-linked-issue bypassed: refactor: title prefix — "
            "structural cleanup without a parent issue."
        ],
        "lookups": [],
    }
    titles = ["refactor: split foo", "Refactor(security): narrow", "REFACTOR(api): x"]
    wrong = {t: r for t in titles if (r := run("", title=t)) != expected}
    assert wrong == {}


def test_refactor_outside_the_title_prefix_does_not_bypass():
    titles = ["Refactoring the auth module", "feat: include refactor: in description"]
    wrong = {t: r for t in titles if (r := run("body", title=t)) != BLOCKED}
    assert wrong == {}
