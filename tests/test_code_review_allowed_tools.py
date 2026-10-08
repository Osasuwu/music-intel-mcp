"""Drift guard for the code-review action's `--allowed-tools` / `--disallowed-tools` lists.

The reviewer runs HEADLESS (`anthropics/claude-code-action@v1`): a tool absent
from `--allowed-tools` is DENIED outright, there is no human to approve it. Both
lists are the contract with the harness, so they are pinned here: the allowlist
as an exact closed set of read verbs, the mutating verbs denied, and the one
write grant, `Edit(./.review/findings.json)`.

The patterns steer the reviewer; they are not a boundary (the harness matches
command text, so a quoted flag slips past a deny). What bounds a reviewer talked
into a write is the read-only job token it runs `gh` on, pinned here too.

The reviewer no longer posts a PR comment; its only output is the findings file.
`Edit(path)` is the one path-scoped write rule the harness consults (a
`Write(path)` rule is never matched, so an unscoped `Write` could write anywhere).
`gh pr comment` is denied so the review has exactly one output channel.
Design: Osasuwu/jarvis#1964, ported in #226.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / ".github" / "workflows").is_dir()
)
LIVE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "code-review.yml"

FINDINGS_PATH = ".review/findings.json"
FINDINGS_GRANT = f"Edit(./{FINDINGS_PATH})"

# The exact allowlist. One entry per read verb: a wholesale `Bash(git:*)` admits
# `git config core.fsmonitor=<cmd>`, `git -c alias.x='!cmd'` and `git grep -O<cmd>`,
# all of which execute programs, and `sort -o`/`uniq in out` write files. A
# "syntax check" is not read-only either: `node --check -r <file>` runs the preload
# module, so `node --check` and `bash -n` stay out. Adding a grant means adding it
# here, which is the review point.
EXPECTED_ALLOWED = frozenset(
    {
        # Native file-reading tools; the reviewer prompt steers it to these.
        "Read",
        "Grep",
        "Glob",
        # The findings file is the reviewer's only output.
        FINDINGS_GRANT,
        "Bash(git log:*)",
        "Bash(git show:*)",
        "Bash(git diff:*)",
        "Bash(git blame:*)",
        "Bash(git status:*)",
        "Bash(git rev-parse:*)",
        "Bash(git ls-files:*)",
        "Bash(git merge-base:*)",
        "Bash(gh pr view:*)",
        "Bash(gh pr diff:*)",
        "Bash(gh pr list:*)",
        "Bash(gh pr checks:*)",
        "Bash(gh issue view:*)",
        "Bash(gh issue list:*)",
        "Bash(gh search:*)",
        "Bash(gh label list:*)",
        # `gh api` stays on narrow path grants; the read-only token is what stops
        # its `-X`/`-f` mutation flags.
        "Bash(gh api repos/*/commits/*:*)",
        "Bash(gh api repos/*/compare/*:*)",
        "Bash(wc:*)",
        "Bash(head:*)",
        "Bash(tail:*)",
        "Bash(cat:*)",
        "Bash(cut:*)",
        "Bash(nl:*)",
        "Bash(tr:*)",
        # Headless permission matching splits compound commands on ; | && and
        # newlines and checks each part, so an un-allowlisted `echo` prefix
        # denies the whole command.
        "Bash(echo:*)",
        "Bash(python -m py_compile:*)",
        "Bash(python3 -m py_compile:*)",
    }
)

# Mutating verbs stay denied as a second line behind the exact allowlist, so a
# future wildcard grant does not silently reopen them.
REQUIRED_DISALLOWED = (
    "Bash(gh pr merge:*)",
    "Bash(gh pr close:*)",
    "Bash(gh pr edit:*)",
    "Bash(gh pr reopen:*)",
    "Bash(gh pr review:*)",
    "Bash(gh pr ready:*)",
    "Bash(gh pr create:*)",
    "Bash(gh pr lock:*)",
    "Bash(gh pr unlock:*)",
    # The PR comment is for humans and not produced by this job.
    "Bash(gh pr comment:*)",
    "Bash(gh issue create:*)",
    "Bash(gh issue edit:*)",
    "Bash(gh issue close:*)",
    "Bash(gh issue reopen:*)",
    "Bash(gh issue delete:*)",
    "Bash(gh issue lock:*)",
    "Bash(gh issue unlock:*)",
    "Bash(gh issue pin:*)",
    "Bash(gh issue unpin:*)",
    "Bash(gh issue transfer:*)",
    "Bash(gh issue comment:*)",
    "Bash(gh label create:*)",
    "Bash(gh label edit:*)",
    "Bash(gh label delete:*)",
    "Bash(git push:*)",
    "Bash(git commit:*)",
    "Bash(git merge:*)",
    "Bash(git reset:*)",
    "Bash(git rebase:*)",
    "Bash(git cherry-pick:*)",
    "Bash(git stash:*)",
    "Bash(git clean:*)",
    "Bash(git rm:*)",
    "Bash(git mv:*)",
    "Bash(git apply:*)",
    "Bash(git am:*)",
    "Bash(git checkout:*)",
    "Bash(git switch:*)",
    "Bash(git restore:*)",
    # `git log/show/diff/blame --output=<file>` writes any path on the runner.
    "Bash(git *--ou*)",
)

_ALLOWED_TOOLS_RE = re.compile(r'--allowed-tools\s+"([^"]*)"')
_DISALLOWED_TOOLS_RE = re.compile(r'--disallowed-tools\s+((?:"[^"]*"\s*)+)')


def _allowed_tools_blocks(path: Path) -> list[str]:
    """Every `--allowed-tools "..."` string in the file."""
    text = path.read_text(encoding="utf-8")
    blocks = _ALLOWED_TOOLS_RE.findall(text)
    assert blocks, f"no --allowed-tools line found in {path}"
    return blocks


def _disallowed_tools_blocks(path: Path) -> list[list[str]]:
    """Every `--disallowed-tools "a" "b" ...` entry list in the file."""
    text = path.read_text(encoding="utf-8")
    raw_blocks = _DISALLOWED_TOOLS_RE.findall(text)
    assert raw_blocks, f"no --disallowed-tools line found in {path}"
    return [re.findall(r'"([^"]*)"', raw) for raw in raw_blocks]


@pytest.mark.parametrize("path", [LIVE_WORKFLOW], ids=["live"])
def test_allowlist_is_exactly_the_read_set(path: Path) -> None:
    """A missing grant is DENIED headless; an extra one widens what an injected
    reviewer can run. Both directions go red."""
    for block in _allowed_tools_blocks(path):
        got = set(block.split(","))
        assert got == EXPECTED_ALLOWED, (
            f"{path.name}: allowlist drifted. Extra: {sorted(got - EXPECTED_ALLOWED)}; "
            f"missing: {sorted(EXPECTED_ALLOWED - got)}"
        )


def _reviewer_steps() -> list[dict]:
    spec = yaml.safe_load(LIVE_WORKFLOW.read_text(encoding="utf-8"))
    steps = [
        s
        for s in spec["jobs"]["review"]["steps"]
        if str(s.get("uses", "")).startswith("anthropics/claude-code-action@")
    ]
    assert steps, "no claude-code-action step found"
    return steps


def test_reviewer_runs_gh_on_the_read_only_job_token() -> None:
    """Without `github_token` the action mints a write-scoped Claude App token
    over OIDC and hands it to the reviewer's `gh`."""
    for step in _reviewer_steps():
        assert step["with"].get("github_token") == "${{ secrets.GITHUB_TOKEN }}", step["name"]


def test_review_job_token_is_read_only_and_has_no_oidc() -> None:
    spec = yaml.safe_load(LIVE_WORKFLOW.read_text(encoding="utf-8"))
    perms = spec["jobs"]["review"]["permissions"]
    assert perms == {"contents": "read", "pull-requests": "read"}


def test_reviewer_subprocess_env_is_scrubbed() -> None:
    spec = yaml.safe_load(LIVE_WORKFLOW.read_text(encoding="utf-8"))
    assert spec["jobs"]["review"]["env"]["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"


@pytest.mark.parametrize("path", [LIVE_WORKFLOW], ids=["live"])
def test_mutating_verbs_disallowed(path: Path) -> None:
    """Every mutating verb stays disallowed behind the exact allowlist."""
    for block in _disallowed_tools_blocks(path):
        for tool in REQUIRED_DISALLOWED:
            assert tool in block, (
                f"{path.name}: --disallowed-tools missing {tool!r}. Disallowed-tools was: {block}"
            )


@pytest.mark.parametrize("path", [LIVE_WORKFLOW], ids=["live"])
def test_only_write_grant_is_the_findings_file(path: Path) -> None:
    """An unscoped `Write` or bare `Edit` lets the reviewer — an LLM
    reading attacker-controlled text — write anywhere in the workspace. The one
    write grant is `Edit(./.review/findings.json)`."""
    for block in _allowed_tools_blocks(path):
        entries = block.split(",")
        writers = [e for e in entries if e.split("(")[0] in ("Write", "Edit", "MultiEdit")]
        assert writers == [FINDINGS_GRANT], (
            f"{path.name}: write-capable grants must be exactly [{FINDINGS_GRANT!r}], got {writers}"
        )


def test_findings_path_in_job_env_matches_the_edit_grant() -> None:
    """The job seeds, validates and uploads $FINDINGS; the reviewer may edit only
    the granted path. Two artifacts that must agree."""
    spec = yaml.safe_load(LIVE_WORKFLOW.read_text(encoding="utf-8"))
    assert spec["jobs"]["review"]["env"]["FINDINGS"] == FINDINGS_PATH
