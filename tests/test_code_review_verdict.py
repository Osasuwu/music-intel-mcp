"""Zero-comments branch of `verify-verdict` in .github/workflows/code-review.yml (#224).

The step's shell script is lifted out of the workflow verbatim and executed against an offline
`gh` stand-in, so these tests exercise the gate that actually runs in CI rather than a copy of it.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_root = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "workflows").is_dir())
_workflow = yaml.safe_load(
    (_root / ".github" / "workflows" / "code-review.yml").read_text(encoding="utf-8")
)
_job = _workflow["jobs"]["verify-verdict"]
_step = next(s for s in _job["steps"] if s.get("name") == "Verify review verdict")

HEAD_SHA = "a" * 40
RUN_ID = "1000"

# Offline stand-in for `gh`: answers exactly the five calls the zero-comments branch makes and
# exits 97 on anything else, so a harness gap can never be mistaken for the gate's own `exit 1`.
_GH_STUB = """\
gh() {
  case "$*" in
    "pr view "*) cat "$GH_STUB_DIR/pr.json" ;;
    "api repos/"*"/commits/"*" -q .commit.committer.date") echo "2026-09-30T10:00:00Z" ;;
    "api repos/"*"/issues/"*"/comments --paginate") cat "$GH_STUB_DIR/comments.json" ;;
    "api repos/"*"/pulls/"*"/commits --paginate") cat "$GH_STUB_DIR/pr_commits.json" ;;
    "api repos/"*"/actions/workflows/code-review.yml/runs?per_page=100 --paginate")
      cat "$GH_STUB_DIR/runs.json" ;;
    *) echo "gh stub: unexpected call: $*" >&2; exit 97 ;;
  esac
}
"""
# jq.exe translates "\n" to "\r\n" on stdout unless told otherwise, which breaks `[ "$x" -eq 0 ]`.
_JQ_WINDOWS_SHIM = 'jq() { command jq --binary "$@"; }\n'


def _bash():
    bash = shutil.which("bash")
    if os.name != "nt" or (bash and "system32" not in bash.lower()):
        return bash
    # `bash` on a Windows PATH is often System32's WSL launcher; use the one Git ships, which
    # sits at <Git>/bin/bash.exe while git.exe is <Git>/cmd or <Git>/mingw64/bin.
    git = shutil.which("git")
    for parent in Path(git).parents if git else ():
        candidate = parent / "bin" / "bash.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _run_verdict(tmp_path, *, attempt_1, attempt_2, pr_state="OPEN", exec_ran="false"):
    bash = _bash()
    if bash is None or shutil.which("jq") is None:
        if os.environ.get("CI"):
            pytest.fail("bash and jq are required to run the verdict script in CI")
        pytest.skip("bash/jq not available locally")

    fixtures = {
        "pr.json": {"headRefOid": HEAD_SHA, "state": pr_state},
        "comments.json": [],
        "pr_commits.json": [{"sha": HEAD_SHA}],
        # A PR's first-ever review run: the only run over its commits is the one executing.
        "runs.json": {
            "workflow_runs": [
                {
                    "id": int(RUN_ID),
                    "head_sha": HEAD_SHA,
                    "status": "in_progress",
                    "conclusion": None,
                }
            ]
        },
    }
    for name, payload in fixtures.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    script = tmp_path / "verdict.sh"
    prefix = _GH_STUB + (_JQ_WINDOWS_SHIM if os.name == "nt" else "")
    script.write_text(prefix + _step["run"], encoding="utf-8", newline="\n")

    env = {
        **os.environ,
        "GH_STUB_DIR": tmp_path.as_posix(),
        "PR": "1",
        "REPO": "owner/repo",
        "GITHUB_RUN_ID": RUN_ID,
        "EXEC_RAN_1": exec_ran,
        "EXEC_RAN_2": "false",
        "RUN_START_1": "",
        "RUN_START_2": "",
        "ATTEMPT_RESULT_1": attempt_1,
        "ATTEMPT_RESULT_2": attempt_2,
    }
    # Same invocation GitHub uses for a `run:` step on a Linux runner.
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", script.as_posix()],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_script_has_no_workflow_expressions():
    # The harness runs the script as plain bash; an inline `${{ }}` would make it diverge from CI.
    assert "${{" not in _step["run"]


def test_verdict_job_runs_after_both_attempts_whatever_their_result():
    assert _job["needs"] == ["attempt-1", "attempt-2"]
    assert _job["if"].startswith("always()")


def test_attempt_results_are_wired_into_the_verdict_step():
    assert _step["env"]["ATTEMPT_RESULT_1"] == "${{ needs.attempt-1.result }}"
    assert _step["env"]["ATTEMPT_RESULT_2"] == "${{ needs.attempt-2.result }}"


@pytest.mark.parametrize(
    ("attempt_1", "attempt_2"),
    [
        ("failure", "failure"),  # the #224 repro: both attempts die before posting
        ("failure", "success"),
        ("failure", "cancelled"),
        ("cancelled", "skipped"),  # run cancelled mid-review; attempt-2 never starts
    ],
)
def test_unreviewed_pr_fails_closed_when_own_attempts_did_not_complete(
    tmp_path, attempt_1, attempt_2
):
    result = _run_verdict(tmp_path, attempt_1=attempt_1, attempt_2=attempt_2)

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"::error::attempt-1={attempt_1} attempt-2={attempt_2}" in result.stdout
    assert "legitimately skipped" not in result.stdout


@pytest.mark.parametrize(
    ("attempt_1", "attempt_2"),
    [
        ("success", "skipped"),  # plugin declined (draft / not eligible / no substantive code)
        ("skipped", "skipped"),  # attempt-1's own `if` excluded the PR (fork, dependabot)
    ],
)
def test_genuine_skip_still_passes(tmp_path, attempt_1, attempt_2):
    result = _run_verdict(tmp_path, attempt_1=attempt_1, attempt_2=attempt_2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "legitimately skipped" in result.stdout


def test_post_factum_run_on_a_merged_pr_still_passes(tmp_path):
    result = _run_verdict(tmp_path, attempt_1="failure", attempt_2="failure", pr_state="MERGED")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "nothing left to gate" in result.stdout
