"""Shape of the two code-gate workflows (#226, ported from jarvis#1964).

code-gate-verdict.yml is the trusted half: it runs from the default branch and
decides `verify-verdict`. Its safety is structural — it must never run PR code, never
hold more than read permissions, and never let a PR-controlled string reach a
shell. code-review.yml produces the evidence; its retry and upload settings are
what make a missing or stale review fail closed. Each test walks the parsed YAML,
not the file text.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = (
    next(p for p in Path(__file__).resolve().parents if (p / ".github" / "workflows").is_dir())
    / ".github"
    / "workflows"
)
REVIEW = yaml.safe_load((WORKFLOWS / "code-review.yml").read_text(encoding="utf-8"))
VERDICT = yaml.safe_load((WORKFLOWS / "code-gate-verdict.yml").read_text(encoding="utf-8"))

# PyYAML parses the bare key `on` as the boolean True.
ON = True


def _steps(spec: dict) -> list[dict]:
    return [step for job in spec["jobs"].values() for step in job["steps"]]


def _run_bodies(spec: dict) -> list[str]:
    return [step["run"] for step in _steps(spec) if "run" in step]


# --- verdict workflow -----------------------------------------------------


def test_verdict_permissions_are_exactly_three_reads():
    assert VERDICT["permissions"] == {
        "contents": "read",
        "actions": "read",
        "pull-requests": "read",
    }
    for name, job in VERDICT["jobs"].items():
        assert "permissions" not in job, f"job {name} must not widen the workflow permissions"


def test_verdict_never_checks_out_the_pr_head():
    checkouts = [
        s for s in _steps(VERDICT) if str(s.get("uses", "")).startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2  # resolve + verdict, both base-pinned
    for step in checkouts:
        assert step["with"]["ref"] == "${{ github.event.repository.default_branch }}"
        assert step["with"]["sparse-checkout"] == ".github/scripts"
        assert step["with"]["persist-credentials"] is False


def test_verdict_triggers_pull_request_target_and_review_workflow_run():
    triggers = VERDICT[ON]
    assert set(triggers) == {"pull_request_target", "workflow_run"}
    assert triggers["workflow_run"] == {"workflows": ["Code Review"], "types": ["completed"]}
    assert triggers["pull_request_target"]["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "edited",
    ]


def test_verdict_concurrency_is_per_pr_and_sha_without_cancel():
    expected_group = (
        "code-gate-${{ needs.resolve.outputs.pr_number }}-${{ needs.resolve.outputs.head_sha }}"
    )
    assert VERDICT["jobs"]["verdict"]["concurrency"] == {
        "group": expected_group,
        "cancel-in-progress": False,
    }


def test_edited_proceeds_only_on_a_base_change_in_both_workflows():
    guard = "github.event.action != 'edited' || github.event.changes.base"
    assert guard in VERDICT["jobs"]["resolve"]["if"]
    assert guard in REVIEW["jobs"]["review"]["if"]


def test_no_run_body_interpolates_pr_controlled_context():
    # PR title, branch names, head repo and dispatch inputs are attacker-chosen;
    # they reach shell only through `env:`.
    pattern = re.compile(
        r"\$\{\{[^}]*(github\.event\.pull_request\.|github\.event\.workflow_run\.|inputs\.|github\.head_ref)"
    )
    for spec in (VERDICT, REVIEW):
        bodies = _run_bodies(spec)
        assert bodies, "expected run bodies to scan"
        for body in bodies:
            assert pattern.search(body) is None, body


def test_only_the_verdict_step_holds_the_app_token():
    holders = [
        s["name"]
        for s in _steps(VERDICT)
        if "GATE_TOKEN" in (s.get("env") or {})
        or "secrets.GATE_APP_PRIVATE_KEY" in str(s.get("with") or {})
    ]
    assert holders == ["Mint osasuwu-ci installation token", "Decide and post verify-verdict"]
    assert VERDICT["jobs"]["verdict"]["environment"] == "code-gate-verdict"


# --- review workflow ------------------------------------------------------


def test_review_concurrency_cancels_a_superseded_review():
    # Job-level: workflow-level concurrency is evaluated before the job `if`, so a
    # skipped `edited` run would cancel the real review and leave no evidence.
    assert "concurrency" not in REVIEW
    assert REVIEW["jobs"]["review"]["concurrency"] == {
        "group": "code-review-${{ github.event.pull_request.number || inputs.pr_number }}",
        "cancel-in-progress": True,
    }


def test_review_run_name_embeds_pr_and_head_sha_for_dispatch_provenance():
    run_name = REVIEW["run-name"]
    assert "format('Code review PR #{0} @ {1}', inputs.pr_number, inputs.head_sha)" in run_name


def test_dispatch_requires_pr_number_and_head_sha():
    inputs = REVIEW[ON]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"pr_number", "head_sha"}
    assert all(spec["required"] is True for spec in inputs.values())


def test_review_checks_out_base_only_with_no_history():
    (checkout,) = [
        s for s in _steps(REVIEW) if str(s.get("uses", "")).startswith("actions/checkout@")
    ]
    assert checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha || github.event.repository.default_branch }}"
    }


def _upload_steps() -> list[dict]:
    return [
        s for s in _steps(REVIEW) if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    ]


EVIDENCE_UPLOAD_NAME = "review-evidence-${{ github.run_attempt }}"


def test_evidence_artifact_is_retained_90_days_one_per_attempt():
    """A re-run attempt uploads its own artifact and cannot overwrite an earlier
    attempt's, so blocking evidence survives a re-run."""
    (evidence,) = [s for s in _upload_steps() if s["with"]["name"] == EVIDENCE_UPLOAD_NAME]
    assert evidence["with"]["retention-days"] == 90
    assert evidence["with"]["overwrite"] is False
    assert evidence["with"]["if-no-files-found"] == "error"
    assert "if" not in evidence, "every run uploads evidence, including skipped ones"


@pytest.mark.parametrize("attempt", ["1", "2", "17"])
def test_uploaded_evidence_name_is_what_the_verdict_reads(attempt):
    """The verdict only reads artifacts whose name fullmatches its pattern; an
    upload name it does not match would read as "missing" for every run. The step
    is found by what it uploads, not by the name under test."""
    (evidence,) = [s for s in _upload_steps() if s["with"]["path"] == "review-evidence.json"]
    name = evidence["with"]["name"].replace("${{ github.run_attempt }}", attempt)
    assert _gate().EVIDENCE_ARTIFACT_NAME.fullmatch(name)


def _gate():
    spec = importlib.util.spec_from_file_location(
        "code_gate_verdict", WORKFLOWS.parent / "scripts" / "code_gate_verdict.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_second_reviewer_attempt_runs_only_when_the_first_findings_are_invalid():
    by_name = {s["name"]: s for s in _steps(REVIEW) if "name" in s}
    first = by_name["Run code review (attempt 1)"]
    second = by_name["Run code review (attempt 2)"]
    assert first["continue-on-error"] is True
    assert by_name["Validate findings (attempt 1)"]["continue-on-error"] is True
    assert second["if"] == "steps.v1.outcome == 'failure'"
    # The retry starts from the sentinel again, not from attempt 1's invalid file.
    names = [s.get("name") for s in _steps(REVIEW)]
    reseed = by_name["Reseed findings file for the retry"]
    assert reseed["if"] == "steps.v1.outcome == 'failure'"
    assert names.index(reseed["name"]) < names.index(second["name"])
    assert "unreviewed" in reseed["run"]


def test_decision_rests_on_the_findings_file_not_on_reviewer_outcome():
    # A run with permission denials but valid findings must pass; a clean run with
    # no findings file must fail. So no step may read the reviewer's outcome or its
    # denial count after validation.
    steps = _steps(REVIEW)
    final_names = {"Check workspace is clean", "Validate findings", "Stamp review evidence"}
    final = [s for s in steps if s.get("name") in final_names]
    assert [s["name"] for s in final] == [
        "Check workspace is clean",
        "Validate findings",
        "Stamp review evidence",
    ]
    for step in final:
        assert "continue-on-error" not in step
        assert "steps.review1" not in step.get("if", "") and "steps.review2" not in step.get(
            "if", ""
        )
    assert "permission_denials" not in yaml.safe_dump(REVIEW)
