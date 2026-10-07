"""Shape of the two code-gate workflows (#226).

code-gate-verdict.yml is the trusted half: it runs from the default branch and
decides verify-verdict. Its safety is structural — it must never run PR code,
never hold more than read permissions, and never let a PR-controlled string reach
a shell. code-review.yml produces the evidence; evidence upload settings ensure a
missing or stale review fails closed. Each test walks the parsed YAML, not the file
text.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
REVIEW = yaml.safe_load((WORKFLOWS / "code-review.yml").read_text(encoding="utf-8"))
VERDICT = yaml.safe_load((WORKFLOWS / "code-gate-verdict.yml").read_text(encoding="utf-8"))

# PyYAML parses the bare key `on` as the boolean True.
ON = True


def _steps(spec: dict) -> list[dict]:
    return [step for job in spec["jobs"].values() for step in job["steps"]]


def _run_bodies(spec: dict) -> list[str]:
    return [step["run"] for step in _steps(spec) if "run" in step]


# --- verdict workflow -----------------------------------------------


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
    concurrency = VERDICT["jobs"]["verdict"]["concurrency"]
    expected_group = (
        "code-gate-${{ needs.resolve.outputs.pr_number }}-${{ needs.resolve.outputs.head_sha }}"
    )
    assert concurrency == {
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


# --- review workflow -----


def test_review_concurrency_cancels_a_superseded_review():
    assert REVIEW["concurrency"] == {
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


def test_review_checks_out_base_only():
    checkout = [
        s for s in _steps(REVIEW) if str(s.get("uses", "")).startswith("actions/checkout@")
    ][0]
    expected_ref = (
        "${{ github.event.pull_request.base.sha || github.event.repository.default_branch }}"
    )
    assert checkout["with"]["ref"] == expected_ref


def test_evidence_artifact_is_retained_90_days_and_overwritable():
    uploads = [
        s for s in _steps(REVIEW) if str(s.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    (evidence,) = [s for s in uploads if s["with"]["name"] == "review-evidence"]
    assert evidence["with"]["retention-days"] == 90
    assert evidence["with"]["overwrite"] is True
    assert evidence["with"]["if-no-files-found"] == "error"


def test_decision_rests_on_the_findings_file():
    # Final validation steps must not depend on the review action outcome.
    steps = _steps(REVIEW)
    final_names = {"Check workspace is clean", "Validate findings", "Stamp review evidence"}
    final = [s for s in steps if s.get("name") in final_names]
    assert len(final) == 3
    assert {s["name"] for s in final} == final_names
    for step in final:
        assert "continue-on-error" not in step
        assert "steps.review" not in step.get("if", "")
