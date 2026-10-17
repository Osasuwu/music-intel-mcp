"""Triggers and jobs of .github/workflows/unblock-ready.yml (#1981).

The scheduled `reconcile` pass is the only route for an issue whose last blocker
closed in another repo; the issue-event jobs stay the primary, immediate path.
Each test reads the parsed workflow, not its text.
"""

from pathlib import Path

import yaml

WORKFLOW = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "unblock-ready.yml").read_text(
        encoding="utf-8"
    )
)
# PyYAML parses the bare key `on` as the boolean True.
ON = WORKFLOW[True]
JOBS = WORKFLOW["jobs"]


def _mode(job):
    """The `MODE` a job hands to unblock_ready.py."""
    (step,) = [s for s in job["steps"] if "run" in s]
    assert step["run"] == "python3 .github/scripts/unblock_ready.py"
    return step["env"]["MODE"]


def test_the_reconcile_pass_runs_hourly_on_the_schedule():
    # Hourly is the cadence #1981 picked: the label is the only cost of a pass.
    assert ON["schedule"] == [{"cron": "17 * * * *"}]
    job = JOBS["reconcile"]
    assert _mode(job) == "reconcile"
    assert "github.event_name == 'schedule'" in job["if"]


def test_a_manual_dispatch_chooses_between_reconcile_and_sweep():
    mode = ON["workflow_dispatch"]["inputs"]["mode"]
    assert mode["options"] == ["reconcile", "sweep"]
    assert "inputs.mode == 'reconcile'" in JOBS["reconcile"]["if"]
    sweep_if = JOBS["sweep"]["if"]
    assert "inputs.mode == 'sweep'" in sweep_if
    assert "schedule" not in sweep_if


def test_same_repo_unblocking_still_fires_on_the_issue_closed_event():
    assert "closed" in ON["issues"]["types"]
    assert "schedule" not in JOBS["resolve"]["if"]
    assert _mode(JOBS["resolve"]) == "resolve"
    assert JOBS["sync"]["needs"] == "resolve"
    assert _mode(JOBS["sync"]) == "apply"
