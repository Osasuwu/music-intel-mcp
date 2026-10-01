"""The `waiting-human-review` merge hold in .github/workflows/waiting-human-review.yml (#235).

The step's github-script body is lifted out of the workflow verbatim and run under node against
stand-ins for `github`, `context` and `core`, so these tests exercise the check that actually
runs in CI rather than a copy of its rule.
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
    (_root / ".github" / "workflows" / "waiting-human-review.yml").read_text(encoding="utf-8")
)
CHECK_NAME = "waiting-human-review"
LABEL = "waiting-human-review"
PR_NUMBER = 7

# `pulls.get` answers from `fresh`, the event payload carries `payload_labels`: the two differ
# on purpose in one test, to pin that the check reads the re-fetched PR.
_HARNESS = """\
const fixture = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const calls = [];
const context = {
  repo: { owner: 'o', repo: 'r' },
  payload: { pull_request: { number: %(pr)d, labels: fixture.payload_labels } },
};
const get = async (args) => { calls.push(args); return { data: fixture.fresh }; };
const github = { rest: { pulls: { get } } };
let failed = null;
const core = { setFailed: (msg) => { failed = msg; } };
const logs = [];
console.log = (msg) => logs.push(msg);
(async () => {
%(script)s
})().then(
  () => process.stdout.write(JSON.stringify({ failed, calls, logs })),
  (err) => { console.error(err); process.exit(97); },
);
"""


def _script() -> str:
    steps = _workflow["jobs"][CHECK_NAME]["steps"]
    return next(s["with"]["script"] for s in steps if "github-script" in s.get("uses", ""))


def _run(tmp_path, *, labels=(), reviewers=(), teams=(), payload_labels=None) -> dict:
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is required to run the waiting-human-review script in CI")
        pytest.skip("node not available locally")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS % {"pr": PR_NUMBER, "script": _script()}, encoding="utf-8")
    fresh = {
        "labels": [{"name": n} for n in labels],
        "requested_reviewers": [{"login": r} for r in reviewers],
        "requested_teams": [{"slug": t} for t in teams],
    }
    fixture = {
        "fresh": fresh,
        "payload_labels": fresh["labels"] if payload_labels is None else payload_labels,
    }
    result = subprocess.run(
        [node, str(harness)],
        input=json.dumps(fixture),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_label_alone_holds_the_pr(tmp_path):
    out = _run(tmp_path, labels=["priority:low", LABEL])
    assert out["failed"] == f"The `{LABEL}` label is present — a human look is owed."


def test_no_label_and_no_request_passes(tmp_path):
    out = _run(tmp_path, labels=["priority:low", "status:owner-queue"])
    assert out["failed"] is None
    assert out["logs"] == [
        "No pending review request and no waiting-human-review label — check passes."
    ]


@pytest.mark.parametrize("who", [{"reviewers": ["alice"]}, {"teams": ["core"]}])
def test_pending_review_request_holds_the_pr(tmp_path, who):
    out = _run(tmp_path, **who)
    assert out["failed"] == (
        "A review has been requested and is still pending — a human look is owed."
    )


def test_label_removed_after_the_event_snapshot_passes(tmp_path):
    # The payload still lists the label; the PR fetched from the API no longer does.
    out = _run(tmp_path, labels=[], payload_labels=[{"name": LABEL}])
    assert out["failed"] is None
    assert out["calls"] == [{"owner": "o", "repo": "r", "pull_number": PR_NUMBER}]


def test_check_reruns_when_the_label_or_a_review_changes():
    triggers = _workflow["on"] if "on" in _workflow else _workflow[True]  # YAML 1.1: `on` -> True
    pr_types = set(triggers["pull_request"]["types"])
    assert {"labeled", "unlabeled", "review_requested", "review_request_removed"} <= pr_types
    assert {"opened", "synchronize", "ready_for_review"} <= pr_types
    assert "submitted" in triggers["pull_request_review"]["types"]


def test_job_id_is_the_required_check_name():
    # Branch protection requires the context by this exact name; a `name:` override renames it
    # and the required check never reports.
    job = _workflow["jobs"][CHECK_NAME]
    assert "name" not in job
