# Code Review Gate — Review Evidence Architecture

## Rule: "Review Evidence"

The `verify-verdict` check is green iff:

1. At least one successful `code-review.yml` run bound to the evaluated head SHA carries a valid, non-blocking `review-evidence.json` artifact, AND no bound run is blocking, unfinished, expired, or missing its artifact (sticky-worst); or
2. Every changed file is cosmetic (documentation, images, licenses).

The gate never reads the PR comment, timestamp, or heading. The artifact is the machine-readable verdict; the comment is for humans only.

## Two-Workflow Architecture

### code-review.yml (Evidence Producer)

**Triggers:** Pull request events + `workflow_dispatch` for Dependabot/fork dispatch

**Runs on:** PR's BASE branch (never PR head code is executed)

**Single job:** `review`

- Snapshots PR metadata and diff through GitHub API
- Stages PR content as inert data under `.pr-head/` (not executable)
- Runs the reviewer (Claude Code with 8 finding classes only)
- Validates and stamps findings into `review-evidence.json`
- Uploads artifact with 90-day retention, overwritable

**Finding classes (judgment, not mechanical):**
- `regression` — reintroduces a bug
- `exception-handling` — wrong recovery strategy
- `intent-vs-logic` — code doesn't match PR description
- `breaking-contract` — breaks callers outside this diff
- `concurrency` — race, deadlock, or ordering bug
- `requirement-semantics` — doesn't satisfy acceptance criterion
- `design-modularity` — real erosion of module boundaries
- `performance` — N+1 or O(n²) added in the diff

**Dispatch path (Dependabot, forks):** Must provide both `pr_number` and `head_sha` inputs; runs on default branch with CLAUDE_CODE_OAUTH_TOKEN available in `untrusted-review` environment.

### code-gate-verdict.yml (Verdict — Base-Pinned)

**Triggers:** `pull_request_target` + `workflow_run` (on code-review completion)

**Runs on:** DEFAULT branch only (PR cannot change its own verdict logic)

**Permissions:** Read-only (contents, actions, pull-requests)

**Two jobs:**
1. `resolve` — parses event, outputs `(pr_number, head_sha)`
2. `verdict` — reads review-evidence artifact, posts `verify-verdict` check via osasuwu-ci GitHub App

**Environment:** `code-gate-verdict` (holds GATE_APP_ID and GATE_APP_PRIVATE_KEY; deployments restricted to default branch only)

## Provenance & Safety

**Run qualification:** A `pull_request` run counts only when `run.pull_requests[]` contains the PR and that PR's base is the default branch. A fork's `pull_request` run carries an empty `pull_requests[]` and is excluded. A `workflow_dispatch` run counts only when it ran from the default branch and its `display_title` embeds both the PR number and head SHA (the run-name does this).

**Artifact binding:** The artifact's own `sha` and `base_ref` must equal the evaluated SHA and PR's base. A mismatch is red.

**Gate machinery protection:** A PR touching the review workflow, verdict workflow, this script, or any agent/gate behavior file (`.claude/` config, `.github/scripts/`, actions) is always red. The sanctioned unblock is a human review-blind admin-merge backed by a fresh-session `/code-review` posted with the final SHA.

**Cosmetic allow-list (music-intel-mcp):**
- Images anywhere: `*.png`, `*.jpg`, `*.gif`, `*.webp`
- Root level: `README.md`, `LICENSE*`, `SECURITY.md`, `THIRD_PARTY_LICENSES`
- `docs/domain/` with `.md` suffix (product documentation, not behavior-carrying)
- Any `.md` in nested paths under root-level allow-list

**Code (requires review):**
- Anything in `docs/` except `docs/domain/*.md`
- All root `.md` files except `README.md` and `LICENSE*` (includes `AGENTS.md`, `CONTEXT.md`, `INVARIANTS.md`)
- `.claude/` configuration and hooks
- `.github/` workflows and scripts
- `src/`, `tests/`, `native/`, `schemas/`, and all production code

See the decision function `is_cosmetic()` in `.github/scripts/code_gate_verdict.py` for the authoritative implementation.

## Required Setup

1. **GitHub App:** Create or import `osasuwu-ci` GitHub App in the repository settings with:
   - Permissions: `checks: write`
   - Subscriptions: none (webhook not needed)

2. **Environments:**
   - `code-gate-verdict`: Deployment restrictions to default branch only. Secrets:
     - `GATE_APP_ID`
     - `GATE_APP_PRIVATE_KEY`
   - `untrusted-review`: Required reviewer (human approval before dispatch); no secrets in this environment. CLAUDE_CODE_OAUTH_TOKEN available only if dispatch path needs it.

3. **Branch protection:** Bind `verify-verdict` check to the osasuwu-ci app ID. Mark as required.

4. **Canary testing:** Before enabling, test on 5 representative PRs:
   - Plain code change (should block on any finding)
   - Cosmetic-only change (should pass)
   - Blocking finding (should fail)
   - Mid-run push (should cancel in-flight review and start over)
   - Gate machinery change (should fail unconditionally)

## Canonical Description

This document is the music-intel-mcp port. For the authoritative design, architecture decisions, and historical context, see [jarvis/docs/reference/github-repo-setup.md](https://github.com/Osasuwu/jarvis/blob/main/docs/reference/github-repo-setup.md) §3 "Review evidence."
