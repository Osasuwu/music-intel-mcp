# Code Review Gate — Review Evidence Architecture

## Rule: "Review Evidence"

The `verify-verdict` check is green iff the PR is out of draft, its full changed-file list was read and is non-empty, and it touches no gate machinery; and then either:

1. At least one successful `code-review.yml` run bound to the evaluated head SHA carries a valid, non-blocking `review-evidence.json` artifact, AND no bound run attempt is blocking, has an artifact that cannot be read, or is still in progress; or
2. Every changed file is on the cosmetic allow-list below: images, a few named root files, and markdown and images under `docs/domain/`. Other documentation is code.

The preconditions fail as red `draft`, `files-incomplete` (the files API serves at most 3000), `files-empty` and `gate-machinery`. Two more reds come from provenance, not evidence: `shared-head`, and `untrusted-needs-dispatch` for a fork or Dependabot PR with no evidence yet (see below).

The gate never reads PR comments, timestamps or headings, and the reviewer cannot comment (`gh pr comment` is disallowed). The artifact is the whole verdict.

**Which red wins:**
- **Blocking is sticky.** A blocking artifact stays in force for its commit even if a later run, or a re-run of the same run, is clean, and even if that re-run failed or was cancelled. The fix is a new commit. Known limit ([like-current-song#236](https://github.com/Osasuwu/like-current-song/issues/236)): this holds only while the blocking artifact is retained (90 days); once it expires, a clean result for the same SHA turns the check green.
- **Unreadable is sticky until re-read.** An artifact the verdict could not download (cut-off body, size mismatch, oversized, network error) is red `evidence-unreadable` whatever else is clean, until the verdict job is re-run and reads it. Only a blocking artifact outranks it.
- **Every other red clears on a clean run.** Missing, expired or malformed evidence is superseded as soon as a run for the same SHA comes back clean.
- **A run still in progress holds the check at `evidence-pending`**, whatever the other runs say.
- **A failed or cancelled run counts only for blocking evidence** an earlier attempt of it left; nothing else from it counts either way.
- **A verdict crash is red.** If the verdict script itself raises, it posts red `verify-verdict: verdict-error` before failing the job; it never leaves the check unset or green.

## Two-Workflow Architecture

### code-review.yml (Evidence Producer)

**Triggers:** Pull request events + `workflow_dispatch` for Dependabot/fork dispatch

**Runs on:** a `pull_request` run takes the workflow YAML from the PR's merge ref, so a PR can change its own review workflow (which is why that file is gate machinery). The job's workspace is the PR's BASE SHA; PR head code is never checked out or executed.

**Single job:** `review`

- Snapshots PR metadata and diff through GitHub API
- Stages PR content as inert data under `.pr-head/` (not executable)
- Runs the reviewer (Claude Code with 8 finding classes only)
- Validates and stamps findings into `review-evidence.json`
- Uploads it as `review-evidence-<run_attempt>`, one artifact per run attempt, never overwritten (`overwrite: false`), 90-day retention. A re-run cannot replace an earlier attempt's evidence; the verdict reads all of them.

**Reviewer sandbox:**
- The job token is read-only (`contents: read`, `pull-requests: read`; no `id-token`), and it is the token handed to the action.
- The reviewer gets an exact read-verb allowlist (`Read`, `Grep`, `Glob`, read-only `git`/`gh` verbs, `py_compile`), not `git:*`, `bash -n` or `node --check`. `tests/test_code_review_allowed_tools.py` pins it.
- `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` scrubs Anthropic, cloud and Actions secrets from the reviewer's subprocess environments. It is best-effort, not a guarantee, and it needs bubblewrap: the job installs it (and lifts Ubuntu's AppArmor restriction on unprivileged user namespaces) in a plain step before the reviewer, and fails at that step, by name, if it cannot. The scrub is never switched off to get a run through.
- Two reviewer attempts; the second reseeds the findings file. If both fail validation, the job fails and uploads no evidence.

**Finding classes (judgment, not mechanical):**
- `regression` — reintroduces a bug
- `exception-handling` — wrong recovery strategy
- `intent-vs-logic` — code doesn't match PR description
- `breaking-contract` — breaks callers outside this diff
- `concurrency` — race, deadlock, or ordering bug
- `requirement-semantics` — doesn't satisfy acceptance criterion
- `design-modularity` — real erosion of module boundaries
- `performance` — N+1 or O(n²) added in the diff

**Dispatch path (Dependabot, forks):** Must provide both `pr_number` and `head_sha` inputs and be dispatched on the default branch. It runs in the `untrusted-review` environment, which holds no secrets: it exists for its required reviewer and its default-branch deployment policy. `CLAUDE_CODE_OAUTH_TOKEN` is a repository secret, used by every review run.

### code-gate-verdict.yml (Verdict — Base-Pinned)

**Triggers:** `pull_request_target` + `workflow_run` (on code-review completion)

**Runs on:** DEFAULT branch only (PR cannot change its own verdict logic)

**Permissions:** Read-only (contents, actions, pull-requests)

**Two jobs:**
1. `resolve` — parses event, outputs `(pr_number, head_sha)`
2. `verdict` — reads review-evidence artifact, posts `verify-verdict` check via osasuwu-ci GitHub App

**Environment:** `code-gate-verdict` (holds GATE_APP_ID and GATE_APP_PRIVATE_KEY; deployments restricted to default branch only)

## Provenance & Safety

**Run qualification:** A `pull_request` run counts only when `run.pull_requests[]` contains the PR and that PR's base is the default branch. A fork's `pull_request` run carries an empty `pull_requests[]` and is excluded. If the run also lists an open PR into another base on the same commit, it may have run that branch's copy of the review workflow, so it is evidence for no PR: red `evidence-foreign-base` until that PR is closed or retargeted, or a maintainer re-dispatches. A blocking artifact it carries still sticks.

A `workflow_dispatch` run counts only when its `display_title` embeds both the PR number and head SHA (the run-name does this) and the commit it ran from is the default branch's tip or one of its ancestors, checked through the compare API (`behind` or `identical`), never through the run's `head_branch`, which the dispatcher chooses. So a dispatch keeps counting after the default branch moves on, and a dispatch from any other ref counts for nothing.

**Other bases and shared heads:** Only a PR into the default branch gets a `verify-verdict`; a PR into another base gets none, because the check belongs to the commit and would land on any PR into the default branch that shares the head. Two open PRs into the default branch on the same head commit are red `shared-head` until one is closed or gets its own commit.

**Artifact binding:** The artifact's own `sha` and `base_ref` must equal the evaluated SHA and PR's base. A mismatch is red.

**Gate machinery protection:** A PR touching `code-review.yml`, `code-gate-verdict.yml`, `.github/scripts/code_gate_verdict.py`, anything under `.github/actions/` or `.claude/hooks/` (any hook runs in the reviewer's harness), or the agent behavior files `.claude/settings.json` and `.claude/marketplace/.claude-plugin/marketplace.json` is always red. Other `.claude/` and `.github/scripts/` files are ordinary code and go through review. The sanctioned unblock is a human review-blind admin-merge backed by a fresh-session `/code-review` posted with the final SHA.

**Cosmetic allow-list (music-intel-mcp):**
- Images anywhere: `*.png`, `*.jpg`, `*.gif`, `*.webp`
- Root level, by exact name: `README.md`, `SECURITY.md`, `THIRD_PARTY_LICENSES`, `LICENSE`, `LICENSE.md`, `LICENSE.txt`, `LICENSE-APACHE`, `LICENSE-MIT`. A name merely starting with `LICENSE` (`LICENSE.py`) is code.
- `docs/domain/` markdown and images at any depth (product documentation, not behavior-carrying)

**Code (requires review):**
- Anything in `docs/` except markdown and images under `docs/domain/`
- Every root file not named in the allow-list above (includes `AGENTS.md`, `CONTEXT.md`, `INVARIANTS.md`, `CLAUDE.md`)
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
   - `untrusted-review`: Required reviewer (human approval before dispatch) and a default-branch deployment policy; no secrets.
   - `CLAUDE_CODE_OAUTH_TOKEN` is a repository secret, used by every review run, dispatched or not.

3. **Branch protection:** Require `verify-verdict` bound to the osasuwu-ci App's app id, with `strict: true` (a PR must be up to date with the default branch). Land the workflows first and let the App post `verify-verdict` once: a required check bound to an App that has never posted it deadlocks every PR.

4. **Canary testing:** Before enabling, test on 7 representative PRs:
   - Plain code change with a clean review (should pass)
   - Cosmetic-only change (should pass)
   - Blocking finding (should fail)
   - Mid-run push (should cancel in-flight review and start over)
   - Re-run of a blocking run that comes back clean (should stay red)
   - Gate machinery change (should fail unconditionally)
   - Fork or Dependabot PR (should fail until a maintainer dispatches the review for its head SHA, then pass on a clean dispatch)

## Canonical Description

This document is the music-intel-mcp port. For the authoritative design, architecture decisions, and historical context, see [jarvis/docs/reference/github-repo-setup.md](https://github.com/Osasuwu/jarvis/blob/main/docs/reference/github-repo-setup.md) §3 "Review evidence."
