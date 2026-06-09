# CLAUDE.md — music-intel-mcp

Three-way split (mirrors Jarvis convention):
- **`CLAUDE.md`** (this file) — *rules*: process, conventions, what to do, what NOT to do.
- **`CONTEXT.md`** — *domain model*: glossary, invariants, architectural decisions. Grows inline through `/grill`. Authoritative source for product terminology.
- Identity (`SOUL.md`) — inherited from user-level `~/.claude/SOUL.md`. No per-repo override yet.

## What this project is

Anti-bubble track-level music recommender. Reverses Spotify's "what you'll definitely like" bias — surfaces tracks likely to surprise/disappoint with high upside.

**Three pillars:** Understand (analytics) → Discover (recommend) → Act (playlist push + MCP for Jarvis).

Product details, data sources, similarity strategies → `CONTEXT.md` (currently empty — to be filled via `/grill` before any product code).

## Stop-points (gates)

1. **`/grill` is mandatory before any product code.** Setup-level work (workflows, CLAUDE.md, labels, deps) is fine. Anything that decides product behavior (recommendation strategy, similarity scoring, data schema, MCP tool surface) requires the SOUL.md grill-trigger checkbox: ≥1 yes ⇒ `/grill` first. Setup itself is 0 yes ⇒ proceed.

2. **No mechanical port from `legacy/java`.** Java code is uni-grade reference for *what existed*, not a TZ for what to build. Read it for credentials/data layout only.

## Definition of Done

Before marking any task complete:

1. **Tests are green** — pytest + ruff + pre-commit all pass. CI status is the source of truth, not local "looks fine".
2. **No hardcoded secrets** — `.env.example` declares the metadata; values live in `.env` (gitignored) or the host env.
3. **CONTEXT.md reflects the change** — any new term, invariant, or architectural shift is documented inline. Don't let CONTEXT.md drift behind code.
4. **Memory** — non-obvious decision or learning → `record_decision` or `memory_store` with `source_provenance`. Code captures *what*; memory captures *why*.

## Process

- **Branches** from `main`. One issue → one PR. PR body must `Closes #NNN` or carry the `priority:critical` label (hotfix bypass, mirrors jarvis pattern).
- **Decisions** belong in memory (`record_decision`), not in PR bodies or markdown files. CONTEXT.md captures *resolved* state; ephemeral debate goes to GitHub Discussions.
- **TDD where the domain decides correctness** — recommendation scoring, similarity, anti-bubble penalty, importers. Write the failing test that defines "right answer" before the implementation.
- **Vertical slices, not horizontal.** Each issue ships end-to-end (data → logic → test → CLI/output). Don't do "all loaders, then all scoring, then all output".
- **No `git add -A`** in scratch-heavy directories. Use explicit paths.

## Project-specific rules

- **External APIs are rate-limited and rate-cost real money** — cache aggressively. Shared track-level metadata → Supabase Postgres (anonymous, 90-day TTL per entry, see CONTEXT.md). Per-user data → local JSON/JSONL files; never to the cloud at V0/V1. Tests must not call live APIs; use fixtures or `respx`-style mocks.
- **AcousticBrainz / MusicBrainz dumps live OUTSIDE the repo** — `.scratch/` or env-pointed paths. Never commit the dump.
- **Spotify API scope is constrained** — our app is NOT grandfathered: no audio-features, no related-artists, no recommendations endpoints. ISRC waterfall via MusicBrainz dump is the fallback. Document any new endpoint dependency in `CONTEXT.md`.

## Key files

| What | Where |
|---|---|
| Domain model | `CONTEXT.md` |
| Python package | `src/music_intel_mcp/` |
| Tests | `tests/` |
| Workflows | `.github/workflows/` |
| Pre-commit | `.pre-commit-config.yaml` |
| Java archive | branch `legacy/java`, tag `v0-uni` (read-only reference) |

## Related context (memory hooks)

- `music_intel_mcp_project_revival` — brainstorm decisions (3 pillars, anti-bubble bias, multi-source enrichment, similarity strategies).
- Decision episode `d94c44fb-03eb-4867-8440-9910f905a903` — repo revival + jarvis-style conventions.
