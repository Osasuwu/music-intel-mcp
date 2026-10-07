# AGENTS.md — music-intel-mcp

Three-way split (mirrors Jarvis convention):
- **`AGENTS.md`** (this file) — *rules*: process, conventions, what to do, what NOT to do. It is the sole project-rules file; Claude Code reads it natively, so there is deliberately no `CLAUDE.md` (any `CLAUDE.md`, `.claude/CLAUDE.md` or `CLAUDE.local.md` would shadow it).
- **`CONTEXT.md`** — *domain model index*: the Glossary, the Invariants pointer, and a **section index** with one line per domain section — its file under `docs/domain/` and a "read when" clause. Read the Glossary and the index, then open only the `docs/domain/` file the task needs; there is no reason to read the whole domain model. Grows through `/grill`: a new term goes into the Glossary, a new section gets its own `docs/domain/` file plus an index line. Authoritative source for product terminology. Its `## Invariants` section is extracted into `INVARIANTS.md` (imported below) so the domain invariants arrive in every session without an agent having to go fetch `CONTEXT.md`.
- Identity (`SOUL.md`) — inherited from the operator's user-level config (imported by `~/.claude/CLAUDE.md`). No per-repo override.

@INVARIANTS.md

## What this project is

Anti-bubble track-level music recommender. Reverses Spotify's "what you'll definitely like" bias — surfaces tracks likely to surprise/disappoint with high upside.

**Three pillars:** Understand (analytics) → Discover (recommend) → Act (playlist push + MCP for Jarvis).

Product details, data sources, pipelines, architectural decisions and open questions → the section index in `CONTEXT.md`, which points at the `docs/domain/` file for each.

## Stop-points (gates)

1. **`/grill` before code that decides product behavior.** The domain model is filled in — the V0 engine and the pilot slices were each grilled and their resolved state is in `CONTEXT.md` / `docs/domain/` — so the gate is per change, not a one-time project gate. Setup-level work (workflows, AGENTS.md, labels, deps) is fine. Anything that decides product behavior (recommendation strategy, similarity scoring, data schema, MCP tool surface, a new pipeline or ingestion source) runs the grill trigger checkbox in `~/.claude/reference/engineering-principles.md` (§ *Grill trigger checkbox*): ≥1 yes ⇒ `/grill` first. Setup itself is 0 yes ⇒ proceed.

2. **No mechanical port from the `legacy/java` branch** (also tagged `v0-uni`; not a directory on `main` — read it with `git show origin/legacy/java:<path>`). Java code is uni-grade reference for *what existed*, not a TZ for what to build. Read it for credentials/data layout only.

## Definition of Done

Before marking any task complete:

1. **Tests are green** — pytest + ruff + pre-commit all pass. CI status is the source of truth, not local "looks fine".
2. **No hardcoded secrets** — `.env.example` declares the metadata; values live in `.env` (gitignored) or the host env.
3. **The domain model reflects the change** — a new term goes into the Glossary in `CONTEXT.md`; a new invariant into `INVARIANTS.md`; a change to a documented area into its `docs/domain/` file; a new area into a new `docs/domain/` file plus a section-index line in `CONTEXT.md`. Don't let the domain docs drift behind code.
4. **Memory** — non-obvious decision or learning → append to `~/.claude/projects/<project>/memory/decisions.md` (dated line: what was decided, why). Code captures *what*; memory captures *why*.

## Process

- **Branches** from `main`. One issue → one PR. PR body must `Closes #NNN`, or carry the `priority:critical` label (hotfix), or contain a `[no-issue]` body marker (drive-by / artifact PRs, e.g. grill CONTEXT.md notes), or use a `refactor:` / `refactor(scope):` title prefix — the four lanes of the universal owned-repo contract (jarvis#428).
- **Decisions** belong in memory (`decisions.md`), not in PR bodies or markdown files. `CONTEXT.md` and `docs/domain/` capture *resolved* state; ephemeral debate goes to GitHub Discussions.
- **TDD where the domain decides correctness** — recommendation scoring, similarity, anti-bubble penalty, importers. Write the failing test that defines "right answer" before the implementation.
- **Vertical slices, not horizontal.** Each issue ships end-to-end (data → logic → test → CLI/output). Don't do "all loaders, then all scoring, then all output".
- **No `git add -A`** in scratch-heavy directories. Use explicit paths.

## Project-specific rules

- **External APIs are rate-limited and rate-cost real money** — cache aggressively. Shared track-level metadata → Supabase Postgres (anonymous, 90-day TTL per entry, see `docs/domain/shared-store-schema.md`). Per-user data → local JSON/JSONL files under the data root (see *Per-user data on disk* below); never to the cloud at V0/V1. Tests must not call live APIs; use fixtures or `respx`-style mocks.
- **AcousticBrainz / MusicBrainz dumps live OUTSIDE the repo** — in `.scratch/` (gitignored, local-only) or at the paths named by `MUSICBRAINZ_DUMP_DIR`, `ACOUSTICBRAINZ_DUMP_DIR` and `MUSICBRAINZ_ISRC_INDEX` (declared in `.env.example`). Never commit the dump.
- **Spotify API scope is constrained** — our app is NOT grandfathered: no audio-features, no related-artists, no recommendations endpoints. ISRC waterfall via MusicBrainz dump is the fallback. Document any new endpoint dependency in the relevant `docs/domain/` file.

## Per-user data on disk

Personal data (listening history, RootProfile snapshots, analyses, tokens) lives in a local **data root**, never in the repo and never in the cloud. Resolution order, from `resolve_data_root()` in `src/music_intel_mcp/store.py`:

1. the `--data-dir` flag of the `music-intel` subcommand;
2. else the `MUSIC_INTEL_DATA_DIR` environment variable — the CLI first loads the nearest gitignored `.env` (searched upward from the invocation directory) into the environment (host env wins), so the variable may be set there;
3. else `./data`, relative to the directory the CLI is run from.

`/data/` is gitignored, so the data root never exists on `main` and a fresh git worktree has none — the real one is wherever the user runs the CLI (usually their main checkout). Layout under the root: `history.jsonl` (listening events), `profiles/<snapshot>.json` (RootProfile snapshots), `library.json`, `audio_analysis/`, `fingerprints/`, `aliases.jsonl`, `identity/` and `artist_identity/` caches, capture/replay journals (`*.jsonl`). `spotify_user_token.json` is an OAuth credential — never read or print it. Pilot participants get their own transient root, `data/participants/<id>/` on the processing node (`docs/pilot-runbook.md`), deleted by `music-intel purge`.

**Finding the root without reading env files.** Never open, `cat` or grep `.env` to look for the path. Instead: (a) check for `data/` in the checkout the user runs the CLI from; (b) check `MUSIC_INTEL_DATA_DIR` in the host environment; (c) if still unknown, from that checkout print the root through the CLI's own resolver — this prints only the path, never a credential:

```
PYTHONPATH=src python -c "from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv(usecwd=True)); from music_intel_mcp.store import resolve_data_root; print(resolve_data_root(None).resolve())"
```

If none of these settles it, ask the user for the path.

## Key files

| What | Where |
|---|---|
| Domain model index (Glossary + section index) | `CONTEXT.md` |
| Domain model sections | `docs/domain/` |
| Domain invariants (imported every session) | `INVARIANTS.md` |
| Per-user data root resolver | `src/music_intel_mcp/store.py` (`resolve_data_root`) |
| Pilot operator runbook | `docs/pilot-runbook.md` |
| Python package | `src/music_intel_mcp/` |
| Tests | `tests/` |
| Workflows | `.github/workflows/` |
| Pre-commit | `.pre-commit-config.yaml` |
| Java archive | branch `legacy/java`, tag `v0-uni` — git refs, not paths on `main` (read-only reference) |

## Related context (memory hooks)

- `music_intel_mcp_project_revival` — brainstorm decisions (3 pillars, anti-bubble bias, multi-source enrichment, similarity strategies).
- Decision episode `d94c44fb-03eb-4867-8440-9910f905a903` — repo revival + jarvis-style conventions.
