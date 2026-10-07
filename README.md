# music-intel-mcp

Track-level music recommender with an anti-bubble bias — surfaces what you *should* like but probably haven't heard, instead of safe rehashes of your last 30 days.

**Status:** v0 bootstrap, Fresh Python rewrite of the dormant `Osasuwu/OOP` Java project. Archived: branch `legacy/java`, tag `v0-uni`.

## Pillars (per brainstorm 2026-05-13)

1. **Understand** — analytics over listening history (phase detection, cluster maps, novelty curves).
2. **Discover** — track-level recommendations biased against the current bubble.
3. **Act** — weekly playlist push to Spotify, MCP surface for Jarvis integration.

Scope, acceptance criteria and architectural decisions are in the domain model — start at the section index in `CONTEXT.md`.

## Repo conventions

- `AGENTS.md` — rules and process for AI agents working in this repo. It is the only rules file: Claude Code reads it natively, so the repo has no `CLAUDE.md`.
- `CONTEXT.md` — domain model index: glossary plus a section index pointing at `docs/domain/`.
- `INVARIANTS.md` — extracted `CONTEXT.md` invariants, imported into every agent session.
- `src/music_intel_mcp/` — Python package.
- `tests/` — pytest suite, green from day 1.

## Development

```bash
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
pytest
ruff check .
```
