# music-intel-mcp

Track-level music recommender with an anti-bubble bias — surfaces what you *should* like but probably haven't heard, instead of safe rehashes of your last 30 days.

**Status:** v0 bootstrap, Fresh Python rewrite of the dormant `Osasuwu/OOP` Java project. Archived: branch `legacy/java`, tag `v0-uni`.

## Pillars (per brainstorm 2026-05-13)

1. **Understand** — analytics over listening history (phase detection, cluster maps, novelty curves).
2. **Discover** — track-level recommendations biased against the current bubble.
3. **Act** — weekly playlist push to Spotify, MCP surface for Jarvis integration.

Scope, acceptance criteria and architectural decisions are pending `/grill` — see `CONTEXT.md` once filled.

## Repo conventions

- `CLAUDE.md` — rules and process for AI agents working in this repo.
- `CONTEXT.md` — domain model (glossary, invariants, architecture). Grows inline.
- `src/music_intel_mcp/` — Python package.
- `tests/` — pytest suite, green from day 1.

## Development

```bash
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
pytest
ruff check .
```
