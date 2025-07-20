# Contributing Guide

This project follows a tight feedback loop between autonomous agents and human maintainers.

## Workflow
1. **Builder** drafts a pull request with code or documentation changes.
2. **Reviewer** runs the full check suite (lint, typing, tests, docs) and posts results.
3. A human maintainer reviews and merges when all checks pass.

Only humans can merge PRs. Reviewer also re-runs checks on `main` nightly to catch regressions.

## Required Checks
- `ruff` for linting and basic formatting (configured in `pyproject.toml`).
- `mypy --strict` for type checking.
- `pytest` for the unit test suite.

The Reviewer role ensures these checks succeed before a PR can be merged.

See `Agents.md` for the full specification and repo architecture.
