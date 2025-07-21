# Contributing Guide

This project follows a tight feedback loop between autonomous agents and human maintainers.

## Workflow
1. **Builder** drafts a pull request with code or documentation changes.
2. **Reviewer** runs the full check suite and verifies the lock file and dataset hashes.
3. The **maintainer** reviews and merges once all checks pass.

Only humans can merge PRs. Reviewer also re-runs checks on `main` nightly to catch regressions.

## Required Checks
- `ruff` for linting and formatting.
- `mypy --strict` for type checking.
- `pytest` for the unit test suite.
- `uv pip sync -r uv.lock` to ensure the environment matches the lock file.
- Dataset SHA-256 verification.

The Reviewer role ensures these checks succeed before a PR can be merged.

See `Agents.md` for the full specification and repo architecture.
