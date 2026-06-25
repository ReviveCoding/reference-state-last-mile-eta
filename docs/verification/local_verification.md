# Local Verification Snapshot

## Status

This repository completed the following local verification steps on Windows with Python 3.11.9 after portability hardening.

| Gate | Result |
|---|---|
| TOML parsing and build-system compatibility | PASS |
| Ruff import sorting, formatting, and lint | PASS |
| Static type check: `mypy src` | PASS |
| Full pytest suite in detached non-interactive execution | PASS |
| Wheel and sdist build | PASS |
| `twine check` | PASS |
| Clean virtual environment wheel installation | PASS |
| Clean work-directory import of `reference_eta` | PASS |
| High-confidence secret pattern scan | PASS |
| Candidate files at or above 25 MiB | None found |
| Portability scan for committed absolute user-home paths | PASS |

## Test-run note

Direct interactive PowerShell pytest runs received a `KeyboardInterrupt` during stale-lock file cleanup. The same targeted stale-lock test and the full suite both passed when executed in a detached non-interactive child process. This is treated as a local console-control/interruption issue, not an assertion failure in the locking implementation.

## Claim boundary

These are local verification results. They do not claim GitHub-hosted CI, container validation, cross-platform validation, production deployment, or production ETA impact.

## Version-control policy

Generated experiment output, raw/normalized data, model binaries, prediction-level CSVs, historical local transcripts, and stale release-handoff metadata are excluded. Sanitized summary evidence is retained in `docs/evidence/`.