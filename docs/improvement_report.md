# Release-candidate improvement report

## Scope

This report covers the transition from the verified v0.4.2 source snapshot to v0.4.3rc1. The
candidate intentionally preserves the ETA, RCOT, calibration, decision, serving, and release
contracts. The work is limited to candidate handoff integrity and a missing static type gate.

## Baseline

- Input ZIP SHA-256: `151909a7782c383ddb3d874ca8cc7b1fdc83b3a1734725d58f28801d153cd8af`
- Baseline version: v0.4.2
- Baseline tests: 132 passed
- Baseline source coverage: 76.40%
- Release gate: `PASS_BASELINE_CHAMPION`
- Quantile champion: `without_rcot`
- Decision champion: `tail_risk`
- RCOT gate: `HOLD`
- Baseline source fingerprint is stored in `artifacts/baseline_source_fingerprint.json`.
- Baseline metric evidence is stored in `artifacts/baseline_candidate_metrics.json`.

## Round 1 — Static type gate

### Problem

The release ran lint, formatting, compile, tests, and packaging but no static type checker. A direct
mypy run identified four source-contract errors in lock metadata conversion, external scalar parsing,
and API Literal fields.

### Correction

- Added mypy and PyYAML stubs to development dependencies.
- Added `make typecheck` and `python scripts/tasks.py typecheck`.
- Added mypy to the exact release and GitHub compatibility jobs.
- Corrected the four type-contract errors without changing runtime behavior.

### Acceptance criterion

`python -m mypy src/reference_eta` must report zero errors and the full behavioral suite must remain
passing.

## Round 2 — Evidence-grounded candidate handoff

### Problem

The repository contained strong release evidence but lacked the required candidate-level handoff,
source/diff fingerprint, clean-copy evidence record, and machine-verifiable next qualification gates.

### Correction

- Added deterministic baseline/candidate source fingerprints and a change-set checksum.
- Added isolated clean-copy testing, installed-wheel import, and manifest verification.
- Added `release_candidate_handoff.json` plus SHA-256 sidecar.
- Added artifact, dependency, metric, command, limitation, and evidence-level records.
- Excluded the handoff from the repository manifest to avoid a circular checksum; the handoff instead
  records and verifies the repository manifest checksum.

### Acceptance criterion

The handoff is emitted only after the release manifest and clean-copy check pass. Every referenced
artifact must exist and match its recorded SHA-256.

## Round 3 — Pure task construction

### Problem

The cross-platform task runner deleted `dist/` while merely constructing the `package-check` command
list. A unit test that inspected the command therefore removed wheel and sdist evidence from the
clean copied candidate and correctly caused repository-manifest failure.

### Correction

- Removed filesystem mutation from `_commands()`.
- Added an explicit `clean-distribution` command that runs only when package-check executes.
- Added a regression test proving command inspection preserves an existing distribution.

### Acceptance criterion

Constructing any task must be side-effect free. Clean-copy tests and repository-manifest verification
must retain both distributions.

## Regression policy

The candidate must preserve all v0.4.2 scientific and operational gates. No model, split, target,
threshold, evaluation formula, or decision capacity is changed. A test, coverage, release, API,
locking, calibration, or distribution regression blocks the candidate.

## Evidence interpretation

- E0: source and documents reviewed.
- E1: command or hosted workflow designed.
- E2: executed in the current working environment.
- E3: executed from a clean copied source and installed distribution.
- E4: exact commit executed by GitHub-hosted qualification; not claimed here.

The generated handoff is a release-candidate transfer artifact, not a production-release declaration.
