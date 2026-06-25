# Changelog

## 0.4.3rc1

- Added a mypy source-package gate and corrected lock, scalar parsing, and API Literal contracts.
- Added deterministic baseline and candidate source fingerprints with a diff checksum.
- Added clean-copy qualification inside the canonical release chain.
- Added a validated `release_candidate_handoff.json` with artifact checksums, metrics, evidence levels, and next qualification gates.
- Added candidate improvement and known-limitations reports.
- Kept the candidate explicitly marked `RELEASE_CANDIDATE_NOT_QUALIFIED` until hosted Windows, GitHub, Docker, official-data, and CUDA gates run.

## 0.4.2

- Added streamed ASGI request-body enforcement for chunked and declared-length requests.
- Added per-worker single-flight artifact loading and hot-reload metadata refresh.
- Reported and validated the artifact model version from run provenance.
- Expanded the real-Uvicorn release gate to two workers, worker distribution, and graceful shutdown.
- Added deterministic CycloneDX identity and wheel-bound SBOM attestation configuration.
- Sanitized client-supplied request IDs before structured telemetry and response echo.
- Hardened locking verification shutdown after multiprocessing evidence is persisted.
- Expanded the suite to 132 tests with 76.40% source coverage before final release.

## 0.4.1

- Added OS-independent task runner and PowerShell release wrapper.
- Added shared POSIX/Windows process-tree control.
- Added release, pipeline, and publish locks with owner metadata and stale recovery.
- Added atomic-publish failure injection and cross-process contention verification.
- Added FastAPI lifespan preload, structured payload-free telemetry, and request-byte cap.
- Added real-Uvicorn concurrent latency and version-consistency release gate.
- Added Windows/Python compatibility matrix, workflow concurrency, CodeQL v4, and Dependabot.
- Excluded active lock and ephemeral coverage state from release hashing.
- Preserved release coverage evidence across root pipeline cleanup.
- Expanded the suite to 108 tests with 75.70% source coverage.


## 0.4.0

- Added deterministic HSG controls, run provenance, and same-seed replay verification.
- Added runtime dependency compatibility checks and deployment bundle/digest pins.
- Added strict batch resource limits and incompatible serving-bundle pruning.
- Isolated optional torch imports from the classical/runtime dependency path.
- Added tested Python 3.11 constraints, CycloneDX SBOM, and tag attestation workflow.
- Added reproducible wheel and normalized-sdist verification.
- Added sdist traversal, link, duplicate-member, and permission hardening.
- Added Ruff format and 70% source coverage gates.
- Expanded numerical, CLI, I/O, publication, and repository-delivery tests.
- Hardened release subprocess supervision and native-library process termination.

## 0.3.0

- Added immutable serving bundles, atomic pointer switching, strict request contracts, wheel/sdist
  verification, predictive/decision release floors, and complete local release orchestration.
