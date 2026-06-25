# Reference-State Last-Mile ETA & Reliability Decision Platform

An end-to-end Applied Science project for last-mile ETA prediction, support-aware reference-state
learning, calibrated uncertainty, and capacity-constrained operational triage.

The default path is runnable without external data. It generates deterministic courier-day events,
builds leakage-safe snapshots, trains strong classical and probabilistic baselines, evaluates RCOT as
a gated challenger, calibrates ETA intervals, selects a triage policy using validation data, writes
SQL evidence, and publishes an integrity-checked serving bundle.

Public-data adapters are included for LaDe delivery events and the Amazon Last Mile Routing
Challenge. The repository does not redistribute either dataset and never represents generated
adapter samples as real-world results.

## Core scientific question

Elapsed clock time is not a consistent measure of progress across routes with different workload,
density, movement pace, weather, and AOI-transition patterns. The project introduces
**Reference-Conditioned Operational Time (RCOT)** to align observed route progress with comparable
historical operating regimes.

RCOT is never assumed to help. It is evaluated as a challenger and emitted with:

- distinct courier-day reference support,
- reference dispersion,
- fallback cohort level,
- out-of-distribution probability,
- trust score.

A do-no-harm promotion gate retains the no-RCOT baseline whenever RCOT lacks meaningful average or
tail improvement, violates non-inferiority, or creates excessive worst-city regression.

## End-to-end system

```text
Synthetic or normalized LaDe-style task events
        ↓
Event-time snapshot builder and grouped chronological split
        ↓
Conditional median, LightGBM, and quantile LightGBM
        ↓
Forward-only temporal RCOT cross-fitting
        ↓
Dedicated conformal calibration period
        ↓
Delayed-label chronological calibration replay
        ↓
Tail probability and expected excess delay
        ↓
Validation-selected or prespecified-low-support triage policy
        ↓
SQL evidence, predictive/decision release gates
        ↓
Immutable serving bundle, manifests, API
```

A separate compact **HSG-ETA** path adds:

- courier/snapshot context encoding,
- variable-size pending-task set encoding,
- task-to-task graph messages,
- an observable-input next-stop auxiliary pointer,
- monotonic q10/q50/q90 ETA outputs,
- deterministic training controls,
- CUDA AMP support with CPU fallback.

It does not use future `actual_rank` to sort or truncate inputs. When the true next task is absent
after observable truncation, the auxiliary route loss is masked.

## Verified small-data results

These are deterministic offline validation results, not production delivery claims.

| Path | Rows | Interval coverage | Delayed-label coverage | RCOT gate | Release gate |
|---|---:|---:|---:|---|---|
| Synthetic core | 816 snapshots | 0.8780 | 0.8130 | HOLD | PASS_BASELINE_CHAMPION |
| Normalized LaDe-style sample | 200 snapshots | 0.8333 | 0.9000 | HOLD | PASS_BASELINE_CHAMPION |
| Compact HSG-ETA | 266 snapshots | 0.8333 | 0.8095 | HOLD | PASS_BASELINE_CHAMPION |

Synthetic-core point results:

| Candidate | Test MAE |
|---|---:|
| Conditional median | 15.8124 |
| LightGBM without RCOT | 5.0454 |
| LightGBM with RCOT | 5.2178 |
| Quantile q50 without RCOT | **5.0263** |
| Quantile q50 with RCOT | 5.2155 |

The validation gate selected `without_rcot` and `tail_risk`. RCOT remained a challenger because it
failed the full promotion contract. This is the intended reliability behavior.

The compact HSG path was executed on CPU in this environment. The code supports CUDA AMP, but no GPU
speed or VRAM claim is made from the bundled evidence.

## Quick start

Python 3.11 or newer is required. The cross-platform Python task runner is the canonical local entry point:

```bash
python -m pip install -e ".[dev]"
python scripts/tasks.py smoke
python scripts/tasks.py lade-smoke
python scripts/tasks.py amazon-smoke
python scripts/tasks.py train-gpu
python scripts/tasks.py coverage
```

On Windows PowerShell, no `make` installation is required:

```powershell
python -m pip install -e ".[dev]"
python scripts/tasks.py smoke
python scripts/tasks.py release
# or
.\scripts\run_release.ps1
```

The Make targets remain thin Linux/macOS aliases over the same scripts.

For the exact tested Python 3.11 dependency set:

```bash
make install-repro
```

Run the complete local release chain:

```bash
python scripts/release.py --preflight-only
python scripts/tasks.py release
```

The release task runs lint and format checks, compile validation, the coverage gate, base-without-torch
isolation, HSG/core/LaDe/Amazon paths, real-Uvicorn concurrency and latency checks, concurrent-writer
lock recovery, deterministic replay, wheel and sdist build/install checks, bitwise distribution
reproducibility, CycloneDX SBOM generation, and all artifact manifests.

Start the API after `make smoke`:

```bash
make serve
```

Endpoints:

```text
GET  /health
GET  /live
GET  /ready
POST /v1/eta/predict
POST /v1/eta/batch
POST /v1/triage/rank
```

Installed-wheel serving is supported:

```bash
export REFERENCE_ETA_ARTIFACT_DIR=/absolute/path/to/artifacts
reference-eta serve --host 127.0.0.1 --port 8000
```

Workflow commands such as `reference-eta smoke` deliberately require the checkout because they use
versioned configs and scripts. Serving does not.

## Release and serving safety

Before deserialization, serving checks:

- byte size and SHA-256 manifest,
- release-decision and provenance schemas,
- predictive and decision release gates,
- champion and serving-policy consistency,
- scikit-learn, LightGBM, and joblib runtime compatibility,
- optional deployment-pinned bundle ID and manifest digest.

Bundles are immutable and selected through an atomic pointer. Pipeline and publish operations use
cross-platform exclusive locks with owner metadata, timeout, and stale-lock recovery. The API lifespan
preloads the current bundle, uses single-flight loading per worker, keeps the previously verified in-memory
model until a complete replacement is available, and emits payload-free structured telemetry with a
bounded/sanitized request ID, artifact model version, bundle identity, worker PID, batch size, status,
and latency. Unreleased bundles are rejected by default. Request bytes are capped while streaming, including chunked bodies without `Content-Length`, and batch
size is independently resource-capped.

Useful deployment controls:

```text
REFERENCE_ETA_MAX_BATCH_SIZE
REFERENCE_ETA_MAX_REQUEST_BYTES
REFERENCE_ETA_EXPECTED_BUNDLE_ID
REFERENCE_ETA_EXPECTED_MANIFEST_SHA256
REFERENCE_ETA_ALLOW_UNRELEASED
REFERENCE_ETA_ALLOW_VERSION_MISMATCH
```

Python model serialization is trusted-only. See `SECURITY.md`.

## Reproducibility and supply-chain evidence

The repository records:

- config and input-data SHA-256,
- seed and deterministic-training request,
- runtime/package/thread environment,
- optional Git commit,
- same-seed double-run output comparison,
- reproducible wheel and normalized-sdist digests,
- CycloneDX runtime SBOM,
- repository, run, and immutable-bundle manifests.

GitHub configuration includes a full Ubuntu/Python 3.11 release job, Windows 3.11/3.13 and Ubuntu
3.13 compatibility jobs, workflow concurrency, CodeQL v4, Dependabot, Docker build, and tag-triggered
artifact attestations. Remote workflow execution is not claimed in the bundled local evidence.

## Commands

```bash
make lint              # Ruff lint, Ruff format check, compileall
make typecheck         # mypy source-package contract
make test              # unit/contract/leakage/integration tests
make coverage          # tests plus >=70% source-package coverage gate
make smoke             # deterministic classical/reliability/decision path
make lade-smoke        # normalized LaDe-style adapter path
make amazon-smoke      # official-shaped Amazon route replay path
make train-gpu         # compact HSG-ETA; CUDA AMP when available
make repro-check       # same-seed HSG/full-output replay comparison
make sbom              # deterministic CycloneDX runtime SBOM
make api-benchmark     # two-worker Uvicorn distribution, consistency, shutdown, and p95 gate
make locking-check     # cross-process lock contention and pointer-failure recovery
make package-check     # wheel/sdist install and bitwise reproducibility checks
make verify-manifest   # verify completed run and bundle manifests
make release-manifest  # build and verify repository release manifest
make clean-candidate   # test wheel/source/manifests from an isolated clean copy
make candidate-handoff # generate and verify release_candidate_handoff.json
make release           # complete validation chain including candidate handoff
make release-bootstrap # constrained dev install, then complete release
```

## Main artifacts

```text
artifacts/release_decision.json
artifacts/current_bundle.json
artifacts/artifact_manifest.json
artifacts/release_manifest.json
artifacts/operational_evidence.db
artifacts/run_provenance.json
reports/run_report.md
reports/model_scorecard.csv
reports/capacity_triage.csv
reports/rolling_calibration_replay.csv
reports/intervention_sensitivity.csv
reports/rcot_bootstrap_ci.json
reports/reproducibility_report.json
reports/distribution_reproducibility.json
reports/sbom.cdx.json
reports/coverage.json
reports/api_concurrency_benchmark.json
reports/locking_recovery_report.json
reports/amazon_official_shape_replay.csv
reports/clean_candidate_validation.json
release_candidate_handoff.json
release_candidate_handoff.json.sha256
```

## Release-candidate handoff

The canonical release creates `release_candidate_handoff.json` only after model/data pipelines,
packaging, manifests, and an isolated clean-copy check pass. The handoff records the baseline and
candidate source fingerprints, diff checksum, dependency checksums, baseline/final metrics,
artifact SHA-256 values, evidence levels, unresolved items, and the hosted qualification gates that
remain. It deliberately reports `RELEASE_CANDIDATE_NOT_QUALIFIED`; local E3 evidence must not be
presented as E4 GitHub-hosted or production qualification.

See `docs/improvement_report.md` and `docs/known_limitations.md`.

## Data roles and claim boundary

### LaDe

LaDe is the intended real-data ETA benchmark. The adapter expects normalized task-event columns and
enforces timestamp, identifier, location, package-count, and event-time contracts. Future delivery
times are labels only. The bundled sample validates mechanics, not official benchmark performance.

### Amazon Last Mile Routing Challenge

Amazon data are used only for route and time-window replay. The adapter normalizes official-shaped
route, package, travel-time, and actual-sequence JSON. It does not invent observed stop-arrival ETA
labels. The bundled sample is synthetic but structurally representative.

### Safe claims

- Built a reproducible ETA and operational-triage framework.
- Evaluated RCOT with controlled ablation, temporal cross-fitting, support slices, and promotion gates.
- Measured at-risk workload capture under fixed review capacity.
- Implemented a GPU-ready structured model and release-oriented controls.

### Claims requiring external evidence

- Full official LaDe performance and cross-city results.
- Official Amazon replay results.
- Actual NVIDIA AMP speedup or VRAM reduction.
- Executed Spark-cluster or AWS SageMaker workloads.
- Actual delay prevention or production impact.
- Successful remote GitHub Actions, attestation, or Docker build.

See `THIRD_PARTY_DATA.md`, `docs/data_claim_boundary.md`, `IMPLEMENTATION_STATUS.md`, and
`FINAL_VALIDATION_REPORT.md`.
