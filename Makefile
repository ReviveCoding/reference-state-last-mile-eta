PYTHON ?= python
CONFIG ?= configs/smoke.yaml
GPU_CONFIG ?= configs/gpu_smoke.yaml
DIST_DIR ?= dist
THREAD_ENV ?= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
BUILD_ENV ?= SOURCE_DATE_EPOCH=1704067200

.PHONY: install install-dev install-repro smoke lade-smoke amazon-smoke train-baselines train-gpu advanced-smoke evaluate simulate serve test lint package-check verify-manifest release-manifest release-preflight release-bootstrap release clean repro-check sbom coverage api-benchmark locking-check typecheck candidate-handoff clean-candidate

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

install-repro:
	$(PYTHON) -m pip install -c constraints/ci-py311.txt -e ".[dev]"

smoke:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --mode smoke --require-release-pass --force-process-exit

lade-smoke:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/generate_lade_sample.py --output artifacts/sample_lade_normalized.csv
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py --config configs/lade_smoke.yaml --mode smoke --output-namespace lade_smoke --require-release-pass --force-process-exit

amazon-smoke:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/generate_amazon_sample.py --output-dir artifacts/amazon_sample
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_amazon_replay.py --input-dir artifacts/amazon_sample --output reports/amazon_official_shape_replay.csv

train-baselines:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --mode baselines --require-release-pass --force-process-exit

train-gpu:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m scripts.train_hsg_eta --config $(GPU_CONFIG) --require-release-pass --force-process-exit

advanced-smoke: train-gpu

evaluate:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --mode evaluate --require-release-pass --force-process-exit

simulate:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --mode simulate --require-release-pass --force-process-exit

serve:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) -m uvicorn reference_eta.serving.api:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=reference_eta --cov-report=term-missing:skip-covered --cov-report=json:reports/coverage.json --cov-fail-under=70

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m ruff format --check src scripts tests
	$(PYTHON) -m compileall -q src scripts


typecheck:
	$(PYTHON) -m mypy src/reference_eta

package-check:
	rm -rf $(DIST_DIR)
	$(BUILD_ENV) $(PYTHON) -m build --no-isolation --sdist --wheel --outdir $(DIST_DIR) .
	$(BUILD_ENV) $(PYTHON) -m scripts.normalize_sdist --dist-dir $(DIST_DIR)
	$(PYTHON) scripts/verify_distribution.py --dist-dir $(DIST_DIR)
	$(BUILD_ENV) PYTHONPATH=src:. $(PYTHON) -m scripts.verify_build_reproducibility --dist-dir $(DIST_DIR) --output reports/distribution_reproducibility.json
	$(PYTHON) scripts/clean_build_metadata.py

repro-check:
	$(THREAD_ENV) PYTHONPATH=src:. $(PYTHON) scripts/verify_reproducibility.py --config $(GPU_CONFIG) --mode gpu


sbom:
	PYTHONPATH=src $(PYTHON) scripts/generate_sbom.py --output reports/sbom.cdx.json

api-benchmark:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/benchmark_api.py --requests 64 --concurrency 8 --max-p95-ms 1000 --output reports/api_concurrency_benchmark.json

locking-check:
	$(THREAD_ENV) PYTHONPATH=src:. $(PYTHON) scripts/verify_locking.py --output reports/locking_recovery_report.json

verify-manifest:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/verify_artifact_manifest.py --all

release-manifest:
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/build_release_manifest.py
	$(THREAD_ENV) PYTHONPATH=src $(PYTHON) scripts/verify_artifact_manifest.py --manifest artifacts/release_manifest.json

clean-candidate:
	$(THREAD_ENV) PYTHONPATH=src:. $(PYTHON) scripts/verify_clean_candidate.py

candidate-handoff:
	$(THREAD_ENV) PYTHONPATH=src:. $(PYTHON) scripts/build_release_candidate_handoff.py

release-preflight:
	$(PYTHON) scripts/release.py --preflight-only

release-bootstrap:
	$(PYTHON) -m pip install -c constraints/ci-py311.txt -e ".[dev]"
	$(MAKE) release

release:
	$(THREAD_ENV) CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src $(PYTHON) scripts/release.py

clean:
	rm -rf artifacts/* artifacts/.locks reports/* .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info src/*.egg-info
