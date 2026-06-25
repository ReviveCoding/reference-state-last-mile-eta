FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY constraints ./constraints
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY sql ./sql

RUN python -m pip install -c constraints/ci-py311.txt setuptools wheel \
    && python -m pip wheel --no-build-isolation -c constraints/ci-py311.txt . --wheel-dir /wheelhouse \
    && python -m pip install --no-index --find-links=/wheelhouse reference-state-last-mile-eta \
    && python scripts/run_pipeline.py \
       --config configs/smoke.yaml --mode smoke --require-release-pass

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    REFERENCE_ETA_ARTIFACT_DIR=/app/artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheelhouse /wheelhouse
RUN python -m pip install --no-index --find-links=/wheelhouse reference-state-last-mile-eta \
    && rm -rf /wheelhouse

WORKDIR /app
COPY --from=builder /app/artifacts ./artifacts

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/ready')); raise SystemExit(0 if d.get('status')=='ready' else 1)"
CMD ["python", "-m", "uvicorn", "reference_eta.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
