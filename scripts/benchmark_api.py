from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reference_eta.io import atomic_write_json  # noqa: E402
from reference_eta.process_control import (  # noqa: E402
    popen_group_kwargs,
    terminate_process_tree,
)

REQUEST_COLUMNS = [
    "city",
    "query_hour",
    "elapsed_minutes",
    "completed_task_count",
    "remaining_task_count",
    "initial_task_count",
    "completed_workload",
    "remaining_workload",
    "initial_workload",
    "observed_progress",
    "route_phase",
    "recent_pace",
    "task_density",
    "remaining_spread",
    "aoi_transition_burden",
    "weather_severity",
    "congestion_proxy",
    "trajectory_missingness",
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_payload() -> dict[str, object]:
    source = ROOT / "artifacts" / "data" / "test_snapshots.csv"
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("API benchmark input is empty")
    missing = set(REQUEST_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"API benchmark input is missing columns: {sorted(missing)}")
    payload: dict[str, object] = {}
    for column in REQUEST_COLUMNS:
        value = frame.iloc[0][column]
        payload[column] = value.item() if hasattr(value, "item") else value
    return payload


async def _wait_ready(base_url: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/ready")
                if response.status_code == 200:
                    return response.json()
            except Exception as exc:  # network startup boundary
                last_error = exc
            await asyncio.sleep(0.1)
    raise TimeoutError(f"API did not become ready: {last_error}")


async def _wait_workers(base_url: str, expected_workers: int, timeout_seconds: float) -> list[int]:
    if expected_workers <= 1:
        return []
    deadline = time.monotonic() + timeout_seconds
    observed: set[int] = set()
    limits = httpx.Limits(max_connections=4, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=2.0, limits=limits) as client:
        while time.monotonic() < deadline and len(observed) < expected_workers:
            responses = await asyncio.gather(
                *(
                    client.get(
                        f"{base_url}/live",
                        headers={"Connection": "close"},
                    )
                    for _ in range(max(4, expected_workers * 2))
                ),
                return_exceptions=True,
            )
            for response in responses:
                if isinstance(response, Exception):
                    continue
                worker_pid = response.headers.get("X-Worker-PID", "")
                if response.status_code == 200 and worker_pid.isdigit():
                    observed.add(int(worker_pid))
            if len(observed) < expected_workers:
                await asyncio.sleep(0.1)
    if len(observed) < expected_workers:
        raise TimeoutError(
            f"Observed {len(observed)} of {expected_workers} configured workers: {sorted(observed)}"
        )
    return sorted(observed)


async def _run_load(
    base_url: str,
    payload: dict[str, object],
    *,
    requests: int,
    concurrency: int,
) -> dict[str, object]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    statuses: list[int] = []
    bundle_ids: set[str] = set()
    model_versions: set[str] = set()
    worker_pids: set[int] = set()
    prediction_signatures: set[tuple[float, float, float]] = set()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=10.0, limits=limits) as client:

        async def one(index: int) -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    f"{base_url}/v1/eta/predict",
                    json=payload,
                    headers={
                        "X-Request-ID": f"benchmark-{index}",
                        "Connection": "close",
                    },
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                statuses.append(response.status_code)
                bundle_ids.add(response.headers.get("X-Bundle-ID", "missing"))
                model_versions.add(response.headers.get("X-Model-Version", "missing"))
                worker_pid = response.headers.get("X-Worker-PID", "")
                if worker_pid.isdigit():
                    worker_pids.add(int(worker_pid))
                if response.status_code == 200:
                    body = response.json()
                    prediction_signatures.add(
                        (float(body["q10"]), float(body["q50"]), float(body["q90"]))
                    )

        await asyncio.gather(*(one(index) for index in range(requests)))

    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))
    errors = sum(status != 200 for status in statuses)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "http_errors": errors,
        "error_rate": errors / max(requests, 1),
        "latency_ms_p50": median(ordered),
        "latency_ms_p95": ordered[p95_index],
        "latency_ms_max": max(ordered),
        "bundle_ids": sorted(bundle_ids),
        "model_versions": sorted(model_versions),
        "worker_pids": sorted(worker_pids),
        "unique_prediction_signatures": len(prediction_signatures),
    }


def _benchmark_passes(result: dict[str, object], *, max_p95_ms: float, workers: int) -> bool:
    minimum_workers_seen = 1 if workers == 1 else 2
    return bool(
        int(result["http_errors"]) == 0
        and len(result["bundle_ids"]) == 1  # type: ignore[arg-type]
        and len(result["model_versions"]) == 1  # type: ignore[arg-type]
        and len(result["worker_pids"]) >= minimum_workers_seen  # type: ignore[arg-type]
        and len(result.get("ready_worker_pids", [])) == workers  # type: ignore[arg-type]
        and int(result["unique_prediction_signatures"]) == 1
        and float(result["latency_ms_p95"]) <= max_p95_ms
    )


async def benchmark(
    requests: int,
    concurrency: int,
    startup_timeout: float,
    max_p95_ms: float,
    workers: int,
) -> dict[str, object]:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env.update(
        {
            "PYTHONPATH": str(SRC)
            if not existing_pythonpath
            else os.pathsep.join([str(SRC), existing_pythonpath]),
            "REFERENCE_ETA_ARTIFACT_DIR": str(ROOT / "artifacts"),
            "REFERENCE_ETA_LOG_REQUESTS": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    descriptor, log_name = tempfile.mkstemp(prefix="reference_eta_uvicorn_", suffix=".log")
    os.close(descriptor)
    log_path = Path(log_name)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "reference_eta.serving.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            **popen_group_kwargs(),
        )
    result: dict[str, object] | None = None
    ready: dict[str, object] | None = None
    try:
        ready = await _wait_ready(base_url, startup_timeout)
        ready_worker_pids = await _wait_workers(base_url, workers, startup_timeout)
        result = await _run_load(
            base_url,
            _request_payload(),
            requests=requests,
            concurrency=concurrency,
        )
    finally:
        shutdown_started = time.perf_counter()
        terminate_process_tree(process, grace_seconds=15.0)
        shutdown_seconds = time.perf_counter() - shutdown_started
        shutdown_return_code = process.poll()
        if result is not None and ready is not None:
            log_path.unlink(missing_ok=True)

    if result is None or ready is None:
        log_tail = (
            log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            if log_path.exists()
            else ""
        )
        log_path.unlink(missing_ok=True)
        raise RuntimeError(f"API benchmark did not produce a result. Uvicorn log tail:\n{log_tail}")

    graceful_shutdown = shutdown_return_code == 0 or (
        os.name == "nt" and shutdown_return_code in {0, 1}
    )
    result.update(
        {
            "benchmark_schema_version": 2,
            "ready": ready,
            "max_p95_ms": max_p95_ms,
            "configured_workers": workers,
            "ready_worker_pids": ready_worker_pids,
            "graceful_shutdown": graceful_shutdown,
            "shutdown_return_code": shutdown_return_code,
            "shutdown_seconds": shutdown_seconds,
        }
    )
    result["status"] = (
        "PASS"
        if _benchmark_passes(result, max_p95_ms=max_p95_ms, workers=workers) and graceful_shutdown
        else "FAIL"
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"API benchmark failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real-Uvicorn concurrent API smoke benchmark"
    )
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--startup-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-p95-ms", type=float, default=1000.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "api_concurrency_benchmark.json",
    )
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.concurrency > args.requests:
        raise SystemExit("requests and concurrency must satisfy 1 <= concurrency <= requests")
    if not np.isfinite(args.max_p95_ms) or args.max_p95_ms <= 0.0:
        raise SystemExit("--max-p95-ms must be finite and positive")
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    result = asyncio.run(
        benchmark(
            args.requests,
            args.concurrency,
            args.startup_timeout_seconds,
            args.max_p95_ms,
            args.workers,
        )
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
