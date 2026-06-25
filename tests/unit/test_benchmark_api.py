from __future__ import annotations

import pytest

from scripts import benchmark_api


def test_free_port_returns_valid_ephemeral_port() -> None:
    port = benchmark_api._free_port()
    assert 1 <= port <= 65535


def test_benchmark_cli_rejects_invalid_latency_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_api.sys,
        "argv",
        ["benchmark_api.py", "--requests", "1", "--concurrency", "1", "--max-p95-ms", "0"],
    )
    with pytest.raises(SystemExit, match="finite and positive"):
        benchmark_api.main()


def test_benchmark_gate_requires_bundle_model_prediction_and_worker_consistency() -> None:
    result = {
        "http_errors": 0,
        "bundle_ids": ["bundle"],
        "model_versions": ["0.4.2"],
        "worker_pids": [100, 101],
        "ready_worker_pids": [100, 101],
        "unique_prediction_signatures": 1,
        "latency_ms_p95": 100.0,
    }
    assert benchmark_api._benchmark_passes(result, max_p95_ms=1000.0, workers=2)
    result["model_versions"] = ["0.4.1", "0.4.2"]
    assert not benchmark_api._benchmark_passes(result, max_p95_ms=1000.0, workers=2)


def test_benchmark_cli_rejects_invalid_worker_count(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark_api.sys,
        "argv",
        ["benchmark_api.py", "--requests", "1", "--concurrency", "1", "--workers", "0"],
    )
    with pytest.raises(SystemExit, match="workers"):
        benchmark_api.main()
