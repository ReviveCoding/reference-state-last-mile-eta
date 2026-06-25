from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from reference_eta.serving.api import app

ROOT = Path(__file__).resolve().parents[1]
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


def _native(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return,union-attr]
    return value


def main() -> None:
    source = ROOT / "artifacts" / "data" / "test_snapshots.csv"
    if not source.is_file():
        raise SystemExit(f"Missing runtime verification data: {source}")
    frame = pd.read_csv(source)
    missing = set(REQUEST_COLUMNS).difference(frame.columns)
    if missing or frame.empty:
        raise SystemExit(f"Runtime verification data are invalid; missing={sorted(missing)}")
    request = {column: _native(frame.iloc[0][column]) for column in REQUEST_COLUMNS}
    with TestClient(app) as client:
        expected_statuses = {
            "/live": 200,
            "/ready": 200,
            "/health": 200,
        }
        responses = {}
        for endpoint, status in expected_statuses.items():
            response = client.get(endpoint)
            if response.status_code != status:
                raise SystemExit(f"{endpoint} failed: {response.status_code} {response.text}")
            responses[endpoint] = response

        single = client.post("/v1/eta/predict", json=request)
        batch = client.post("/v1/eta/batch", json={"snapshots": [request, request]})
        triage = client.post(
            "/v1/triage/rank", json={"snapshots": [request, request], "capacity": 0.5}
        )
        for name, response in (("predict", single), ("batch", batch), ("triage", triage)):
            if response.status_code != 200:
                raise SystemExit(f"{name} failed: {response.status_code} {response.text}")

        health = responses["/health"].json()
        model_version = str(health.get("model_version", ""))
        bundle_id = single.headers.get("X-Bundle-ID", "")
        if not model_version or single.headers.get("X-Model-Version") != model_version:
            raise SystemExit("Prediction header model version does not match loaded artifact")
        if not bundle_id or batch.headers.get("X-Bundle-ID") != bundle_id:
            raise SystemExit("Single and batch requests did not use the same serving bundle")
        worker_pid = single.headers.get("X-Worker-PID", "")
        if not worker_pid.isdigit():
            raise SystemExit("Prediction response is missing a valid worker PID")

        result = {
            "health": health,
            "prediction_quantiles": {key: single.json()[key] for key in ("q10", "q50", "q90")},
            "batch_size": len(batch.json()["predictions"]),
            "triage_selected": triage.json()["selected_count"],
            "bundle_id": bundle_id,
            "model_version": model_version,
            "worker_pid": int(worker_pid),
            "lifespan_preload": app.state.startup_error is None,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
