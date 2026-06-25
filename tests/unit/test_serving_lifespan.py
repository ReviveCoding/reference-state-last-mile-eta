from __future__ import annotations

import json
import logging

import pandas as pd
from fastapi.testclient import TestClient

import reference_eta.serving.api as api
from reference_eta.serving.api import ArtifactBundle, app


class _RCOT:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["rcot_minutes"] = result["elapsed_minutes"]
        result["reference_support"] = 0.8
        result["rcot_trust"] = 0.7
        return result


class _Model:
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"q10": [10.0] * len(frame), "q50": [20.0] * len(frame), "q90": [30.0] * len(frame)}
        )


class _Calibrator:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame


def _bundle() -> ArtifactBundle:
    return ArtifactBundle(
        rcot=_RCOT(),
        model=_Model(),
        calibrator=_Calibrator(),
        thresholds={("__global__", -1): 25.0},
        release_decision={
            "release_gate": "PASS_BASELINE_CHAMPION",
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
        integrity_verified=True,
        bundle_id="a" * 20,
        manifest_sha256="b" * 64,
    )


def _payload() -> dict[str, object]:
    return {
        "city": "Boston",
        "query_hour": 9.0,
        "elapsed_minutes": 60.0,
        "completed_task_count": 4,
        "remaining_task_count": 6,
        "initial_task_count": 10,
        "completed_workload": 40.0,
        "remaining_workload": 60.0,
        "initial_workload": 100.0,
        "observed_progress": 0.4,
        "route_phase": 0.4,
        "recent_pace": 0.8,
        "task_density": 1.0,
        "remaining_spread": 2.0,
        "aoi_transition_burden": 0.2,
        "weather_severity": 0.1,
        "congestion_proxy": 1.1,
        "trajectory_missingness": 0.0,
    }


def test_lifespan_preloads_bundle(monkeypatch) -> None:
    monkeypatch.setattr(api, "_load_artifacts", _bundle)
    with TestClient(app) as client:
        assert app.state.startup_error is None
        assert app.state.bundle_id == "a" * 20
        assert client.get("/ready").status_code == 200


def test_lifespan_records_nonfatal_startup_error(monkeypatch) -> None:
    def fail() -> ArtifactBundle:
        raise RuntimeError("missing")

    monkeypatch.setattr(api, "_load_artifacts", fail)
    monkeypatch.delenv("REFERENCE_ETA_FAIL_STARTUP_IF_NOT_READY", raising=False)
    with TestClient(app):
        assert "RuntimeError" in app.state.startup_error


def test_structured_log_excludes_payload_and_sets_headers(monkeypatch, caplog) -> None:
    monkeypatch.setattr(api, "_load_artifacts", _bundle)
    monkeypatch.setenv("REFERENCE_ETA_LOG_REQUESTS", "1")
    caplog.set_level(logging.INFO, logger="reference_eta.serving")
    response = TestClient(app).post(
        "/v1/eta/predict",
        json=_payload(),
        headers={"X-Request-ID": "known-request"},
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "known-request"
    assert response.headers["X-Bundle-ID"] == "a" * 20
    records = [
        json.loads(record.message) for record in caplog.records if record.message.startswith("{")
    ]
    assert records[-1]["request_id"] == "known-request"
    assert records[-1]["batch_size"] == 1
    assert "Boston" not in caplog.text


def test_ready_headers_follow_loaded_bundle_not_stale_startup_state(monkeypatch) -> None:
    current = _bundle()
    current = ArtifactBundle(
        rcot=current.rcot,
        model=current.model,
        calibrator=current.calibrator,
        thresholds=current.thresholds,
        release_decision=current.release_decision,
        integrity_verified=True,
        bundle_id="c" * 20,
        manifest_sha256="d" * 64,
        model_version="9.9.9",
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: current)
    app.state.bundle_id = "stale"
    app.state.model_version = "0.0.0"
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.headers["X-Bundle-ID"] == "c" * 20
    assert response.headers["X-Model-Version"] == "9.9.9"
    assert response.json()["model_version"] == "9.9.9"
