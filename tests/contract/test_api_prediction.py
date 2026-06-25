import json

import pandas as pd
from fastapi.testclient import TestClient

import reference_eta.serving.api as api
from reference_eta.serving.api import ArtifactBundle, app


class _FakeRCOT:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["rcot_minutes"] = output["elapsed_minutes"]
        output["progress_gap"] = 0.0
        output["pace_ratio"] = 1.0
        output["reference_support"] = 0.8
        output["reference_support_groups"] = 20.0
        output["reference_support_rows"] = 80.0
        output["reference_dispersion"] = 5.0
        output["reference_ood_probability"] = 0.0
        output["rcot_trust"] = 0.7
        output["reference_level"] = "global"
        output["reference_regime"] = "dense_service"
        return output


class _FakeModel:
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"q10": [20.0] * len(frame), "q50": [30.0] * len(frame), "q90": [45.0] * len(frame)}
        )


class _FakeCalibrator:
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()


def _request() -> dict[str, object]:
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


def test_prediction_batch_and_triage_contract(monkeypatch) -> None:
    bundle = ArtifactBundle(
        rcot=_FakeRCOT(),
        model=_FakeModel(),
        calibrator=_FakeCalibrator(),
        thresholds={("Boston", 1): 35.0, ("__global__", -1): 35.0},
        release_decision={
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: bundle)
    client = TestClient(app)

    single = client.post("/v1/eta/predict", json=_request())
    assert single.status_code == 200
    assert single.json()["q10"] <= single.json()["q50"] <= single.json()["q90"]
    assert 0.0 <= single.json()["tail_probability"] <= 1.0

    batch = client.post("/v1/eta/batch", json={"snapshots": [_request(), _request()]})
    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2

    triage = client.post(
        "/v1/triage/rank",
        json={"snapshots": [_request(), _request()], "capacity": 0.5},
    )
    assert triage.status_code == 200
    assert triage.json()["selected_count"] == 1
    assert triage.json()["decision_policy"] == "tail_risk"


def test_inconsistent_snapshot_is_rejected() -> None:
    payload = _request()
    payload["initial_task_count"] = 99
    response = TestClient(app).post("/v1/eta/predict", json=payload)
    assert response.status_code == 422


def test_numeric_strings_are_rejected_by_strict_request_contract() -> None:
    payload = _request()
    payload["elapsed_minutes"] = "60.0"
    response = TestClient(app).post("/v1/eta/predict", json=payload)
    assert response.status_code == 422


def test_batch_limit_returns_413(monkeypatch) -> None:
    bundle = ArtifactBundle(
        rcot=_FakeRCOT(),
        model=_FakeModel(),
        calibrator=_FakeCalibrator(),
        thresholds={("__global__", -1): 35.0},
        release_decision={
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: bundle)
    monkeypatch.setenv("REFERENCE_ETA_MAX_BATCH_SIZE", "1")
    response = TestClient(app).post("/v1/eta/batch", json={"snapshots": [_request(), _request()]})
    assert response.status_code == 413


def test_request_body_size_limit_returns_413(monkeypatch) -> None:
    bundle = ArtifactBundle(
        rcot=_FakeRCOT(),
        model=_FakeModel(),
        calibrator=_FakeCalibrator(),
        thresholds={("__global__", -1): 35.0},
        release_decision={
            "release_gate": "PASS_BASELINE_CHAMPION",
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
        bundle_id="a" * 20,
        manifest_sha256="b" * 64,
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: bundle)
    monkeypatch.setenv("REFERENCE_ETA_MAX_REQUEST_BYTES", "1024")
    response = TestClient(app).post(
        "/v1/eta/predict",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": "2048"},
    )
    assert response.status_code == 413


def test_invalid_request_size_configuration_returns_503(monkeypatch) -> None:
    monkeypatch.setenv("REFERENCE_ETA_MAX_REQUEST_BYTES", "not-an-integer")
    response = TestClient(app).post(
        "/v1/eta/predict",
        content=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": "2"},
    )
    assert response.status_code == 503


def test_prediction_and_headers_report_artifact_model_version(monkeypatch) -> None:
    bundle = ArtifactBundle(
        rcot=_FakeRCOT(),
        model=_FakeModel(),
        calibrator=_FakeCalibrator(),
        thresholds={("__global__", -1): 35.0},
        release_decision={
            "release_gate": "PASS_BASELINE_CHAMPION",
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
        bundle_id="a" * 20,
        manifest_sha256="b" * 64,
        model_version="9.9.9",
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: bundle)
    response = TestClient(app).post("/v1/eta/predict", json=_request())
    assert response.status_code == 200
    assert response.json()["model_version"] == "9.9.9"
    assert response.headers["X-Model-Version"] == "9.9.9"
    assert int(response.headers["X-Worker-PID"]) > 0


def test_chunked_request_body_cannot_bypass_byte_limit(monkeypatch) -> None:
    bundle = ArtifactBundle(
        rcot=_FakeRCOT(),
        model=_FakeModel(),
        calibrator=_FakeCalibrator(),
        thresholds={("__global__", -1): 35.0},
        release_decision={
            "release_gate": "PASS_BASELINE_CHAMPION",
            "quantile_champion": "without_rcot",
            "rcot_promotion_gate": "HOLD",
            "decision_champion": "tail_risk",
            "serving_policy": {},
        },
    )
    monkeypatch.setattr(api, "_load_artifacts", lambda: bundle)
    monkeypatch.setenv("REFERENCE_ETA_MAX_REQUEST_BYTES", "1024")
    encoded = json.dumps(_request()).encode("utf-8")

    def chunks():  # noqa: ANN202
        yield encoded
        yield b" " * 700

    response = TestClient(app).post(
        "/v1/eta/predict",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
