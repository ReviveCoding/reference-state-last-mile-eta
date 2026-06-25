from fastapi.testclient import TestClient

from reference_eta.serving.api import app


def test_health_contract() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ready", "not_ready"}


def test_liveness_and_readiness_contracts() -> None:
    client = TestClient(app)
    live = client.get("/live")
    assert live.status_code == 200
    assert live.json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code in {200, 503}
