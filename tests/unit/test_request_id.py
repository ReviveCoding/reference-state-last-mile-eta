from __future__ import annotations

import re

from fastapi.testclient import TestClient

from reference_eta.serving.api import _safe_request_id, app


def test_safe_request_id_preserves_bounded_identifier() -> None:
    assert _safe_request_id("job-42.trace_1:retry") == "job-42.trace_1:retry"


def test_safe_request_id_replaces_unsafe_or_oversized_values() -> None:
    generated = _safe_request_id("contains spaces")
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    oversized = _safe_request_id("a" * 129)
    assert re.fullmatch(r"[0-9a-f]{32}", oversized)
    assert oversized != "a" * 129


def test_unsafe_request_id_is_not_echoed_or_logged_as_header_value() -> None:
    with TestClient(app) as client:
        response = client.get("/live", headers={"X-Request-ID": "unsafe request id"})
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    assert request_id != "unsafe request id"
