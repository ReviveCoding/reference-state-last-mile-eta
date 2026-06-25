from __future__ import annotations

import pytest

from scripts.verify_reproducibility import _assert_json_equal, canonical_summary


def test_canonical_summary_removes_namespace_specific_fields() -> None:
    result = canonical_summary(
        {"release_gate": "PASS", "output_namespace": "a", "serving_bundle": {}}
    )
    assert result == {"release_gate": "PASS"}


def test_json_comparison_uses_bounded_numeric_tolerance() -> None:
    _assert_json_equal({"x": [1.0]}, {"x": [1.0 + 1e-9]}, atol=1e-8)
    with pytest.raises(AssertionError):
        _assert_json_equal({"x": 1.0}, {"x": 1.1}, atol=1e-8)
