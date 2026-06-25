import pytest

from reference_eta.serving.api import _validate_release_decision


def _decision() -> dict[str, object]:
    return {
        "artifact_schema_version": 1,
        "release_gate": "PASS_BASELINE_CHAMPION",
        "quantile_champion": "without_rcot",
        "rcot_promotion_gate": "HOLD",
        "decision_champion": "tail_risk",
    }


def test_serving_rejects_unreleased_or_inconsistent_decision(monkeypatch) -> None:
    decision = _decision()
    _validate_release_decision(decision)

    decision["release_gate"] = "ITERATE_SYSTEM"
    with pytest.raises(ValueError, match="unreleased"):
        _validate_release_decision(decision)

    monkeypatch.setenv("REFERENCE_ETA_ALLOW_UNRELEASED", "1")
    _validate_release_decision(decision)

    decision = _decision()
    decision["quantile_champion"] = "with_rcot"
    with pytest.raises(ValueError, match="PROMOTE"):
        _validate_release_decision(decision)
