from __future__ import annotations

import pytest

import reference_eta.serving.api as api


def _provenance() -> dict[str, object]:
    return {
        "provenance_schema_version": 2,
        "project": {
            "name": "reference-state-last-mile-eta",
            "version": api.__version__,
        },
        "runtime": {
            "packages": {
                "scikit-learn": api.version("scikit-learn"),
                "lightgbm": api.version("lightgbm"),
                "joblib": api.version("joblib"),
            }
        },
    }


def test_runtime_compatibility_accepts_exact_training_versions(monkeypatch) -> None:
    assert api._validate_runtime_compatibility(_provenance()) == api.__version__
    value = _provenance()
    value["runtime"]["packages"]["joblib"] = "0.0.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="Runtime package mismatch"):
        api._validate_runtime_compatibility(value)
    monkeypatch.setenv("REFERENCE_ETA_ALLOW_VERSION_MISMATCH", "1")
    api._validate_runtime_compatibility(value)


def test_runtime_compatibility_rejects_artifact_code_version_mismatch(monkeypatch) -> None:
    value = _provenance()
    value["project"]["version"] = "0.0.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="Artifact/code version mismatch"):
        api._validate_runtime_compatibility(value)
    monkeypatch.setenv("REFERENCE_ETA_ALLOW_ARTIFACT_VERSION_MISMATCH", "1")
    assert api._validate_runtime_compatibility(value) == "0.0.0"
