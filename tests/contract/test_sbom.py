from __future__ import annotations

from pathlib import Path

from packaging.utils import canonicalize_name

from scripts.generate_sbom import build_sbom


def test_sbom_contains_project_and_sorted_runtime_components() -> None:
    sbom = build_sbom(Path("pyproject.toml"), include_dev=False)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["metadata"]["component"]["name"] == "reference-state-last-mile-eta"
    names = [canonicalize_name(component["name"]) for component in sbom["components"]]
    assert names == sorted(names)
    assert "numpy" in names
    assert "lightgbm" in names
    assert sbom["dependencies"][0]["ref"].startswith("pkg:pypi/reference-state-last-mile-eta@")
    assert sbom["dependencies"][0]["dependsOn"]


def test_sbom_serial_is_deterministic_and_non_nil() -> None:
    first = build_sbom(Path("pyproject.toml"), include_dev=False)
    second = build_sbom(Path("pyproject.toml"), include_dev=False)
    assert first["serialNumber"] == second["serialNumber"]
    assert first["serialNumber"].startswith("urn:uuid:")
    assert first["serialNumber"] != "urn:uuid:00000000-0000-0000-0000-000000000000"
