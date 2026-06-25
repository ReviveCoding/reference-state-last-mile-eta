from __future__ import annotations

from pathlib import Path

from reference_eta.provenance.run import build_run_provenance


def test_run_provenance_uses_portable_repository_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    config = root / "config.yaml"
    data = root / "data.csv"
    config.write_text("seed: 1\n", encoding="utf-8")
    data.write_text("x\n1\n", encoding="utf-8")
    result = build_run_provenance(
        root=root,
        config_path=config,
        seed=1,
        data_source="lade_normalized",
        data_path=data,
        deterministic_requested=True,
    )
    assert result["provenance_schema_version"] == 2
    assert result["project"]["name"] == "reference-state-last-mile-eta"
    assert result["project"]["version"]
    assert result["config"]["path"] == "config.yaml"
    assert result["data"]["path"] == "data.csv"
    assert result["config"]["sha256"]
    assert result["data"]["sha256"]
