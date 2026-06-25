from pathlib import Path

from scripts.run_pipeline import _clean_run_outputs, _run_manifest_files


def test_run_manifest_excludes_mutable_release_log(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    data = artifacts / "data"
    artifacts.mkdir()
    reports.mkdir()
    data.mkdir()
    (artifacts / "model.joblib").write_bytes(b"model")
    (reports / "metrics.csv").write_text("metric,value\nmae,1\n", encoding="utf-8")
    (reports / "final_release_log.txt").write_text("mutable", encoding="utf-8")

    files = _run_manifest_files(artifacts, reports, data, namespaced=False)

    assert artifacts / "model.joblib" in files
    assert reports / "metrics.csv" in files
    assert reports / "final_release_log.txt" not in files


def test_root_cleanup_preserves_release_orchestration_evidence(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    (artifacts / "stale.bin").write_bytes(b"stale")
    (reports / "stale.csv").write_text("stale", encoding="utf-8")
    (reports / "final_release_log.txt").write_text("release", encoding="utf-8")
    (reports / "coverage.json").write_text("{}", encoding="utf-8")

    _clean_run_outputs(artifacts, reports, namespaced=False)

    assert not (artifacts / "stale.bin").exists()
    assert not (reports / "stale.csv").exists()
    assert (reports / "final_release_log.txt").read_text(encoding="utf-8") == "release"
    assert (reports / "coverage.json").read_text(encoding="utf-8") == "{}"


def test_run_manifest_excludes_coverage_report(tmp_path):
    from scripts.run_pipeline import _run_manifest_files

    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    data = tmp_path / "data"
    for directory in [artifacts, reports, data]:
        directory.mkdir()
    (reports / "coverage.json").write_text("{}", encoding="utf-8")
    (reports / "run_report.md").write_text("ok", encoding="utf-8")
    (artifacts / "release_decision.json").write_text("{}", encoding="utf-8")
    files = _run_manifest_files(artifacts, reports, data, namespaced=False)
    names = {path.name for path in files}
    assert "coverage.json" not in names
    assert {"run_report.md", "release_decision.json"}.issubset(names)
