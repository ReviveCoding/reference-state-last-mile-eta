from __future__ import annotations

import json
from pathlib import Path

from reference_eta import __version__
from scripts.run_pipeline import _publish_serving_bundle, _serving_bundle_is_complete

REQUIRED = (
    "rcot.joblib",
    "quantile_champion.joblib",
    "cqr_calibrator.joblib",
    "tail_thresholds.joblib",
    "release_decision.json",
    "run_provenance.json",
)


def _write_source_artifacts(root: Path) -> None:
    for index, name in enumerate(REQUIRED):
        if name == "run_provenance.json":
            (root / name).write_text(
                json.dumps(
                    {
                        "provenance_schema_version": 2,
                        "project": {
                            "name": "reference-state-last-mile-eta",
                            "version": __version__,
                        },
                    }
                ),
                encoding="utf-8",
            )
        else:
            (root / name).write_bytes(f"payload-{index}".encode())


def test_publication_prunes_incompatible_legacy_bundle(tmp_path: Path) -> None:
    _write_source_artifacts(tmp_path)
    legacy = tmp_path / "serving_bundles" / ("a" * 20)
    legacy.mkdir(parents=True)
    (legacy / "old.joblib").write_bytes(b"old")
    (legacy / "artifact_manifest.json").write_text(
        json.dumps([{"path": "old.joblib", "size_bytes": 3, "sha256": "0" * 64}]),
        encoding="utf-8",
    )

    pointer = _publish_serving_bundle(tmp_path)
    current = tmp_path / "serving_bundles" / str(pointer["bundle_id"])
    assert current.is_dir()
    assert _serving_bundle_is_complete(current, REQUIRED)
    assert not legacy.exists()


def test_bundle_completeness_rejects_missing_provenance(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    records = []
    for name in REQUIRED[:-1]:
        path = bundle / name
        path.write_bytes(b"x")
        records.append({"path": name, "size_bytes": 1, "sha256": "0" * 64})
    (bundle / "artifact_manifest.json").write_text(json.dumps(records), encoding="utf-8")
    assert not _serving_bundle_is_complete(bundle, REQUIRED)


def test_publish_failure_does_not_switch_current_pointer(tmp_path: Path) -> None:
    _write_source_artifacts(tmp_path)
    first = _publish_serving_bundle(tmp_path)
    original_pointer = (tmp_path / "current_bundle.json").read_text(encoding="utf-8")
    (tmp_path / "release_decision.json").write_bytes(b"new-release")

    import pytest

    with pytest.raises(RuntimeError, match="Injected publish failure"):
        _publish_serving_bundle(tmp_path, fault_stage="after_bundle_before_pointer")

    assert (tmp_path / "current_bundle.json").read_text(encoding="utf-8") == original_pointer
    assert json.loads(original_pointer)["bundle_id"] == first["bundle_id"]
    assert not list((tmp_path / "serving_bundles").glob(".*.tmp"))


def test_publish_lock_rejects_concurrent_writer(monkeypatch, tmp_path: Path) -> None:
    import pytest

    from reference_eta.locking import ExclusiveFileLock, LockTimeoutError

    _write_source_artifacts(tmp_path)
    held = ExclusiveFileLock(tmp_path / ".locks" / "publish.lock", purpose="held").acquire()
    monkeypatch.setenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "0")
    try:
        with pytest.raises(LockTimeoutError):
            _publish_serving_bundle(tmp_path)
    finally:
        held.release()


def test_bundle_completeness_rejects_old_provenance_schema(tmp_path: Path) -> None:
    _write_source_artifacts(tmp_path)
    (tmp_path / "run_provenance.json").write_text(
        json.dumps({"provenance_schema_version": 1}), encoding="utf-8"
    )
    import pytest

    with pytest.raises(ValueError, match="incompatible run provenance"):
        _publish_serving_bundle(tmp_path)
