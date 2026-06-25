import hashlib
import json
from pathlib import Path

import pytest

import reference_eta.serving.api as api


def test_atomic_bundle_pointer_resolves_content_digest(monkeypatch, tmp_path: Path) -> None:
    bundle_id = "a" * 20
    bundle = tmp_path / "serving_bundles" / bundle_id
    bundle.mkdir(parents=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "current_bundle.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "bundle_id": bundle_id,
                "manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(
        api,
        "_load_artifacts_for_manifest",
        lambda bundle_dir, manifest_sha: (Path(bundle_dir), manifest_sha),
    )
    resolved_dir, resolved_digest = api._load_artifacts()
    assert resolved_dir == bundle.resolve()
    assert resolved_digest == digest


def test_bundle_pointer_rejects_unsafe_identifier(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "current_bundle.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "bundle_id": "../escape",
                "manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        api._load_artifacts()


def test_bundle_pointer_honors_deployment_pins(monkeypatch, tmp_path: Path) -> None:
    bundle_id = "b" * 20
    bundle = tmp_path / "serving_bundles" / bundle_id
    bundle.mkdir(parents=True)
    manifest = bundle / "artifact_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "current_bundle.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": 1,
                "bundle_id": bundle_id,
                "manifest_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(
        api,
        "_load_artifacts_for_manifest",
        lambda bundle_dir, manifest_sha: (Path(bundle_dir), manifest_sha),
    )
    monkeypatch.setenv("REFERENCE_ETA_EXPECTED_BUNDLE_ID", bundle_id)
    monkeypatch.setenv("REFERENCE_ETA_EXPECTED_MANIFEST_SHA256", digest)
    api._load_artifacts()

    monkeypatch.setenv("REFERENCE_ETA_EXPECTED_BUNDLE_ID", "c" * 20)
    with pytest.raises(ValueError, match="pinned bundle"):
        api._load_artifacts()


def test_legacy_manifest_honors_digest_pin(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(api, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(
        api,
        "_load_artifacts_for_manifest",
        lambda bundle_dir, manifest_sha: (Path(bundle_dir), manifest_sha),
    )
    monkeypatch.setenv("REFERENCE_ETA_EXPECTED_MANIFEST_SHA256", digest)
    api._load_artifacts()
    monkeypatch.setenv("REFERENCE_ETA_EXPECTED_MANIFEST_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="pinned digest"):
        api._load_artifacts()
