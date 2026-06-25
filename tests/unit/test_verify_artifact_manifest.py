import hashlib
import json
from pathlib import Path

import pytest

import scripts.verify_artifact_manifest as verifier


def _record(path: Path, stored_path: str) -> dict[str, object]:
    return {
        "path": stored_path,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_verify_root_relative_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    artifact = tmp_path / "artifacts" / "model.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"model")
    manifest = tmp_path / "artifacts" / "artifact_manifest.json"
    manifest.write_text(json.dumps([_record(artifact, "artifacts/model.bin")]), encoding="utf-8")

    assert verifier.verify_manifest(manifest) == 1


def test_verify_serving_bundle_manifest_relative_to_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    bundle = tmp_path / "artifacts" / "serving_bundles" / "bundle123"
    bundle.mkdir(parents=True)
    artifact = bundle / "model.bin"
    artifact.write_bytes(b"model")
    manifest = bundle / "artifact_manifest.json"
    manifest.write_text(json.dumps([_record(artifact, "model.bin")]), encoding="utf-8")

    assert verifier.verify_manifest(manifest) == 1


def test_manifest_rejects_symbolic_links(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "target.txt"
    target.write_text("trusted", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": "link.txt",
                    "size_bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="symbolic links"):
        verifier.verify_manifest(manifest)
