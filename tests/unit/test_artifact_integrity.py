import hashlib
import json
from pathlib import Path

import pytest

from reference_eta.serving.api import _verify_artifact_integrity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_integrity_accepts_matching_files_and_rejects_corruption(tmp_path: Path) -> None:
    model = tmp_path / "quantile_champion.joblib"
    release = tmp_path / "release_decision.json"
    model.write_bytes(b"model")
    release.write_text("{}", encoding="utf-8")
    paths = {"model": model, "release": release}
    manifest = tmp_path / "artifact_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "path": f"artifacts/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in paths.values()
            ]
        ),
        encoding="utf-8",
    )
    _verify_artifact_integrity(paths, manifest)
    model.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        _verify_artifact_integrity(paths, manifest)
