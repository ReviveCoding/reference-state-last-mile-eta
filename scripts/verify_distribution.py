from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify built wheel and sdist contents and importability"
    )
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    dist_dir = args.dist_dir if args.dist_dir.is_absolute() else ROOT / args.dist_dir
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(
            f"Expected exactly one wheel and one sdist in {dist_dir}; "
            f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
        )
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise SystemExit(f"Wheel must contain exactly one METADATA file: {metadata_names}")
        metadata = archive.read(metadata_names[0]).decode("utf-8")
    version_lines = [line for line in metadata.splitlines() if line.startswith("Version: ")]
    if len(version_lines) != 1:
        raise SystemExit("Wheel METADATA must contain exactly one Version field")
    expected_version = version_lines[0].split(":", 1)[1].strip()
    requires_dist = [
        line.split(":", 1)[1].strip()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    ]
    unconditional_torch = [
        requirement
        for requirement in requires_dist
        if requirement.lower().startswith("torch") and "extra ==" not in requirement.lower()
    ]
    if unconditional_torch:
        raise SystemExit(f"Torch must remain optional in wheel metadata: {unconditional_torch}")
    if not any(requirement.lower().startswith("torch") for requirement in requires_dist):
        raise SystemExit("Wheel metadata is missing the advanced/gpu torch extra")
    required_suffixes = {
        "reference_eta/__init__.py",
        "reference_eta/cli.py",
        "reference_eta/data/__init__.py",
        "reference_eta/data/lade.py",
        "reference_eta/evaluation/bootstrap.py",
        "reference_eta/models/baselines.py",
        "reference_eta/serving/api.py",
    }
    missing = [
        suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)
    ]
    if missing:
        raise SystemExit(f"Wheel is missing required package files: {missing}")
    forbidden = [name for name in names if "/tests/" in name or name.startswith("tests/")]
    if forbidden:
        raise SystemExit(f"Wheel unexpectedly contains tests: {forbidden[:5]}")

    with tarfile.open(sdists[0], "r:gz") as archive:
        sdist_names = archive.getnames()
    forbidden_suffixes = (".joblib", ".pt", ".db", ".sqlite", ".sqlite3")
    forbidden_sdist = [
        name
        for name in sdist_names
        if any(part in {"artifacts", "reports", "dist"} for part in Path(name).parts)
        or name.endswith(forbidden_suffixes)
    ]
    if forbidden_sdist:
        raise SystemExit(
            f"Sdist unexpectedly contains generated artifacts or model files: {forbidden_sdist[:5]}"
        )

    def verify_install(distribution: Path, prefix: str, *, no_build_isolation: bool) -> None:
        with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
            target = Path(temporary) / "site"
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
            ]
            if no_build_isolation:
                command.append("--no-build-isolation")
            command.append(str(distribution))
            subprocess.run(
                command,
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(target)
            verification_code = """
import importlib.abc
import json
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally blocked")
        return None

sys.meta_path.insert(0, BlockTorch())
import reference_eta
from reference_eta.data.lade import build_closed_set_snapshots
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.models.baselines import LightGBMPointModel
from reference_eta.serving.api import app
from reference_eta.serving.schemas import SnapshotRequest
print(json.dumps({
    "version": reference_eta.__version__,
    "schema": SnapshotRequest.__name__,
    "file": reference_eta.__file__,
    "app": app.title,
}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", verification_code],
                check=True,
                cwd=Path(temporary),
                env=env,
                capture_output=True,
                text=True,
            )
            result = json.loads(completed.stdout.strip())
            installed_file = Path(str(result.get("file", ""))).resolve()
            if (
                result.get("version") != expected_version
                or result.get("schema") != "SnapshotRequest"
                or result.get("app") != "Reference-State Last-Mile ETA"
                or not installed_file.is_relative_to(target.resolve())
            ):
                raise SystemExit(
                    f"Installed-distribution import verification failed for {distribution.name}: "
                    f"{result}"
                )

    verify_install(wheel, "reference-eta-wheel-", no_build_isolation=False)
    verify_install(sdists[0], "reference-eta-sdist-", no_build_isolation=True)
    print(f"Verified distribution: {wheel.name} and {sdists[0].name}")


if __name__ == "__main__":
    main()
