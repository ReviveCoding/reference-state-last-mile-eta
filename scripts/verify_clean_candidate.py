from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reference_eta import __version__
from reference_eta.io import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def _command_timeout_seconds() -> int:
    raw = os.getenv("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS", "600")
    try:
        timeout = int(raw)
    except ValueError as error:
        raise ValueError(
            "REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS must be an integer"
        ) from error
    if timeout < 60:
        raise ValueError("REFERENCE_ETA_CLEAN_COMMAND_TIMEOUT_SECONDS must be at least 60")
    return timeout


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    timeout = _command_timeout_seconds()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the release candidate from a clean copy")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "clean_candidate_validation.json",
    )
    args = parser.parse_args()
    wheel_candidates = sorted((ROOT / "dist").glob("*.whl"))
    if len(wheel_candidates) != 1:
        raise SystemExit("Clean validation requires exactly one wheel in dist/")

    with tempfile.TemporaryDirectory(prefix="reference-eta-clean-") as temporary:
        temporary_root = Path(temporary)
        checkout = temporary_root / "checkout"
        shutil.copytree(
            ROOT,
            checkout,
            ignore=shutil.ignore_patterns(
                ".git",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                "__pycache__",
                "*.pyc",
                "*.pyo",
                "*.egg-info",
                "build",
                ".coverage*",
                ".locks",
                "release_candidate_handoff.json",
                "release_candidate_handoff.json.sha256",
            ),
        )
        site = temporary_root / "site"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(site),
                str(wheel_candidates[0]),
            ],
            cwd=temporary_root,
            env=os.environ.copy(),
        )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        inherited_pythonpath = env.get("PYTHONPATH")
        clean_paths = [str(site), str(checkout / "src"), str(checkout)]
        if inherited_pythonpath:
            clean_paths.append(inherited_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(clean_paths)
        tests = _run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
            cwd=checkout,
            env=env,
        )
        manifests = _run(
            [sys.executable, "scripts/verify_artifact_manifest.py", "--all"],
            cwd=checkout,
            env=env,
        )
        # A release manifest is generated output and is not copied into the
        # temporary checkout. Rebuild it there before validating it.
        print("Rebuilding clean-checkout release manifest")
        _run(
            [sys.executable, "scripts/build_release_manifest.py"],
            cwd=checkout,
            env=env,
        )
        repository_manifest = _run(
            [
                sys.executable,
                "scripts/verify_artifact_manifest.py",
                "--manifest",
                "artifacts/release_manifest.json",
            ],
            cwd=checkout,
            env=env,
        )
        installed = _run(
            [
                sys.executable,
                "-c",
                "import reference_eta; print(reference_eta.__version__)",
            ],
            cwd=temporary_root,
            env={
                **env,
                "PYTHONPATH": os.pathsep.join(
                    [str(site)] + ([inherited_pythonpath] if inherited_pythonpath else [])
                ),
            },
        )
        matches = re.findall(r"(\d+) passed", tests.stdout + tests.stderr)
        if not matches:
            raise RuntimeError("Could not parse clean-copy pytest result")
        manifest_matches = re.findall(r"Verified (\d+) records", repository_manifest.stdout)
        if not manifest_matches:
            raise RuntimeError("Could not parse repository manifest count")
        if installed.stdout.strip() != __version__:
            raise RuntimeError("Installed-wheel version differs from source candidate version")
        report = {
            "schema_version": 1,
            "status": "PASS",
            "command": "clean-copy pytest + installed-wheel import + all manifest verification",
            "exit_code": 0,
            "tests_passed": max(int(value) for value in matches),
            "installed_version": installed.stdout.strip(),
            "run_manifest_output": manifests.stdout.strip(),
            "repository_manifest_records": int(manifest_matches[-1]),
        }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
