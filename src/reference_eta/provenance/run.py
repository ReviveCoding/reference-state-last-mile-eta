from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from reference_eta import __version__

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "joblib",
    "PyYAML",
    "fastapi",
    "uvicorn",
    "pydantic",
    "torch",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def build_run_provenance(
    *,
    root: Path,
    config_path: Path,
    seed: int,
    data_source: str,
    data_path: Path | None,
    deterministic_requested: bool,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = Path(root).resolve()
    data_record: dict[str, Any] = {"source": str(data_source)}
    if data_path is not None:
        resolved = Path(data_path).resolve()
        data_record.update(
            {
                "path": _portable_path(resolved, root),
                "exists": resolved.is_file(),
                "sha256": sha256_file(resolved) if resolved.is_file() else None,
                "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
            }
        )
    return {
        "provenance_schema_version": 2,
        "project": {
            "name": "reference-state-last-mile-eta",
            "version": __version__,
        },
        "seed": int(seed),
        "deterministic_requested": bool(deterministic_requested),
        "config": {
            "path": _portable_path(config_path, root),
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "data": data_record,
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": _package_versions(),
            "thread_environment": {
                name: os.getenv(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "PYTHONHASHSEED",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        },
        "source": {"git_commit": _git_commit(root)},
    }
