from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _project_root() -> Path:
    candidates = [
        Path(os.getenv("REFERENCE_ETA_HOME", "")).expanduser()
        if os.getenv("REFERENCE_ETA_HOME")
        else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "scripts" / "run_pipeline.py").exists():
            return candidate.resolve()
    raise SystemExit(
        "Could not locate the project checkout. Run from the repository root or set REFERENCE_ETA_HOME."
    )


def _subprocess_env(root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for name, value in THREAD_DEFAULTS.items():
        env.setdefault(name, value)
    if root is not None:
        src = str(root / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


def _run(
    command: list[str],
    root: Path | None,
    *,
    exit_on_complete: bool = True,
) -> int:
    completed = subprocess.run(
        command,
        cwd=root or Path.cwd(),
        env=_subprocess_env(root),
        check=False,
    )
    if exit_on_complete:
        raise SystemExit(completed.returncode)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-state ETA project CLI")
    parser.add_argument("command", choices=["smoke", "lade-smoke", "gpu", "test", "serve"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-namespace", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("REFERENCE_ETA_WORKERS", "1")),
    )
    args = parser.parse_args()

    # Serving works from an installed wheel. Set REFERENCE_ETA_ARTIFACT_DIR to a completed
    # artifact bundle when running outside the repository checkout.
    if args.command == "serve":
        if not 1 <= args.workers <= 8:
            raise SystemExit("--workers must be between 1 and 8")
        _run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "reference_eta.serving.api:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--workers",
                str(args.workers),
            ],
            root=None,
        )

    root = _project_root()
    if args.command == "test":
        _run([sys.executable, "-m", "pytest"], root)

    if args.command == "lade-smoke":
        sample_path = root / "artifacts" / "sample_lade_normalized.csv"
        _run(
            [
                sys.executable,
                "scripts/generate_lade_sample.py",
                "--output",
                str(sample_path),
            ],
            root,
            exit_on_complete=False,
        )

    config = args.config or (
        root / "configs/gpu_smoke.yaml"
        if args.command == "gpu"
        else root / "configs/lade_smoke.yaml"
        if args.command == "lade-smoke"
        else root / "configs/smoke.yaml"
    )
    command = [
        sys.executable,
        "scripts/run_pipeline.py",
        "--config",
        str(config),
        "--mode",
        "gpu" if args.command == "gpu" else "smoke",
        "--require-release-pass",
    ]
    namespace = args.output_namespace
    if namespace is None and args.command in {"gpu", "lade-smoke"}:
        namespace = "gpu_smoke" if args.command == "gpu" else "lade_smoke"
    if namespace:
        command.extend(["--output-namespace", namespace])
    _run(command, root)
