from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reference_eta.process_control import (  # noqa: E402
    popen_group_kwargs,
    terminate_process_tree,
)

THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "SOURCE_DATE_EPOCH": "1704067200",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(THREAD_DEFAULTS)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not current else os.pathsep.join([str(SRC), current])
    return env


def _serve_workers() -> int:
    raw = os.getenv("REFERENCE_ETA_WORKERS", "1")
    try:
        workers = int(raw)
    except ValueError as error:
        raise ValueError("REFERENCE_ETA_WORKERS must be an integer") from error
    if not 1 <= workers <= 8:
        raise ValueError("REFERENCE_ETA_WORKERS must be between 1 and 8")
    return workers


def _run(command: Command) -> None:
    print(f"== {command.name} ==", flush=True)
    print("$ " + subprocess.list2cmdline(command.argv), flush=True)
    process = subprocess.Popen(command.argv, cwd=ROOT, env=_environment(), **popen_group_kwargs())
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        terminate_process_tree(process)
        raise SystemExit(130) from None
    if return_code != 0:
        raise SystemExit(f"Task step failed ({return_code}): {command.name}")


def _commands(task: str) -> list[Command]:
    py = sys.executable
    mapping: dict[str, list[Command]] = {
        "lint": [
            Command("ruff", [py, "-m", "ruff", "check", "src", "scripts", "tests"]),
            Command(
                "ruff-format",
                [py, "-m", "ruff", "format", "--check", "src", "scripts", "tests"],
            ),
            Command("compileall", [py, "-m", "compileall", "-q", "src", "scripts"]),
        ],
        "test": [Command("pytest", [py, "-m", "pytest"])],
        "typecheck": [Command("mypy", [py, "-m", "mypy", "src/reference_eta"])],
        "coverage": [
            Command(
                "coverage",
                [
                    py,
                    "-m",
                    "pytest",
                    "--cov=reference_eta",
                    "--cov-report=term-missing:skip-covered",
                    "--cov-report=json:reports/coverage.json",
                    "--cov-fail-under=70",
                ],
            )
        ],
        "smoke": [
            Command(
                "synthetic-smoke",
                [
                    py,
                    "scripts/run_pipeline.py",
                    "--config",
                    "configs/smoke.yaml",
                    "--mode",
                    "smoke",
                    "--require-release-pass",
                    "--force-process-exit",
                ],
            )
        ],
        "train-gpu": [
            Command(
                "compact-hsg",
                [
                    py,
                    "-m",
                    "scripts.train_hsg_eta",
                    "--config",
                    "configs/gpu_smoke.yaml",
                    "--require-release-pass",
                    "--force-process-exit",
                ],
            )
        ],
        "lade-smoke": [
            Command(
                "generate-lade-sample",
                [
                    py,
                    "scripts/generate_lade_sample.py",
                    "--output",
                    "artifacts/sample_lade_normalized.csv",
                ],
            ),
            Command(
                "lade-smoke",
                [
                    py,
                    "scripts/run_pipeline.py",
                    "--config",
                    "configs/lade_smoke.yaml",
                    "--mode",
                    "smoke",
                    "--output-namespace",
                    "lade_smoke",
                    "--require-release-pass",
                    "--force-process-exit",
                ],
            ),
        ],
        "amazon-smoke": [
            Command(
                "generate-amazon-sample",
                [
                    py,
                    "scripts/generate_amazon_sample.py",
                    "--output-dir",
                    "artifacts/amazon_sample",
                ],
            ),
            Command(
                "amazon-replay",
                [
                    py,
                    "scripts/run_amazon_replay.py",
                    "--input-dir",
                    "artifacts/amazon_sample",
                    "--output",
                    "reports/amazon_official_shape_replay.csv",
                ],
            ),
        ],
        "repro-check": [
            Command(
                "deterministic-replay",
                [
                    py,
                    "scripts/verify_reproducibility.py",
                    "--config",
                    "configs/gpu_smoke.yaml",
                    "--mode",
                    "gpu",
                ],
            )
        ],
        "sbom": [
            Command(
                "sbom",
                [py, "scripts/generate_sbom.py", "--output", "reports/sbom.cdx.json"],
            )
        ],
        "api-benchmark": [
            Command(
                "concurrent-api-benchmark",
                [
                    py,
                    "scripts/benchmark_api.py",
                    "--requests",
                    "64",
                    "--concurrency",
                    "8",
                    "--max-p95-ms",
                    "1000",
                    "--output",
                    "reports/api_concurrency_benchmark.json",
                ],
            )
        ],
        "locking-check": [
            Command(
                "locking-and-publish-recovery",
                [
                    py,
                    "scripts/verify_locking.py",
                    "--output",
                    "reports/locking_recovery_report.json",
                    "--force-process-exit",
                ],
            )
        ],
        "verify-manifest": [
            Command("verify-manifests", [py, "scripts/verify_artifact_manifest.py", "--all"])
        ],
        "clean-candidate": [Command("clean-candidate", [py, "scripts/verify_clean_candidate.py"])],
        "candidate-handoff": [
            Command("candidate-handoff", [py, "scripts/build_release_candidate_handoff.py"])
        ],
        "release-manifest": [
            Command("build-release-manifest", [py, "scripts/build_release_manifest.py"]),
            Command(
                "verify-release-manifest",
                [
                    py,
                    "scripts/verify_artifact_manifest.py",
                    "--manifest",
                    "artifacts/release_manifest.json",
                ],
            ),
        ],
        "release": [Command("release", [py, "scripts/release.py"])],
    }
    if task == "package-check":
        return [
            Command("clean-distribution", [py, "scripts/clean_distribution.py"]),
            Command(
                "distribution-build",
                [
                    py,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    "dist",
                    ".",
                ],
            ),
            Command(
                "normalize-sdist",
                [py, "-m", "scripts.normalize_sdist", "--dist-dir", "dist"],
            ),
            Command(
                "verify-distribution",
                [py, "scripts/verify_distribution.py", "--dist-dir", "dist"],
            ),
            Command(
                "reproducible-distribution",
                [
                    py,
                    "-m",
                    "scripts.verify_build_reproducibility",
                    "--dist-dir",
                    "dist",
                    "--output",
                    "reports/distribution_reproducibility.json",
                ],
            ),
            Command("clean-build-metadata", [py, "scripts/clean_build_metadata.py"]),
        ]
    if task == "serve":
        return [
            Command(
                "serve",
                [
                    py,
                    "-m",
                    "uvicorn",
                    "reference_eta.serving.api:app",
                    "--host",
                    os.getenv("REFERENCE_ETA_HOST", "127.0.0.1"),
                    "--port",
                    os.getenv("REFERENCE_ETA_PORT", "8000"),
                    "--workers",
                    str(_serve_workers()),
                ],
            )
        ]
    if task not in mapping:
        raise ValueError(f"Unsupported task: {task}")
    return mapping[task]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform project task runner")
    parser.add_argument(
        "task",
        choices=[
            "lint",
            "test",
            "typecheck",
            "coverage",
            "smoke",
            "train-gpu",
            "lade-smoke",
            "amazon-smoke",
            "repro-check",
            "sbom",
            "package-check",
            "api-benchmark",
            "locking-check",
            "verify-manifest",
            "release-manifest",
            "clean-candidate",
            "candidate-handoff",
            "release",
            "serve",
        ],
    )
    args = parser.parse_args()
    for command in _commands(args.task):
        _run(command)


if __name__ == "__main__":
    main()
