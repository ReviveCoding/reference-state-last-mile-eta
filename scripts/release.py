from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reference_eta.locking import ExclusiveFileLock  # noqa: E402
from reference_eta.process_control import (  # noqa: E402
    popen_group_kwargs,
    terminate_process_tree,
)

LOG_PATH = ROOT / "reports" / "final_release_log.txt"
THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "MALLOC_ARENA_MAX": "2",
    "SOURCE_DATE_EPOCH": "1704067200",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

REQUIRED_RELEASE_MODULES = ("build", "mypy", "packaging", "pytest", "ruff", "torch", "wheel")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    retry_on_timeout: bool = False


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in THREAD_DEFAULTS.items():
        env[key] = value
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    terminate_process_tree(process)


def _run_step(step: Step, timeout_seconds: int) -> tuple[str, float]:
    attempts = 2 if step.retry_on_timeout else 1
    ignore_interrupts = os.getenv("REFERENCE_ETA_IGNORE_INTERRUPT", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            mode="w+",
            encoding="utf-8",
            prefix="release_step_",
            suffix=".log",
            delete=False,
        ) as handle:
            output_path = Path(handle.name)
            process = subprocess.Popen(
                step.command,
                cwd=ROOT,
                env=_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_group_kwargs(),
            )

        timed_out = False
        deadline = time.monotonic() + timeout_seconds
        next_heartbeat = time.monotonic() + 10.0

        while process.poll() is None:
            now = time.monotonic()

            if now >= deadline:
                timed_out = True
                _terminate_process_group(process)
                break

            if now >= next_heartbeat:
                elapsed = time.perf_counter() - started
                print(
                    f"RUNNING: {step.name} (elapsed={elapsed:.1f}s, pid={process.pid})",
                    flush=True,
                )
                next_heartbeat = now + 10.0

            try:
                time.sleep(0.1)
            except KeyboardInterrupt as error:
                elapsed = time.perf_counter() - started

                if ignore_interrupts:
                    print(
                        f"WARN: ignored KeyboardInterrupt while {step.name} "
                        f"(elapsed={elapsed:.1f}s, pid={process.pid})",
                        flush=True,
                    )
                    continue

                _terminate_process_group(process)
                output = output_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).rstrip()
                output_path.unlink(missing_ok=True)

                raise RuntimeError(
                    f"Release interrupted while '{step.name}' after {elapsed:.1f}s\n{output}"
                ) from error

        output = output_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).rstrip()
        output_path.unlink(missing_ok=True)

        elapsed = time.perf_counter() - started

        if timed_out:
            if attempt < attempts:
                print(
                    f"RETRY: {step.name} timed out; starting attempt {attempt + 1}/{attempts}",
                    flush=True,
                )
                continue

            raise RuntimeError(
                f"Release step '{step.name}' timed out after {timeout_seconds}s\n{output}"
            )

        if process.returncode != 0:
            raise RuntimeError(
                f"Release step '{step.name}' failed with exit code {process.returncode}\n{output}"
            )

        retry_note = f" (attempt {attempt})" if attempt > 1 else ""
        return (
            f"## {step.name}{retry_note}\n$ {' '.join(step.command)}\n{output}\n",
            elapsed,
        )

    raise RuntimeError(f"Release step '{step.name}' did not run")


def _preflight() -> None:
    missing = [name for name in REQUIRED_RELEASE_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Release dependencies are missing: "
            f'{joined}. Install them with `python -m pip install -e ".[dev]"` '
            "or run `make release-bootstrap`."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete local release validation chain")
    parser.add_argument("--step-timeout-seconds", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.step_timeout_seconds < 30:
        raise SystemExit("--step-timeout-seconds must be at least 30")

    _preflight()
    if args.preflight_only:
        print("Release dependency preflight passed")
        return

    release_lock = ExclusiveFileLock(
        ROOT / "artifacts" / ".locks" / "release.lock",
        timeout_seconds=float(os.getenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "5")),
        stale_after_seconds=float(os.getenv("REFERENCE_ETA_LOCK_STALE_SECONDS", "3600")),
        purpose="full release",
    )
    with release_lock:
        _execute_release(args)


def _execute_release(args: argparse.Namespace) -> None:
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    steps = [
        Step("Ruff", [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"]),
        Step(
            "Ruff format",
            [sys.executable, "-m", "ruff", "format", "--check", "src", "scripts", "tests"],
        ),
        Step("Compileall", [sys.executable, "-m", "compileall", "-q", "src", "scripts"]),
        Step("Mypy", [sys.executable, "-m", "mypy", "src/reference_eta"]),
        Step(
            "Pytest with coverage",
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=reference_eta",
                "--cov-report=term-missing:skip-covered",
                "--cov-report=json:reports/coverage.json",
                "--cov-fail-under=70",
            ],
        ),
        Step(
            "Base dependency isolation",
            [sys.executable, "scripts/verify_base_without_torch.py"],
            retry_on_timeout=True,
        ),
        Step(
            "Compact HSG-ETA",
            [
                sys.executable,
                "-m",
                "scripts.train_hsg_eta",
                "--config",
                "configs/gpu_smoke.yaml",
                "--require-release-pass",
                "--force-process-exit",
            ],
            retry_on_timeout=True,
        ),
        Step(
            "Synthetic core",
            [
                sys.executable,
                "scripts/run_pipeline.py",
                "--config",
                "configs/smoke.yaml",
                "--mode",
                "smoke",
                "--require-release-pass",
                "--force-process-exit",
            ],
            retry_on_timeout=True,
        ),
        Step(
            "Generate LaDe-style sample",
            [
                sys.executable,
                "scripts/generate_lade_sample.py",
                "--output",
                "artifacts/sample_lade_normalized.csv",
            ],
        ),
        Step(
            "Normalized LaDe-style pipeline",
            [
                sys.executable,
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
            retry_on_timeout=True,
        ),
        Step(
            "Generate Amazon official-shaped sample",
            [
                sys.executable,
                "scripts/generate_amazon_sample.py",
                "--output-dir",
                "artifacts/amazon_sample",
            ],
        ),
        Step(
            "Amazon official-shaped replay",
            [
                sys.executable,
                "scripts/run_amazon_replay.py",
                "--input-dir",
                "artifacts/amazon_sample",
                "--output",
                "reports/amazon_official_shape_replay.csv",
            ],
        ),
        Step("Runtime API verification", [sys.executable, "scripts/verify_runtime_api.py"]),
        Step(
            "Concurrent API benchmark",
            [
                sys.executable,
                "scripts/benchmark_api.py",
                "--requests",
                "64",
                "--concurrency",
                "8",
                "--output",
                "reports/api_concurrency_benchmark.json",
            ],
            retry_on_timeout=True,
        ),
        Step(
            "Locking and publish recovery",
            [
                sys.executable,
                "scripts/verify_locking.py",
                "--output",
                "reports/locking_recovery_report.json",
                "--force-process-exit",
            ],
            retry_on_timeout=True,
        ),
        Step(
            "Deterministic replay",
            [
                sys.executable,
                "scripts/verify_reproducibility.py",
                "--config",
                "configs/gpu_smoke.yaml",
                "--mode",
                "gpu",
            ],
            retry_on_timeout=True,
        ),
        Step(
            "Distribution build",
            [
                sys.executable,
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
        Step(
            "Normalize source distribution",
            [sys.executable, "-m", "scripts.normalize_sdist", "--dist-dir", "dist"],
        ),
        Step(
            "Distribution verification",
            [sys.executable, "scripts/verify_distribution.py", "--dist-dir", "dist"],
        ),
        Step(
            "Distribution reproducibility",
            [
                sys.executable,
                "-m",
                "scripts.verify_build_reproducibility",
                "--dist-dir",
                "dist",
                "--output",
                "reports/distribution_reproducibility.json",
            ],
        ),
        Step(
            "CycloneDX SBOM",
            [sys.executable, "scripts/generate_sbom.py", "--output", "reports/sbom.cdx.json"],
        ),
        Step("Build metadata cleanup", [sys.executable, "scripts/clean_build_metadata.py"]),
        Step(
            "Run artifact manifests",
            [sys.executable, "scripts/verify_artifact_manifest.py", "--all"],
        ),
    ]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_sections = ["# Final release execution log\n"]
    LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
    total_started = time.perf_counter()
    try:
        for step in steps:
            print(f"START: {step.name}", flush=True)
            log_sections.append(f"## START {step.name}\n$ {' '.join(step.command)}\n")
            LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
            section, elapsed = _run_step(step, args.step_timeout_seconds)
            section += f"Elapsed seconds: {elapsed:.3f}\n"
            log_sections.append(section)
            LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
            print(f"PASS: {step.name} ({elapsed:.2f}s)", flush=True)
    except RuntimeError as exc:
        log_sections.append(f"## FAILURE\n{exc}\n")
        LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
        raise SystemExit(str(exc)) from exc

    log_sections.append(
        f"## Completed pre-manifest release steps\nElapsed seconds: "
        f"{time.perf_counter() - total_started:.3f}\n"
    )
    LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")

    try:
        for final_step in [
            Step("Build release manifest", [sys.executable, "scripts/build_release_manifest.py"]),
            Step(
                "Verify release manifest",
                [
                    sys.executable,
                    "scripts/verify_artifact_manifest.py",
                    "--manifest",
                    "artifacts/release_manifest.json",
                ],
            ),
            Step(
                "Clean candidate validation",
                [
                    sys.executable,
                    "scripts/verify_clean_candidate.py",
                    "--output",
                    "reports/clean_candidate_validation.json",
                ],
            ),
            Step("Rebuild release manifest", [sys.executable, "scripts/build_release_manifest.py"]),
            Step(
                "Reverify release manifest",
                [
                    sys.executable,
                    "scripts/verify_artifact_manifest.py",
                    "--manifest",
                    "artifacts/release_manifest.json",
                ],
            ),
            Step(
                "Build release-candidate handoff",
                [sys.executable, "scripts/build_release_candidate_handoff.py"],
            ),
        ]:
            log_sections.append(f"## START {final_step.name}\n$ {' '.join(final_step.command)}\n")
            LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
            section, elapsed = _run_step(final_step, args.step_timeout_seconds)
            section += f"Elapsed seconds: {elapsed:.3f}\n"
            log_sections.append(section)
            print(f"PASS: {final_step.name} ({elapsed:.2f}s)", flush=True)
    except RuntimeError as exc:
        log_sections.append(f"## FAILURE\n{exc}\n")
        LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
        raise SystemExit(str(exc)) from exc

    total_elapsed = time.perf_counter() - total_started
    log_sections.append(f"## Release completed\nElapsed seconds: {total_elapsed:.3f}\n")
    LOG_PATH.write_text("\n".join(log_sections), encoding="utf-8")
    print(f"Release completed. Detailed log: {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
