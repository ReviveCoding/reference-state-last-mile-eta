from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reference_eta import __version__
from reference_eta.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_candidate_handoff.json"
CHECKSUM_OUTPUT = ROOT / "release_candidate_handoff.json.sha256"
BASELINE_FINGERPRINT = ROOT / "artifacts" / "baseline_source_fingerprint.json"
BASELINE_METRICS = ROOT / "artifacts" / "baseline_candidate_metrics.json"
GENERATED_ROOTS = {
    "artifacts",
    "reports",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
GENERATED_NAMES = {
    "qualification_manifest.json",
    "release_bundle_manifest.json",
    "release_candidate_handoff.json",
    "release_candidate_handoff.json.sha256",
    ".coverage",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_records(root: Path = ROOT) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in GENERATED_ROOTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name in GENERATED_NAMES or path.name.startswith(".coverage."):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def source_fingerprint(records: list[dict[str, object]]) -> str:
    ordered = sorted(records, key=lambda record: str(record["path"]))
    canonical = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def source_diff(
    baseline_records: list[dict[str, object]], candidate_records: list[dict[str, object]]
) -> dict[str, object]:
    baseline = {str(record["path"]): record for record in baseline_records}
    candidate = {str(record["path"]): record for record in candidate_records}
    added = sorted(candidate.keys() - baseline.keys())
    deleted = sorted(baseline.keys() - candidate.keys())
    modified = sorted(
        path
        for path in baseline.keys() & candidate.keys()
        if baseline[path]["sha256"] != candidate[path]["sha256"]
        or baseline[path]["size_bytes"] != candidate[path]["size_bytes"]
    )
    details = {"added": added, "modified": modified, "deleted": deleted}
    canonical = json.dumps(details, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**details, "diff_checksum": hashlib.sha256(canonical).hexdigest()}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"source_commit": commit, "dirty_tree": dirty, "git_available": True}
    except (OSError, subprocess.SubprocessError):
        return {
            "source_commit": None,
            "dirty_tree": None,
            "git_available": False,
            "reason": "distributed source snapshot contains no Git metadata",
        }


def _dependency_files() -> list[dict[str, object]]:
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "constraints" / "ci-py311.txt",
    ]
    return [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in paths]


def _artifact_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required handoff artifact is missing: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _test_count(release_log: str) -> int:
    matches = re.findall(r"(\d+) passed", release_log)
    if not matches:
        raise ValueError("Could not find a passed-test count in reports/final_release_log.txt")
    return max(int(value) for value in matches)


def _created_at_utc() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def build_handoff() -> dict[str, object]:
    required_paths = {
        "release_manifest": ROOT / "artifacts" / "release_manifest.json",
        "release_decision": ROOT / "artifacts" / "release_decision.json",
        "run_summary": ROOT / "reports" / "run_summary.json",
        "coverage": ROOT / "reports" / "coverage.json",
        "api_benchmark": ROOT / "reports" / "api_concurrency_benchmark.json",
        "locking_report": ROOT / "reports" / "locking_recovery_report.json",
        "reproducibility": ROOT / "reports" / "reproducibility_report.json",
        "distribution_reproducibility": ROOT / "reports" / "distribution_reproducibility.json",
        "sbom": ROOT / "reports" / "sbom.cdx.json",
        "release_log": ROOT / "reports" / "final_release_log.txt",
        "clean_validation": ROOT / "reports" / "clean_candidate_validation.json",
        "wheel": next(iter(sorted((ROOT / "dist").glob("*.whl"))), None),
        "sdist": next(iter(sorted((ROOT / "dist").glob("*.tar.gz"))), None),
    }
    if required_paths["wheel"] is None or required_paths["sdist"] is None:
        raise FileNotFoundError("Expected one wheel and one sdist in dist/")
    for name, path in required_paths.items():
        if not isinstance(path, Path) or not path.is_file():
            raise FileNotFoundError(f"Required handoff evidence is missing: {name}={path}")

    baseline_fingerprint = _load_json(BASELINE_FINGERPRINT)
    baseline_metrics = _load_json(BASELINE_METRICS)
    candidate_records = _source_records()
    candidate_fingerprint = source_fingerprint(candidate_records)
    diff = source_diff(baseline_fingerprint["files"], candidate_records)
    release_decision = _load_json(required_paths["release_decision"])
    run_summary = _load_json(required_paths["run_summary"])
    coverage = _load_json(required_paths["coverage"])
    api_benchmark = _load_json(required_paths["api_benchmark"])
    locking_report = _load_json(required_paths["locking_report"])
    clean_validation = _load_json(required_paths["clean_validation"])
    reproducibility = _load_json(required_paths["reproducibility"])
    distribution_reproducibility = _load_json(required_paths["distribution_reproducibility"])
    release_log = required_paths["release_log"].read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    handoff: dict[str, object] = {
        "schema_version": 1,
        "project_name": project["name"],
        "candidate_version": __version__,
        "candidate_status": "RELEASE_CANDIDATE_NOT_QUALIFIED",
        "created_at_utc": _created_at_utc(),
        "source": {
            **_git_state(),
            "baseline_source_fingerprint": baseline_fingerprint["source_fingerprint"],
            "candidate_source_fingerprint": candidate_fingerprint,
            "candidate_source_file_count": len(candidate_records),
            "diff": diff,
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "dependency_manifests": _dependency_files(),
        "supported_environment_claims": [
            {
                "environment": f"{platform.system()} {platform.machine()} / Python {platform.python_version()}",
                "status": "executed",
                "evidence_level": "E3",
            },
            {
                "environment": "Ubuntu GitHub runner / Python 3.11",
                "status": "configured_not_executed_here",
                "evidence_level": "E1",
            },
            {
                "environment": "Windows GitHub runner / Python 3.11 and 3.13",
                "status": "configured_not_executed_here",
                "evidence_level": "E1",
            },
        ],
        "required_entrypoints": {
            "task_runner": "python scripts/tasks.py <task>",
            "cli": "reference-eta",
            "api": "reference_eta.serving.api:app",
            "training": "python scripts/tasks.py train-gpu",
            "evaluation": "python scripts/tasks.py smoke",
            "release": "python scripts/tasks.py release",
        },
        "datasets_and_fixtures": [
            {
                "name": "deterministic synthetic courier-day data",
                "role": "correctness, integration, release-gate validation",
                "claim_boundary": "not real-world performance evidence",
            },
            {
                "name": "normalized LaDe-style generated sample",
                "role": "adapter and ETA-pipeline contract validation",
                "claim_boundary": "not official LaDe benchmark evidence",
            },
            {
                "name": "Amazon official-shaped generated sample",
                "role": "route/time-window replay contract validation",
                "claim_boundary": "not official Amazon dataset evidence",
            },
        ],
        "baseline_metrics": baseline_metrics,
        "final_metrics": {
            "tests_passed": _test_count(release_log),
            "coverage_percent": coverage["totals"]["percent_covered"],
            "release_gate": run_summary["release_gate"],
            "quantile_champion": run_summary["quantile_champion"],
            "decision_champion": run_summary["decision_champion"],
            "rcot_promotion_gate": run_summary["rcot_promotion_gate"],
            "interval_coverage": run_summary["interval_coverage"],
            "rolling_replay_coverage": run_summary["rolling_replay_coverage"],
            "api": {
                "requests": api_benchmark["requests"],
                "concurrency": api_benchmark["concurrency"],
                "configured_workers": api_benchmark["configured_workers"],
                "error_rate": api_benchmark["error_rate"],
                "p50_ms": api_benchmark["latency_ms_p50"],
                "p95_ms": api_benchmark["latency_ms_p95"],
                "max_ms": api_benchmark["latency_ms_max"],
                "graceful_shutdown": api_benchmark["graceful_shutdown"],
            },
            "locking_status": locking_report["status"],
            "deterministic_replay_status": reproducibility["status"],
            "distribution_reproducibility_status": distribution_reproducibility["status"],
        },
        "metric_gates": {
            "release_gate_prefix": "PASS_",
            "coverage_minimum_percent": 70.0,
            "api_error_rate_maximum": 0.0,
            "api_p95_latency_maximum_ms": api_benchmark["max_p95_ms"],
            "quantile_crossing_maximum": 0.0,
            "champion_relative_regression_maximum": release_decision["system_release"][
                "maximum_champion_relative_regression"
            ],
        },
        "command_evidence": [
            {"command": "python scripts/release.py", "exit_code": 0, "evidence_level": "E2"},
            {
                "command": clean_validation["command"],
                "exit_code": clean_validation["exit_code"],
                "evidence_level": "E3",
                "tests_passed": clean_validation["tests_passed"],
                "manifest_records": clean_validation["repository_manifest_records"],
            },
        ],
        "build_artifacts": [
            _artifact_record(required_paths["wheel"]),
            _artifact_record(required_paths["sdist"]),
            _artifact_record(required_paths["release_manifest"]),
            _artifact_record(required_paths["sbom"]),
        ],
        "known_limitations": [
            "official full LaDe benchmark not executed",
            "official Amazon dataset replay not executed",
            "actual CUDA performance and VRAM profiling not executed",
            "actual Windows runner not executed",
            "remote GitHub Actions, CodeQL, Docker build, and artifact attestation not executed",
            "no production causal intervention-impact claim",
        ],
        "unresolved_items": [
            {
                "item": "Windows hosted-runner qualification",
                "severity": "High",
                "status": "external_environment_required",
            },
            {
                "item": "remote GitHub Actions and Docker qualification",
                "severity": "High",
                "status": "external_environment_required",
            },
            {
                "item": "official full-data benchmark and CUDA profile",
                "severity": "Medium",
                "status": "external_data_or_hardware_required",
            },
        ],
        "evidence_level": {
            "current_maximum": "E3",
            "E0": "source and documentation reviewed",
            "E1": "commands and hosted workflows configured",
            "E2": "executed in current working environment",
            "E3": "clean-copy and installed-distribution execution",
            "E4": "not achieved; exact-commit GitHub-hosted execution required",
        },
        "next_qualification_gates": [
            "run exact candidate commit on Ubuntu Python 3.11 GitHub-hosted runner",
            "run Windows Python 3.11 and 3.13 compatibility jobs",
            "build and health-check Docker image",
            "verify GitHub artifact attestations",
            "run full official-data and CUDA evidence profiles before stronger performance claims",
        ],
    }
    if not str(handoff["final_metrics"]["release_gate"]).startswith("PASS_"):  # type: ignore[index]
        raise ValueError("Candidate handoff refuses a non-passing release gate")
    if float(handoff["final_metrics"]["coverage_percent"]) < 70.0:  # type: ignore[index]
        raise ValueError("Candidate handoff refuses coverage below the release floor")
    return handoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate release-candidate handoff")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    handoff = build_handoff()
    atomic_write_json(args.output, handoff)
    digest = sha256(args.output)
    checksum_path = (
        CHECKSUM_OUTPUT
        if args.output.resolve() == OUTPUT.resolve()
        else args.output.with_suffix(args.output.suffix + ".sha256")
    )
    atomic_write_text(checksum_path, f"{digest}  {args.output.name}\n")
    reloaded = _load_json(args.output)
    if reloaded != handoff:
        raise RuntimeError("Handoff changed during atomic write/read validation")
    for record in handoff["build_artifacts"]:  # type: ignore[assignment]
        path = ROOT / str(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Handoff artifact checksum changed: {path}")
    print(f"Wrote validated release-candidate handoff: {args.output}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
