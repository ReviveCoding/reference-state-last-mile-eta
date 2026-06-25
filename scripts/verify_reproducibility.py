from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from reference_eta.io import atomic_write_json  # noqa: E402
from scripts.run_pipeline import ROOT, run  # noqa: E402

SUMMARY_EXCLUSIONS = {"output_namespace", "serving_bundle"}
DYNAMIC_KEYS = {"training_seconds", "epoch_seconds"}
CSV_FILES = (
    "model_scorecard.csv",
    "test_predictions.csv",
    "capacity_triage.csv",
    "slice_performance.csv",
    "rolling_calibration_replay.csv",
    "hsg_test_predictions.csv",
)


def _strip_dynamic(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_dynamic(item) for key, item in value.items() if key not in DYNAMIC_KEYS}
    if isinstance(value, list):
        return [_strip_dynamic(item) for item in value]
    return value


def canonical_summary(value: dict[str, Any]) -> dict[str, Any]:
    return _strip_dynamic(
        {key: item for key, item in value.items() if key not in SUMMARY_EXCLUSIONS}
    )


def _assert_json_equal(first: Any, second: Any, *, path: str = "root", atol: float) -> None:
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            raise AssertionError(f"JSON keys differ at {path}: {set(first) ^ set(second)}")
        for key in sorted(first):
            _assert_json_equal(first[key], second[key], path=f"{path}.{key}", atol=atol)
        return
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            raise AssertionError(f"JSON list lengths differ at {path}")
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_json_equal(left, right, path=f"{path}[{index}]", atol=atol)
        return
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        if not np.isclose(float(first), float(second), rtol=0.0, atol=atol, equal_nan=True):
            raise AssertionError(f"JSON numeric values differ at {path}: {first} != {second}")
        return
    if first != second:
        raise AssertionError(f"JSON values differ at {path}: {first!r} != {second!r}")


def verify_reproducibility(
    config: Path,
    *,
    mode: str,
    atol: float,
    first_namespace: str = "reproducibility_a",
    second_namespace: str = "reproducibility_b",
) -> dict[str, Any]:
    artifact_dirs = [ROOT / "artifacts" / first_namespace, ROOT / "artifacts" / second_namespace]
    report_dirs = [ROOT / "reports" / first_namespace, ROOT / "reports" / second_namespace]
    for directory in [*artifact_dirs, *report_dirs]:
        shutil.rmtree(directory, ignore_errors=True)

    try:
        first_summary = run(config, mode, output_namespace=first_namespace)
        second_summary = run(config, mode, output_namespace=second_namespace)
        _assert_json_equal(
            canonical_summary(first_summary),
            canonical_summary(second_summary),
            path="summary",
            atol=atol,
        )

        first_release = json.loads(
            (artifact_dirs[0] / "release_decision.json").read_text(encoding="utf-8")
        )
        second_release = json.loads(
            (artifact_dirs[1] / "release_decision.json").read_text(encoding="utf-8")
        )
        _assert_json_equal(first_release, second_release, path="release_decision", atol=atol)

        compared_files: list[str] = []
        for name in CSV_FILES:
            left_path = report_dirs[0] / name
            right_path = report_dirs[1] / name
            if left_path.exists() != right_path.exists():
                raise AssertionError(f"Reproducibility output presence differs for {name}")
            if not left_path.exists():
                continue
            left = pd.read_csv(left_path)
            right = pd.read_csv(right_path)
            assert_frame_equal(
                left,
                right,
                check_dtype=False,
                check_exact=False,
                rtol=0.0,
                atol=atol,
            )
            compared_files.append(name)

        return {
            "reproducibility_schema_version": 1,
            "status": "PASS",
            "config": str(Path(config)),
            "mode": mode,
            "absolute_tolerance": float(atol),
            "compared_files": compared_files,
            "release_gate": first_summary["release_gate"],
            "quantile_champion": first_summary["quantile_champion"],
            "decision_champion": first_summary["decision_champion"],
        }
    finally:
        for directory in [*artifact_dirs, *report_dirs]:
            shutil.rmtree(directory, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the same compact pipeline twice and compare outputs"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gpu_smoke.yaml")
    parser.add_argument("--mode", choices=["smoke", "gpu", "all"], default="gpu")
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/reproducibility_report.json")
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0.0:
        raise SystemExit("--atol must be finite and nonnegative")
    result = verify_reproducibility(args.config, mode=args.mode, atol=float(args.atol))
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
