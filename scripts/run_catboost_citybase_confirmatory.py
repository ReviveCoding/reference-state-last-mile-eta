from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import point_metrics
from reference_eta.features.tabular import TabularFeatureEncoder

TARGET = "target_route_remaining_minutes"
BASE_NUMERIC = [
    "query_hour",
    "elapsed_minutes",
    "completed_task_count",
    "remaining_task_count",
    "initial_task_count",
    "completed_workload",
    "remaining_workload",
    "initial_workload",
    "observed_progress",
    "route_phase",
    "recent_pace",
    "task_density",
    "remaining_spread",
    "aoi_transition_burden",
    "weather_severity",
    "congestion_proxy",
    "trajectory_missingness",
]
CATBOOST_FEATURES = [*BASE_NUMERIC, "city"]


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(CATBOOST_FEATURES + [TARGET, "courier_id", "work_date", "query_time"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required snapshot columns: {sorted(missing)}")
    return frame


def _point_prediction_offset(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.median(np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)))


def _apply_offset(prediction: np.ndarray, offset: float) -> np.ndarray:
    return np.maximum(np.asarray(prediction, dtype=float) + float(offset), 0.0)


def _tail_mae(y: np.ndarray, prediction: np.ndarray, threshold: float) -> float:
    mask = np.asarray(y, dtype=float) >= threshold
    if not mask.any():
        return float("nan")
    return float(point_metrics(np.asarray(y)[mask], np.asarray(prediction)[mask])["mae"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-spec independent CatBoost CityBase confirmation against LightGBM."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _load(data_dir / "train_snapshots.csv")
    validation = _load(data_dir / "validation_snapshots.csv")
    calibration = _load(data_dir / "calibration_snapshots.csv")
    test = _load(data_dir / "test_snapshots.csv")

    for frame in (train, validation, calibration, test):
        frame["city"] = frame["city"].astype(str)
        for column in BASE_NUMERIC:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    y_train = train[TARGET].to_numpy(dtype=float)
    y_validation = validation[TARGET].to_numpy(dtype=float)
    y_calibration = calibration[TARGET].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=float)

    city_categories = sorted(train["city"].unique().tolist())
    for frame in (train, validation, calibration, test):
        frame["city"] = pd.Categorical(
            frame["city"],
            categories=city_categories,
        ).astype(str)

    lightgbm_encoder = TabularFeatureEncoder(include_rcot=False)
    lightgbm_train = lightgbm_encoder.fit_transform(train)
    lightgbm = LGBMRegressor(
        objective="mae",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=48,
        min_child_samples=30,
        random_state=int(args.seed),
        n_jobs=1,
        verbosity=-1,
    )
    lightgbm.fit(lightgbm_train, y_train)

    catboost = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=800,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=8.0,
        random_strength=0.5,
        random_seed=int(args.seed),
        task_type="CPU",
        thread_count=1,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=50,
        verbose=False,
    )
    catboost.fit(
        train[CATBOOST_FEATURES],
        y_train,
        cat_features=["city"],
        eval_set=(validation[CATBOOST_FEATURES], y_validation),
        use_best_model=True,
        verbose=False,
    )

    prediction_sets: dict[str, dict[str, np.ndarray]] = {}

    lightgbm_calibration = np.asarray(
        lightgbm.predict(lightgbm_encoder.transform(calibration)),
        dtype=float,
    )
    lightgbm_test = np.asarray(
        lightgbm.predict(lightgbm_encoder.transform(test)),
        dtype=float,
    )
    catboost_calibration = np.asarray(
        catboost.predict(calibration[CATBOOST_FEATURES]),
        dtype=float,
    )
    catboost_test = np.asarray(
        catboost.predict(test[CATBOOST_FEATURES]),
        dtype=float,
    )

    for name, calibration_prediction, test_prediction in [
        ("lightgbm", lightgbm_calibration, lightgbm_test),
        ("catboost_city_base", catboost_calibration, catboost_test),
    ]:
        offset = _point_prediction_offset(y_calibration, calibration_prediction)
        prediction_sets[name] = {
            "raw_test": np.maximum(test_prediction, 0.0),
            "calibrated_test": _apply_offset(test_prediction, offset),
            "offset": np.asarray([offset], dtype=float),
        }

    tail_threshold = float(np.quantile(y_train, 0.90))

    rows: list[dict[str, Any]] = []
    for name, values in prediction_sets.items():
        for variant, prediction in {
            "raw": values["raw_test"],
            "median_residual_calibrated": values["calibrated_test"],
        }.items():
            metrics = point_metrics(y_test, prediction)
            rows.append(
                {
                    "model": name,
                    "variant": variant,
                    **metrics,
                    "tail_mae": _tail_mae(y_test, prediction, tail_threshold),
                    "calibration_offset_minutes": float(values["offset"][0]),
                }
            )

    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(output_dir / "confirmatory_scorecard.csv", index=False)

    baseline = prediction_sets["lightgbm"]["calibrated_test"]
    challenger = prediction_sets["catboost_city_base"]["calibrated_test"]
    bootstrap = clustered_mae_difference_ci(
        test,
        y_test,
        baseline,
        challenger,
        seed=int(args.seed),
    )

    baseline_metrics = point_metrics(y_test, baseline)
    challenger_metrics = point_metrics(y_test, challenger)
    baseline_tail = _tail_mae(y_test, baseline, tail_threshold)
    challenger_tail = _tail_mae(y_test, challenger, tail_threshold)
    relative_improvement = float(
        (baseline_metrics["mae"] - challenger_metrics["mae"])
        / max(float(baseline_metrics["mae"]), 1e-9)
    )
    tail_guardrail_pass = bool(challenger_tail <= baseline_tail * 1.02)
    ci_supports_improvement = bool(float(bootstrap["ci_upper"]) < 0.0)
    claim_supported = bool(
        relative_improvement > 0.0 and ci_supports_improvement and tail_guardrail_pass
    )

    importance = pd.DataFrame(
        {
            "feature": CATBOOST_FEATURES,
            "importance": catboost.feature_importances_,
        }
    ).sort_values("importance", ascending=False, kind="stable")
    importance.to_csv(output_dir / "catboost_citybase_feature_importance.csv", index=False)

    predictions = test[["snapshot_id", "courier_id", "work_date", "query_time", "city"]].copy()
    predictions["target"] = y_test
    predictions["lightgbm_calibrated"] = baseline
    predictions["catboost_calibrated"] = challenger
    predictions["lightgbm_absolute_error"] = np.abs(y_test - baseline)
    predictions["catboost_absolute_error"] = np.abs(y_test - challenger)
    predictions.to_csv(output_dir / "confirmatory_test_predictions.csv", index=False)

    summary = {
        "experiment": "pre_specified_citybase_independent_cohort_confirmation",
        "status": "CLAIM_SUPPORTED" if claim_supported else "HOLD_LIGHTGBM",
        "claim_supported": claim_supported,
        "claim_boundary": (
            "independent deterministic LaDe-D whole-courier-day cohort; "
            "offline ETA evidence only; no production delay-prevention claim"
        ),
        "model_specification": {
            "catboost_features": CATBOOST_FEATURES,
            "catboost_categorical_features": ["city"],
            "catboost_depth": 7,
            "catboost_learning_rate": 0.04,
            "catboost_l2_leaf_reg": 8.0,
            "catboost_random_strength": 0.5,
            "catboost_task_type": "CPU",
            "catboost_thread_count": 1,
            "selection": "pre-specified, no candidate sweep on this cohort",
            "point_calibration": "median residual fit on the dedicated calibration partition for both models",
        },
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "test": {
            "baseline": {
                **baseline_metrics,
                "tail_mae": baseline_tail,
            },
            "catboost_city_base": {
                **challenger_metrics,
                "tail_mae": challenger_tail,
            },
            "relative_mae_improvement": relative_improvement,
            "tail_guardrail_pass": tail_guardrail_pass,
            "bootstrap": bootstrap,
            "ci_supports_improvement": ci_supports_improvement,
        },
    }
    (output_dir / "confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
