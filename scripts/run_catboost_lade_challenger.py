from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from reference_eta.config import load_config
from reference_eta.decisions.triage import fit_tail_thresholds, threshold_for_rows
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import point_metrics
from reference_eta.features.tabular import BASE_NUMERIC_FEATURES, RCOT_NUMERIC_FEATURES
from reference_eta.models.baselines import LightGBMConfig, LightGBMPointModel

TARGET = "target_route_remaining_minutes"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    use_courier: bool
    use_rcot: bool
    use_reference_labels: bool
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    random_strength: float


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _read_partition(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} partition: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{name} partition is empty")
    if TARGET not in frame.columns:
        raise ValueError(f"{name} partition is missing target: {TARGET}")
    return frame


def _model_config(config: dict[str, Any]) -> LightGBMConfig:
    model = config["model"]
    return LightGBMConfig(
        n_estimators=int(model["n_estimators"]),
        learning_rate=float(model["learning_rate"]),
        num_leaves=int(model["num_leaves"]),
        min_child_samples=int(model["min_child_samples"]),
        random_state=int(model["random_state"]),
    )


def _candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="catboost_city_base",
            use_courier=False,
            use_rcot=False,
            use_reference_labels=False,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=6.0,
            random_strength=0.5,
        ),
        CandidateSpec(
            name="catboost_city_courier",
            use_courier=True,
            use_rcot=False,
            use_reference_labels=False,
            depth=7,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            random_strength=0.5,
        ),
        CandidateSpec(
            name="catboost_city_courier_rcot",
            use_courier=True,
            use_rcot=True,
            use_reference_labels=True,
            depth=7,
            learning_rate=0.04,
            l2_leaf_reg=10.0,
            random_strength=0.75,
        ),
    ]


def _available_reference_labels(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("reference_level", "reference_regime") if column in frame.columns]


def _feature_columns(frame: pd.DataFrame, spec: CandidateSpec) -> tuple[list[str], list[str]]:
    numeric = list(BASE_NUMERIC_FEATURES)
    if spec.use_rcot:
        numeric.extend(RCOT_NUMERIC_FEATURES)
    missing_numeric = [column for column in numeric if column not in frame.columns]
    if missing_numeric:
        raise ValueError(f"{spec.name} is missing numeric columns: {missing_numeric}")

    categorical = ["city"]
    if spec.use_courier:
        if "courier_id" not in frame.columns:
            raise ValueError(f"{spec.name} requires courier_id")
        categorical.append("courier_id")
    if spec.use_reference_labels:
        categorical.extend(_available_reference_labels(frame))

    return numeric, categorical


def _build_features(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    spec: CandidateSpec,
) -> tuple[pd.DataFrame, list[str]]:
    numeric_columns, categorical_columns = _feature_columns(train, spec)
    all_columns = [*numeric_columns, *categorical_columns]
    result = frame.reindex(columns=all_columns).copy()

    train_numeric = train[numeric_columns].apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median(numeric_only=True).fillna(0.0)

    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = values.fillna(float(medians[column])).astype(float)

    for column in categorical_columns:
        values = result[column].astype("string").fillna("__MISSING__")
        result[column] = values.replace("", "__MISSING__").astype(str)

    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{spec.name} contains non-finite numeric features")

    return result, categorical_columns


def _build_model(spec: CandidateSpec, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=1500,
        depth=spec.depth,
        learning_rate=spec.learning_rate,
        l2_leaf_reg=spec.l2_leaf_reg,
        random_strength=spec.random_strength,
        random_seed=seed,
        task_type="CPU",
        thread_count=1,
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=100,
    )


def _tail_mae(y_true: np.ndarray, prediction: np.ndarray, thresholds: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    threshold = np.asarray(thresholds, dtype=float)
    mask = y > threshold
    if not mask.any():
        return float("nan")
    return float(point_metrics(y[mask], pred[mask])["mae"])


def _slice_rows(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    challenger_prediction: np.ndarray,
    *,
    column: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, group in frame.groupby(column, dropna=False):
        positions = group.index.to_numpy()
        y = np.asarray(y_true, dtype=float)[positions]
        baseline = np.asarray(baseline_prediction, dtype=float)[positions]
        challenger = np.asarray(challenger_prediction, dtype=float)[positions]
        baseline_mae = float(point_metrics(y, baseline)["mae"])
        challenger_mae = float(point_metrics(y, challenger)["mae"])
        rows.append(
            {
                "slice": column,
                "value": str(value),
                "count": int(len(group)),
                "baseline_mae": baseline_mae,
                "catboost_mae": challenger_mae,
                "relative_improvement": float(
                    (baseline_mae - challenger_mae) / max(baseline_mae, 1e-9)
                ),
            }
        )
    return rows


def _route_phase_band(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(
        frame["route_phase"],
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)


def _select_candidate(
    validation_y: np.ndarray,
    validation_baseline: np.ndarray,
    candidate_predictions: dict[str, np.ndarray],
    tail_thresholds: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    baseline_mae = float(point_metrics(validation_y, validation_baseline)["mae"])
    baseline_tail_mae = _tail_mae(validation_y, validation_baseline, tail_thresholds)
    rows: list[dict[str, object]] = []

    for name, prediction in candidate_predictions.items():
        metrics = point_metrics(validation_y, prediction)
        candidate_tail_mae = _tail_mae(validation_y, prediction, tail_thresholds)
        tail_regression = float(candidate_tail_mae / max(baseline_tail_mae, 1e-9) - 1.0)
        rows.append(
            {
                "model": name,
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": float(metrics["rmse"]),
                "validation_median_absolute_error": float(metrics["median_absolute_error"]),
                "validation_tail_mae": candidate_tail_mae,
                "relative_mae_improvement": float(
                    (baseline_mae - float(metrics["mae"])) / max(baseline_mae, 1e-9)
                ),
                "tail_mae_regression": tail_regression,
                "tail_guardrail_pass": bool(tail_regression <= 0.02),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["validation_mae", "validation_tail_mae", "model"],
        ascending=[True, True, True],
        kind="stable",
    )
    eligible = result.loc[result["tail_guardrail_pass"]].copy()
    if eligible.empty:
        return "baseline_fallback", result
    return str(eligible.iloc[0]["model"]), result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic CPU CatBoost challenger on fixed LaDe-D partitions"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    train = _read_partition(args.train, "train")
    validation = _read_partition(args.validation, "validation")
    test = _read_partition(args.test, "test")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_y = train[TARGET].to_numpy(dtype=float)
    validation_y = validation[TARGET].to_numpy(dtype=float)
    test_y = test[TARGET].to_numpy(dtype=float)

    lightgbm = LightGBMPointModel(_model_config(config), include_rcot=False).fit(train, TARGET)
    validation_lightgbm = lightgbm.predict(validation)
    test_lightgbm = lightgbm.predict(test)

    candidate_models: dict[str, CatBoostRegressor] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    candidate_metadata: dict[str, dict[str, object]] = {}

    for spec in _candidate_specs():
        train_x, categorical_columns = _build_features(train, train, spec)
        validation_x, _ = _build_features(train, validation, spec)
        test_x, _ = _build_features(train, test, spec)
        cat_indices = [train_x.columns.get_loc(column) for column in categorical_columns]
        train_pool = Pool(train_x, label=train_y, cat_features=cat_indices)
        validation_pool = Pool(validation_x, label=validation_y, cat_features=cat_indices)
        model = _build_model(spec, int(args.seed))
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True, verbose=False)
        candidate_models[spec.name] = model
        validation_predictions[spec.name] = np.maximum(model.predict(validation_x), 0.0)
        test_predictions[spec.name] = np.maximum(model.predict(test_x), 0.0)
        candidate_metadata[spec.name] = {
            **asdict(spec),
            "feature_columns": list(train_x.columns),
            "categorical_columns": categorical_columns,
            "best_iteration": int(model.get_best_iteration()),
        }

    tail_threshold_fit = fit_tail_thresholds(train, TARGET)
    validation_tail_thresholds = threshold_for_rows(validation, tail_threshold_fit)
    test_tail_thresholds = threshold_for_rows(test, tail_threshold_fit)

    selected_name, validation_frame = _select_candidate(
        validation_y,
        validation_lightgbm,
        validation_predictions,
        validation_tail_thresholds,
    )

    if selected_name == "baseline_fallback":
        selected_test_prediction = test_lightgbm.copy()
        selected_validation_prediction = validation_lightgbm.copy()
        selected_metadata: dict[str, object] = {
            "selection_reason": "No CatBoost candidate satisfied validation tail guardrail"
        }
    else:
        selected_test_prediction = test_predictions[selected_name]
        selected_validation_prediction = validation_predictions[selected_name]
        selected_metadata = candidate_metadata[selected_name]

    lightgbm_test_metrics = point_metrics(test_y, test_lightgbm)
    selected_test_metrics = point_metrics(test_y, selected_test_prediction)
    relative_mae_improvement = float(
        (float(lightgbm_test_metrics["mae"]) - float(selected_test_metrics["mae"]))
        / max(float(lightgbm_test_metrics["mae"]), 1e-9)
    )

    bootstrap = clustered_mae_difference_ci(
        test,
        test_y,
        test_lightgbm,
        selected_test_prediction,
        seed=int(args.seed),
    )
    ci_supports_improvement = bool(float(bootstrap["ci_upper"]) < 0.0)
    selected_test_tail_mae = _tail_mae(test_y, selected_test_prediction, test_tail_thresholds)
    lightgbm_test_tail_mae = _tail_mae(test_y, test_lightgbm, test_tail_thresholds)
    tail_guardrail_pass = bool(selected_test_tail_mae <= lightgbm_test_tail_mae * 1.02)
    claim_supported = bool(
        selected_name != "baseline_fallback"
        and relative_mae_improvement > 0.0
        and ci_supports_improvement
        and tail_guardrail_pass
    )

    test_rows: list[dict[str, object]] = [
        {
            "model": "lightgbm_without_rcot_reproduced",
            **lightgbm_test_metrics,
            "tail_mae": lightgbm_test_tail_mae,
            "selected": False,
        }
    ]
    for name, prediction in test_predictions.items():
        metrics = point_metrics(test_y, prediction)
        test_rows.append(
            {
                "model": name,
                **metrics,
                "tail_mae": _tail_mae(test_y, prediction, test_tail_thresholds),
                "selected": bool(name == selected_name),
            }
        )

    test_scorecard = pd.DataFrame(test_rows).sort_values(
        ["mae", "model"], ascending=[True, True], kind="stable"
    )
    test_scorecard.to_csv(args.output_dir / "catboost_challenger_scorecard.csv", index=False)

    validation_frame["selected"] = validation_frame["model"].eq(selected_name)
    validation_frame.to_csv(args.output_dir / "catboost_challenger_validation.csv", index=False)

    if selected_name != "baseline_fallback":
        candidate_models[selected_name].save_model(args.output_dir / "selected_catboost_model.cbm")
        importance = pd.DataFrame(
            {
                "feature": selected_metadata["feature_columns"],
                "importance": candidate_models[selected_name].get_feature_importance(),
            }
        ).sort_values("importance", ascending=False, kind="stable")
        importance.to_csv(args.output_dir / "selected_catboost_feature_importance.csv", index=False)

    predictions = test[
        [
            column
            for column in ("snapshot_id", "courier_id", "work_date", "city", "route_phase")
            if column in test.columns
        ]
    ].copy()
    predictions["target"] = test_y
    predictions["lightgbm_without_rcot_prediction"] = test_lightgbm
    predictions["selected_catboost_prediction"] = selected_test_prediction
    predictions["lightgbm_absolute_error"] = np.abs(test_y - test_lightgbm)
    predictions["selected_catboost_absolute_error"] = np.abs(test_y - selected_test_prediction)
    predictions["selected_model"] = selected_name
    predictions.to_csv(args.output_dir / "catboost_challenger_test_predictions.csv", index=False)

    slice_frame = test.reset_index(drop=True).copy()
    slice_frame["route_phase_band"] = _route_phase_band(slice_frame)
    train_couriers = set(train["courier_id"].astype(str))
    slice_frame["courier_seen_in_train"] = (
        slice_frame["courier_id"]
        .astype(str)
        .isin(train_couriers)
        .map({True: "seen", False: "unseen"})
    )
    slice_rows: list[dict[str, object]] = []
    for column in ("city", "route_phase_band", "courier_seen_in_train"):
        slice_rows.extend(
            _slice_rows(
                slice_frame,
                test_y,
                test_lightgbm,
                selected_test_prediction,
                column=column,
            )
        )
    pd.DataFrame(slice_rows).to_csv(args.output_dir / "catboost_challenger_slices.csv", index=False)

    summary = {
        "experiment": "deterministic_cpu_catboost_challenger",
        "status": "CLAIM_SUPPORTED" if claim_supported else "HOLD_LIGHTGBM",
        "claim_supported": claim_supported,
        "claim_boundary": (
            "offline normalized LaDe-D pilot; fixed chronological split; "
            "no production delay-prevention claim"
        ),
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "selection": {
            "selection_dataset": "validation_only",
            "selected_model": selected_name,
            "selected_model_metadata": selected_metadata,
            "validation_baseline_mae": float(
                point_metrics(validation_y, validation_lightgbm)["mae"]
            ),
            "validation_selected_mae": float(
                point_metrics(validation_y, selected_validation_prediction)["mae"]
            ),
        },
        "test": {
            "lightgbm_without_rcot": lightgbm_test_metrics,
            "selected_model": selected_test_metrics,
            "relative_mae_improvement": relative_mae_improvement,
            "lightgbm_tail_mae": lightgbm_test_tail_mae,
            "selected_tail_mae": selected_test_tail_mae,
            "tail_guardrail_pass": tail_guardrail_pass,
            "bootstrap": bootstrap,
            "ci_supports_improvement": ci_supports_improvement,
            "observed_best_existing_q50_mae": 88.38095292920616,
            "beats_observed_best_existing_q50_mae": bool(
                float(selected_test_metrics["mae"]) < 88.38095292920616
            ),
        },
        "reproducibility": {"task_type": "CPU", "thread_count": 1, "random_seed": int(args.seed)},
    }
    (args.output_dir / "catboost_challenger_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
