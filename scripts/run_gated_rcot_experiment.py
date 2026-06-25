from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from reference_eta.config import load_config
from reference_eta.decisions.triage import fit_tail_thresholds, threshold_for_rows
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import point_metrics
from reference_eta.evaluation.release import evaluate_rcot_promotion
from reference_eta.features.tabular import TabularFeatureEncoder
from reference_eta.models.baselines import LightGBMConfig, LightGBMPointModel

ROOT = Path(__file__).resolve().parents[1]
TARGET = "target_route_remaining_minutes"


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_partitions(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "train": data_dir / "train_snapshots.csv",
        "validation": data_dir / "validation_snapshots.csv",
        "test": data_dir / "test_snapshots.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved pipeline partitions:\n" + "\n".join(missing))

    train = pd.read_csv(paths["train"])
    validation = pd.read_csv(paths["validation"])
    test = pd.read_csv(paths["test"])

    for name, frame in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if frame.empty:
            raise ValueError(f"{name} partition is empty")
        if TARGET not in frame.columns:
            raise ValueError(f"{name} partition is missing {TARGET}")
        if "work_date" not in frame.columns:
            raise ValueError(f"{name} partition is missing work_date")

    return train, validation, test


def _model_config(config: dict[str, Any]) -> LightGBMConfig:
    model_cfg = config["model"]
    return LightGBMConfig(
        n_estimators=int(model_cfg["n_estimators"]),
        learning_rate=float(model_cfg["learning_rate"]),
        num_leaves=int(model_cfg["num_leaves"]),
        min_child_samples=int(model_cfg["min_child_samples"]),
        random_state=int(model_cfg["random_state"]),
    )


def _fit_experts(
    frame: pd.DataFrame,
    config: LightGBMConfig,
) -> tuple[LightGBMPointModel, LightGBMPointModel]:
    baseline = LightGBMPointModel(config, include_rcot=False).fit(frame, TARGET)
    rcot = LightGBMPointModel(config, include_rcot=True).fit(frame, TARGET)
    return baseline, rcot


def _forward_oof_expert_predictions(
    train: pd.DataFrame,
    config: LightGBMConfig,
    *,
    n_splits: int,
) -> pd.DataFrame:
    dates = np.array(sorted(train["work_date"].astype(str).unique()))
    if len(dates) < 3:
        raise ValueError("Need at least three chronological work dates for OOF gate training")

    split_count = max(3, min(int(n_splits), len(dates)))
    blocks = [block for block in np.array_split(dates, split_count) if len(block)]
    parts: list[pd.DataFrame] = []

    for block_index, held_dates in enumerate(blocks):
        if block_index == 0:
            continue

        historical_dates = np.concatenate(blocks[:block_index]).tolist()
        history = train.loc[train["work_date"].astype(str).isin(set(historical_dates))].copy()
        held = train.loc[train["work_date"].astype(str).isin(set(held_dates.tolist()))].copy()

        minimum_history_rows = max(40, config.min_child_samples * 2)
        if len(history) < minimum_history_rows or held.empty:
            continue

        baseline, rcot = _fit_experts(history, config)
        part = held.copy()
        part["_original_index"] = held.index.to_numpy()
        part["_base_prediction"] = baseline.predict(held)
        part["_rcot_prediction"] = rcot.predict(held)
        parts.append(part)

    if not parts:
        raise RuntimeError("Could not create chronological OOF expert predictions")

    output = pd.concat(parts, ignore_index=True)
    output = output.sort_values("_original_index", kind="stable").reset_index(drop=True)

    if len(output) < 40:
        raise RuntimeError(
            f"Only {len(output)} OOF rows available for gate training; need at least 40"
        )
    return output


class SelectiveRCOTGate:
    """Predict whether the RCOT expert has lower absolute error than baseline."""

    def __init__(self, *, random_state: int) -> None:
        self.encoder = TabularFeatureEncoder(include_rcot=True)
        self.model: LGBMClassifier | None = None
        self.constant_probability: float | None = None
        self.columns_: list[str] | None = None
        self.positive_rate_: float | None = None
        self.random_state = int(random_state)

    def _features(
        self,
        frame: pd.DataFrame,
        baseline_prediction: np.ndarray,
        rcot_prediction: np.ndarray,
    ) -> pd.DataFrame:
        baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)
        rcot = np.asarray(rcot_prediction, dtype=float).reshape(-1)

        if len(frame) != len(baseline) or len(frame) != len(rcot):
            raise ValueError("Gate frame and prediction lengths are inconsistent")
        if not np.isfinite(baseline).all() or not np.isfinite(rcot).all():
            raise ValueError("Gate predictions must be finite")

        encoded = self.encoder.transform(frame).reset_index(drop=True)
        correction = rcot - baseline
        expert_features = pd.DataFrame(
            {
                "baseline_prediction": baseline,
                "rcot_prediction": rcot,
                "rcot_minus_baseline": correction,
                "absolute_correction": np.abs(correction),
                "prediction_midpoint": 0.5 * (baseline + rcot),
            }
        )
        return pd.concat([encoded, expert_features], axis=1)

    def fit(
        self,
        frame: pd.DataFrame,
        y_true: np.ndarray,
        baseline_prediction: np.ndarray,
        rcot_prediction: np.ndarray,
    ) -> SelectiveRCOTGate:
        y = np.asarray(y_true, dtype=float).reshape(-1)
        baseline = np.asarray(baseline_prediction, dtype=float).reshape(-1)
        rcot = np.asarray(rcot_prediction, dtype=float).reshape(-1)

        if len(frame) != len(y) or len(y) != len(baseline) or len(y) != len(rcot):
            raise ValueError("Gate training arrays have inconsistent lengths")

        labels = (np.abs(y - rcot) < np.abs(y - baseline)).astype(int)
        self.positive_rate_ = float(labels.mean())

        self.encoder.fit(frame)
        features = self._features(frame, baseline, rcot)
        self.columns_ = list(features.columns)

        if len(np.unique(labels)) == 1:
            self.constant_probability = float(labels[0])
            self.model = None
            return self

        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=20,
            reg_lambda=3.0,
            min_split_gain=0.01,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=1,
        )
        self.model.fit(features, labels)
        self.constant_probability = None
        return self

    def predict_probability(
        self,
        frame: pd.DataFrame,
        baseline_prediction: np.ndarray,
        rcot_prediction: np.ndarray,
    ) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("SelectiveRCOTGate has not been fit")

        features = self._features(frame, baseline_prediction, rcot_prediction)
        features = features.reindex(columns=self.columns_, fill_value=0.0)

        if self.constant_probability is not None:
            return np.full(len(features), self.constant_probability, dtype=float)
        if self.model is None:
            raise RuntimeError("Gate model is unavailable")
        return self.model.predict_proba(features)[:, 1]


def _gated_prediction(
    baseline_prediction: np.ndarray,
    rcot_prediction: np.ndarray,
    gate_probability: np.ndarray,
    *,
    threshold: float,
    clip_minutes: float,
    blend: float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(baseline_prediction, dtype=float)
    rcot = np.asarray(rcot_prediction, dtype=float)
    probability = np.asarray(gate_probability, dtype=float)

    correction = np.clip(rcot - baseline, -clip_minutes, clip_minutes)
    active = probability >= threshold
    prediction = np.maximum(
        baseline + active.astype(float) * blend * correction,
        0.0,
    )
    return prediction, active


def _tail_mae(
    y_true: np.ndarray,
    prediction: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    tail_mask = y > np.asarray(thresholds, dtype=float)
    if not tail_mask.any():
        return float("nan")
    return float(point_metrics(y[tail_mask], pred[tail_mask])["mae"])


def _paired_slice_rows(
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
                "gated_rcot_mae": challenger_mae,
                "relative_improvement": float(
                    (baseline_mae - challenger_mae) / max(baseline_mae, 1e-9)
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological OOF selective RCOT experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "smoke.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "artifacts" / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "gated_rcot_experiment",
    )
    parser.add_argument("--oof-splits", type=int, default=5)
    args = parser.parse_args()

    config = load_config(_resolve_path(args.config))
    data_dir = _resolve_path(args.data_dir)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train, validation, test = _load_partitions(data_dir)
    model_config = _model_config(config)

    oof = _forward_oof_expert_predictions(
        train,
        model_config,
        n_splits=int(args.oof_splits),
    )

    gate = SelectiveRCOTGate(random_state=model_config.random_state).fit(
        oof,
        oof[TARGET].to_numpy(dtype=float),
        oof["_base_prediction"].to_numpy(dtype=float),
        oof["_rcot_prediction"].to_numpy(dtype=float),
    )

    baseline_model, rcot_model = _fit_experts(train, model_config)

    validation_y = validation[TARGET].to_numpy(dtype=float)
    validation_base = baseline_model.predict(validation)
    validation_rcot = rcot_model.predict(validation)
    validation_probability = gate.predict_probability(
        validation,
        validation_base,
        validation_rcot,
    )

    tail_thresholds = fit_tail_thresholds(train, TARGET)
    validation_tail_thresholds = threshold_for_rows(validation, tail_thresholds)

    baseline_validation_mae = float(point_metrics(validation_y, validation_base)["mae"])
    baseline_validation_tail_mae = _tail_mae(
        validation_y,
        validation_base,
        validation_tail_thresholds,
    )

    candidates: list[dict[str, object]] = []
    search_grid = product(
        [0.50, 0.60, 0.70, 0.80],
        [0.5, 1.0, 2.0, 4.0, 8.0, 1_000_000.0],
        [0.25, 0.50, 0.75, 1.00],
    )

    for threshold, clip_minutes, blend in search_grid:
        prediction, active = _gated_prediction(
            validation_base,
            validation_rcot,
            validation_probability,
            threshold=float(threshold),
            clip_minutes=float(clip_minutes),
            blend=float(blend),
        )

        promotion = evaluate_rcot_promotion(
            validation,
            validation_y,
            validation_base,
            prediction,
            validation_tail_thresholds,
        )

        validation_mae = float(point_metrics(validation_y, prediction)["mae"])
        validation_tail_mae = _tail_mae(
            validation_y,
            prediction,
            validation_tail_thresholds,
        )

        candidates.append(
            {
                "threshold": float(threshold),
                "clip_minutes": float(clip_minutes),
                "blend": float(blend),
                "activation_rate": float(active.mean()),
                "validation_mae": validation_mae,
                "validation_tail_mae": validation_tail_mae,
                "relative_mae_improvement": float(
                    (baseline_validation_mae - validation_mae) / max(baseline_validation_mae, 1e-9)
                ),
                "tail_mae_change": float(validation_tail_mae - baseline_validation_tail_mae),
                "promote": bool(promotion.promote),
                "promotion_reason": promotion.reason,
                "average_relative_improvement": float(promotion.average_relative_improvement),
                "tail_relative_improvement": float(promotion.tail_relative_improvement),
                "worst_slice_relative_regression": float(promotion.worst_slice_relative_regression),
            }
        )

    candidate_frame = pd.DataFrame(candidates).sort_values(
        ["validation_mae", "validation_tail_mae", "activation_rate"],
        ascending=[True, True, True],
        kind="stable",
    )

    eligible = candidate_frame.loc[
        candidate_frame["promote"] & candidate_frame["activation_rate"].between(0.02, 0.98)
    ].copy()

    baseline_choice = {
        "threshold": 1.10,
        "clip_minutes": 0.0,
        "blend": 0.0,
        "activation_rate": 0.0,
        "validation_mae": baseline_validation_mae,
        "validation_tail_mae": baseline_validation_tail_mae,
        "relative_mae_improvement": 0.0,
        "tail_mae_change": 0.0,
        "promote": False,
        "promotion_reason": "baseline fallback",
        "average_relative_improvement": 0.0,
        "tail_relative_improvement": 0.0,
        "worst_slice_relative_regression": 0.0,
    }

    if eligible.empty:
        selected = baseline_choice
        selected_name = "baseline_fallback"
    else:
        selected = eligible.iloc[0].to_dict()
        selected_name = "gated_rcot"

    test_y = test[TARGET].to_numpy(dtype=float)
    test_base = baseline_model.predict(test)
    test_rcot = rcot_model.predict(test)
    test_probability = gate.predict_probability(test, test_base, test_rcot)

    if selected_name == "baseline_fallback":
        test_gated = test_base.copy()
        test_active = np.zeros(len(test), dtype=bool)
    else:
        test_gated, test_active = _gated_prediction(
            test_base,
            test_rcot,
            test_probability,
            threshold=float(selected["threshold"]),
            clip_minutes=float(selected["clip_minutes"]),
            blend=float(selected["blend"]),
        )

    test_thresholds = threshold_for_rows(test, tail_thresholds)
    test_base_metrics = point_metrics(test_y, test_base)
    test_rcot_metrics = point_metrics(test_y, test_rcot)
    test_gated_metrics = point_metrics(test_y, test_gated)

    bootstrap = clustered_mae_difference_ci(
        test,
        test_y,
        test_base,
        test_gated,
        seed=model_config.random_state,
    )

    test_relative_improvement = float(
        (test_base_metrics["mae"] - test_gated_metrics["mae"])
        / max(float(test_base_metrics["mae"]), 1e-9)
    )

    ci_supports_improvement = bool(float(bootstrap["ci_upper"]) < 0.0)
    claim_supported = bool(
        selected_name == "gated_rcot"
        and test_relative_improvement > 0.0
        and ci_supports_improvement
    )

    scorecard = pd.DataFrame(
        [
            {"model": "lightgbm_without_rcot", **test_base_metrics},
            {"model": "lightgbm_with_rcot", **test_rcot_metrics},
            {"model": "gated_rcot", **test_gated_metrics},
        ]
    )
    scorecard.to_csv(output_dir / "gated_rcot_scorecard.csv", index=False)
    candidate_frame.to_csv(output_dir / "gated_rcot_validation_grid.csv", index=False)

    predictions = test[
        [
            column
            for column in [
                "snapshot_id",
                "courier_id",
                "work_date",
                "city",
                "route_phase",
            ]
            if column in test.columns
        ]
    ].copy()
    predictions["target"] = test_y
    predictions["baseline_prediction"] = test_base
    predictions["rcot_prediction"] = test_rcot
    predictions["gate_probability"] = test_probability
    predictions["gate_active"] = test_active.astype(int)
    predictions["gated_prediction"] = test_gated
    predictions["baseline_absolute_error"] = np.abs(test_y - test_base)
    predictions["gated_absolute_error"] = np.abs(test_y - test_gated)
    predictions.to_csv(output_dir / "gated_rcot_test_predictions.csv", index=False)

    slice_frame = test.reset_index(drop=True).copy()
    slice_frame["route_phase_band"] = pd.cut(
        slice_frame["route_phase"],
        bins=[-np.inf, 0.33, 0.67, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)
    slice_frame["rcot_trust_band"] = pd.cut(
        slice_frame["rcot_trust"],
        bins=[-np.inf, 0.20, 0.50, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    slice_rows: list[dict[str, object]] = []
    for column in ["city", "route_phase_band", "rcot_trust_band"]:
        slice_rows.extend(
            _paired_slice_rows(
                slice_frame,
                test_y,
                test_base,
                test_gated,
                column=column,
            )
        )
    pd.DataFrame(slice_rows).to_csv(
        output_dir / "gated_rcot_slice_comparison.csv",
        index=False,
    )

    if gate.model is not None and gate.columns_ is not None:
        importance = pd.DataFrame(
            {
                "feature": gate.columns_,
                "importance": gate.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False, kind="stable")
        importance.to_csv(
            output_dir / "gated_rcot_gate_importance.csv",
            index=False,
        )

    joblib.dump(gate, output_dir / "selective_rcot_gate.joblib")

    summary = {
        "experiment": "chronological_oof_selective_rcot",
        "status": "CLAIM_SUPPORTED" if claim_supported else "HOLD_BASELINE",
        "claim_supported": claim_supported,
        "claim_boundary": (
            "offline experiment on saved smoke partitions; not a full-scale or production ETA claim"
        ),
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "gate_oof_train": int(len(oof)),
        },
        "gate": {
            "oof_rcot_better_rate": gate.positive_rate_,
            "selected_model": selected_name,
            "selected_threshold": float(selected["threshold"]),
            "selected_clip_minutes": float(selected["clip_minutes"]),
            "selected_blend": float(selected["blend"]),
            "validation_activation_rate": float(selected["activation_rate"]),
            "test_activation_rate": float(test_active.mean()),
        },
        "validation": {
            "baseline_mae": baseline_validation_mae,
            "baseline_tail_mae": baseline_validation_tail_mae,
            "selected_mae": float(selected["validation_mae"]),
            "selected_tail_mae": float(selected["validation_tail_mae"]),
            "selected_relative_mae_improvement": float(selected["relative_mae_improvement"]),
            "promotion_reason": str(selected["promotion_reason"]),
        },
        "test": {
            "baseline": test_base_metrics,
            "rcot": test_rcot_metrics,
            "gated_rcot": test_gated_metrics,
            "baseline_tail_mae": _tail_mae(test_y, test_base, test_thresholds),
            "gated_tail_mae": _tail_mae(test_y, test_gated, test_thresholds),
            "relative_mae_improvement": test_relative_improvement,
            "bootstrap": bootstrap,
            "ci_supports_improvement": ci_supports_improvement,
        },
    }

    (output_dir / "gated_rcot_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
