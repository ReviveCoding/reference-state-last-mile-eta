from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from reference_eta.evaluation.metrics import point_metrics


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    average_relative_improvement: float
    tail_relative_improvement: float
    worst_slice_relative_regression: float
    average_noninferior: bool
    meaningful_gain: bool
    worst_slice_safe: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_improvement(baseline: float, challenger: float) -> float:
    return float((baseline - challenger) / max(abs(baseline), 1e-9))


def _promotion_inputs(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    challenger_pred: np.ndarray,
    tail_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty or "city" not in frame.columns:
        raise ValueError("Promotion frame must be nonempty and include city")
    vectors = []
    for name, values in (
        ("y_true", y_true),
        ("baseline_pred", baseline_pred),
        ("challenger_pred", challenger_pred),
        ("tail_thresholds", tail_thresholds),
    ):
        vector = np.asarray(values, dtype=float).reshape(-1)
        if len(vector) != len(frame) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must match frame length and be finite")
        vectors.append(vector)
    if (vectors[0] < 0.0).any() or (vectors[1] < 0.0).any() or (vectors[2] < 0.0).any():
        raise ValueError("ETA targets and predictions must be nonnegative")
    return tuple(vectors)  # type: ignore[return-value]


def _worst_slice_regression(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    challenger_pred: np.ndarray,
    *,
    slice_column: str,
) -> float:
    working = frame[[slice_column]].copy()
    working["y_true"] = y_true
    working["baseline"] = baseline_pred
    working["challenger"] = challenger_pred
    regressions: list[float] = []
    for _, group in working.groupby(slice_column, dropna=False):
        baseline_mae = point_metrics(group["y_true"].to_numpy(), group["baseline"].to_numpy())[
            "mae"
        ]
        challenger_mae = point_metrics(group["y_true"].to_numpy(), group["challenger"].to_numpy())[
            "mae"
        ]
        regressions.append((challenger_mae - baseline_mae) / max(baseline_mae, 1e-9))
    return float(max(regressions, default=0.0))


def evaluate_rcot_promotion(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    challenger_pred: np.ndarray,
    tail_thresholds: np.ndarray,
    *,
    min_average_gain: float = 0.005,
    min_tail_gain: float = 0.02,
    average_noninferiority_margin: float = 0.01,
    worst_slice_regression_limit: float = 0.05,
) -> PromotionDecision:
    parameters = {
        "min_average_gain": min_average_gain,
        "min_tail_gain": min_tail_gain,
        "average_noninferiority_margin": average_noninferiority_margin,
        "worst_slice_regression_limit": worst_slice_regression_limit,
    }
    if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in parameters.values()):
        raise ValueError("Promotion thresholds must be finite and nonnegative")
    y_true, baseline_pred, challenger_pred, tail_thresholds = _promotion_inputs(
        frame, y_true, baseline_pred, challenger_pred, tail_thresholds
    )

    baseline_mae = point_metrics(y_true, baseline_pred)["mae"]
    challenger_mae = point_metrics(y_true, challenger_pred)["mae"]
    average_gain = _relative_improvement(baseline_mae, challenger_mae)

    tail_mask = np.asarray(y_true, dtype=float) > np.asarray(tail_thresholds, dtype=float)
    if tail_mask.any():
        baseline_tail = point_metrics(y_true[tail_mask], baseline_pred[tail_mask])["mae"]
        challenger_tail = point_metrics(y_true[tail_mask], challenger_pred[tail_mask])["mae"]
        tail_gain = _relative_improvement(baseline_tail, challenger_tail)
    else:
        tail_gain = 0.0

    worst_slice_regression = _worst_slice_regression(
        frame,
        y_true,
        baseline_pred,
        challenger_pred,
        slice_column="city",
    )
    average_noninferior = average_gain >= -average_noninferiority_margin
    meaningful_gain = average_gain >= min_average_gain or tail_gain >= min_tail_gain
    worst_slice_safe = worst_slice_regression <= worst_slice_regression_limit
    promote = bool(average_noninferior and meaningful_gain and worst_slice_safe)

    failures = []
    if not average_noninferior:
        failures.append("average MAE exceeds non-inferiority margin")
    if not meaningful_gain:
        failures.append("no meaningful average or tail gain")
    if not worst_slice_safe:
        failures.append("worst-city regression exceeds limit")
    reason = "all promotion criteria passed" if promote else "; ".join(failures)
    return PromotionDecision(
        promote=promote,
        average_relative_improvement=average_gain,
        tail_relative_improvement=tail_gain,
        worst_slice_relative_regression=worst_slice_regression,
        average_noninferior=average_noninferior,
        meaningful_gain=meaningful_gain,
        worst_slice_safe=worst_slice_safe,
        reason=reason,
    )


def evaluate_system_release(
    *,
    interval_coverage: float,
    target_coverage: float,
    quantile_crossing_rate: float,
    predictions: pd.DataFrame,
    test_rows: int,
    champion_mae: float,
    business_baseline_mae: float,
    decision_excess_capture: float,
    random_excess_capture: float,
    test_tail_events: int,
    coverage_tolerance: float = 0.10,
    minimum_test_rows: int = 50,
    maximum_champion_relative_regression: float = 0.02,
    minimum_test_tail_events: int = 1,
) -> dict[str, object]:
    required = {"q10", "q50", "q90"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Release predictions are missing columns: {sorted(missing)}")
    if predictions.empty or int(test_rows) != len(predictions):
        raise ValueError("test_rows must equal the nonzero prediction row count")
    scalar_values = {
        "interval_coverage": interval_coverage,
        "target_coverage": target_coverage,
        "quantile_crossing_rate": quantile_crossing_rate,
        "coverage_tolerance": coverage_tolerance,
        "champion_mae": champion_mae,
        "business_baseline_mae": business_baseline_mae,
        "decision_excess_capture": decision_excess_capture,
        "random_excess_capture": random_excess_capture,
        "maximum_champion_relative_regression": maximum_champion_relative_regression,
    }
    if any(not np.isfinite(float(value)) for value in scalar_values.values()):
        raise ValueError("Release metrics must be finite")
    if not 0.0 <= float(interval_coverage) <= 1.0:
        raise ValueError("interval_coverage must be within [0, 1]")
    if not 0.0 < float(target_coverage) < 1.0:
        raise ValueError("target_coverage must be within (0, 1)")
    if not 0.0 <= float(quantile_crossing_rate) <= 1.0:
        raise ValueError("quantile_crossing_rate must be within [0, 1]")
    if not 0.0 < float(coverage_tolerance) < 1.0:
        raise ValueError("coverage_tolerance must be within (0, 1)")
    if int(minimum_test_rows) < 1:
        raise ValueError("minimum_test_rows must be positive")
    if int(minimum_test_tail_events) < 1 or int(test_tail_events) < 0:
        raise ValueError("Tail-event counts must be nonnegative with a positive minimum")
    if float(champion_mae) < 0.0 or float(business_baseline_mae) <= 0.0:
        raise ValueError("Release MAE values must be nonnegative with a positive baseline")
    if (
        not 0.0 <= float(decision_excess_capture) <= 1.0
        or not 0.0 <= float(random_excess_capture) <= 1.0
    ):
        raise ValueError("Decision capture metrics must be within [0, 1]")
    if float(maximum_champion_relative_regression) < 0.0:
        raise ValueError("maximum_champion_relative_regression must be nonnegative")

    values = predictions[["q10", "q50", "q90"]].to_numpy(dtype=float)
    finite = bool(np.isfinite(values).all())
    nonnegative = bool((values >= 0.0).all())
    actual_crossing_rate = float(
        np.mean((values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2]))
    )
    if not np.isclose(actual_crossing_rate, float(quantile_crossing_rate), atol=1e-12):
        raise ValueError("quantile_crossing_rate does not match the supplied predictions")
    coverage_ok = bool(abs(interval_coverage - target_coverage) <= coverage_tolerance)
    crossing_ok = bool(actual_crossing_rate == 0.0)
    enough_rows = bool(test_rows >= minimum_test_rows)
    champion_relative_regression = float(
        (champion_mae - business_baseline_mae) / max(business_baseline_mae, 1e-9)
    )
    champion_predictive_quality = bool(
        champion_relative_regression <= maximum_champion_relative_regression
    )
    decision_noninferior_to_random = bool(decision_excess_capture + 1e-12 >= random_excess_capture)
    enough_tail_events = bool(int(test_tail_events) >= int(minimum_test_tail_events))
    checks = {
        "finite_predictions": finite,
        "nonnegative_predictions": nonnegative,
        "coverage_within_tolerance": coverage_ok,
        "zero_quantile_crossing": crossing_ok,
        "minimum_test_rows": enough_rows,
        "champion_predictive_quality": champion_predictive_quality,
        "decision_noninferior_to_random": decision_noninferior_to_random,
        "minimum_test_tail_events": enough_tail_events,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "coverage_error": float(abs(interval_coverage - target_coverage)),
        "coverage_tolerance": float(coverage_tolerance),
        "champion_relative_regression": champion_relative_regression,
        "maximum_champion_relative_regression": float(maximum_champion_relative_regression),
        "decision_excess_capture": float(decision_excess_capture),
        "random_excess_capture": float(random_excess_capture),
        "test_tail_events": int(test_tail_events),
        "minimum_test_tail_events": int(minimum_test_tail_events),
    }
