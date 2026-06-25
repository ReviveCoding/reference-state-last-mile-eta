from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _paired_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(target) == 0:
        raise ValueError("Metric inputs are empty")
    if len(target) != len(prediction):
        raise ValueError("Metric inputs have different lengths")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric inputs contain non-finite values")
    return target, prediction


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    target, prediction = _paired_arrays(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(target, prediction))),
        "median_absolute_error": float(np.median(np.abs(target - prediction))),
        "signed_bias": float(np.mean(prediction - target)),
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    target, prediction = _paired_arrays(y_true, y_pred)
    residual = target - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def interval_metrics(y_true: np.ndarray, predictions: pd.DataFrame) -> dict[str, float]:
    required = {"q10", "q50", "q90"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing quantile columns: {sorted(missing)}")
    target = np.asarray(y_true, dtype=float).reshape(-1)
    values = predictions[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if len(target) == 0 or len(target) != len(values):
        raise ValueError("Interval inputs are empty or have different lengths")
    if not np.isfinite(target).all() or not np.isfinite(values).all():
        raise ValueError("Interval inputs contain non-finite values")
    lower, median, upper = values.T
    covered = (target >= lower) & (target <= upper)
    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "q10_pinball": pinball_loss(target, lower, 0.10),
        "q50_pinball": pinball_loss(target, median, 0.50),
        "q90_pinball": pinball_loss(target, upper, 0.90),
        "quantile_crossing_rate": float(np.mean((lower > median) | (median > upper))),
    }


def slice_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    slice_column: str,
) -> list[dict[str, Any]]:
    if slice_column not in frame.columns:
        raise ValueError(f"Missing slice column: {slice_column}")
    target, prediction = _paired_arrays(y_true, y_pred)
    if len(frame) != len(target):
        raise ValueError("Frame and metric inputs have different lengths")
    working = frame[[slice_column]].copy()
    working["y_true"] = target
    working["y_pred"] = prediction
    records: list[dict[str, Any]] = []
    for value, group in working.groupby(slice_column, dropna=False):
        records.append(
            {
                "slice": slice_column,
                "value": str(value),
                "count": int(len(group)),
                **point_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    return records
