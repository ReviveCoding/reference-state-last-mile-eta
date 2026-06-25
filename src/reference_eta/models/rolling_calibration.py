from __future__ import annotations

import heapq
from collections import deque

import numpy as np
import pandas as pd


class RollingConformalReplay:
    """Chronological calibration replay with optional delayed outcome availability.

    A prediction's nonconformity score enters the rolling calibration window only after its
    outcome availability time. This is a monitoring and recalibration simulation, not model
    selection, and it does not imply individual conditional coverage.
    """

    def __init__(self, *, target_coverage: float = 0.80, window_size: int = 200) -> None:
        if not 0.0 < float(target_coverage) < 1.0:
            raise ValueError("target_coverage must be strictly between 0 and 1")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        self.target_coverage = float(target_coverage)
        self.window_size = int(window_size)

    @staticmethod
    def nonconformity(lower: float, upper: float, target: float) -> float:
        values = np.asarray([lower, upper, target], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonconformity inputs must be finite")
        return float(max(lower - target, target - upper))

    def _correction(self, scores: deque[float]) -> float:
        if not scores:
            return 0.0
        values = np.asarray(scores, dtype=float)
        level = min(float(np.ceil((len(values) + 1) * self.target_coverage) / len(values)), 1.0)
        return float(np.quantile(values, level, method="higher"))

    def replay(
        self,
        initial_scores: np.ndarray,
        predictions: pd.DataFrame,
        y_true: np.ndarray,
        *,
        prediction_times: pd.Series | np.ndarray | None = None,
        label_available_times: pd.Series | np.ndarray | None = None,
    ) -> pd.DataFrame:
        required = {"q10", "q50", "q90"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"Predictions are missing quantile columns: {sorted(missing)}")
        pred_values = predictions[["q10", "q50", "q90"]].to_numpy(dtype=float)
        targets = np.asarray(y_true, dtype=float).reshape(-1)
        initial = np.asarray(initial_scores, dtype=float).reshape(-1)
        if len(pred_values) != len(targets):
            raise ValueError("Prediction/target length mismatch")
        if len(pred_values) == 0:
            raise ValueError("Replay predictions are empty")
        if not np.isfinite(pred_values).all() or not np.isfinite(targets).all():
            raise ValueError("Replay inputs contain non-finite values")
        if not np.isfinite(initial).all():
            raise ValueError("Initial calibration scores contain non-finite values")

        delayed = prediction_times is not None or label_available_times is not None
        if delayed and (prediction_times is None or label_available_times is None):
            raise ValueError("prediction_times and label_available_times must be provided together")
        if delayed:
            predicted_at = pd.to_datetime(pd.Series(prediction_times), errors="raise")
            available_at = pd.to_datetime(pd.Series(label_available_times), errors="raise")
            if len(predicted_at) != len(targets) or len(available_at) != len(targets):
                raise ValueError("Replay timestamps must match prediction length")
            if not predicted_at.is_monotonic_increasing:
                raise ValueError("prediction_times must be nondecreasing")
            if (available_at < predicted_at).any():
                raise ValueError("A label cannot be available before its prediction")
        else:
            predicted_at = pd.Series([pd.NaT] * len(targets))
            available_at = pd.Series([pd.NaT] * len(targets))

        scores: deque[float] = deque((float(v) for v in initial), maxlen=self.window_size)
        pending: list[tuple[int, int, float]] = []
        rows: list[dict[str, float | int | str]] = []
        for index, (raw_lower, median, raw_upper) in enumerate(pred_values):
            released_count = 0
            if delayed:
                current_ns = int(predicted_at.iloc[index].value)
                while pending and pending[0][0] <= current_ns:
                    _, _, released_score = heapq.heappop(pending)
                    scores.append(released_score)
                    released_count += 1

            correction = self._correction(scores)
            median_value = max(float(median), 0.0)
            lower = min(max(float(raw_lower) - correction, 0.0), median_value)
            upper = max(max(float(raw_upper) + correction, 0.0), median_value)
            target = float(targets[index])
            covered = int(lower <= target <= upper)
            raw_score = self.nonconformity(float(raw_lower), float(raw_upper), target)
            rows.append(
                {
                    "step": index,
                    "prediction_time": (
                        predicted_at.iloc[index].isoformat() if delayed else "immediate"
                    ),
                    "label_available_time": (
                        available_at.iloc[index].isoformat() if delayed else "immediate"
                    ),
                    "lower": lower,
                    "median": median_value,
                    "upper": upper,
                    "target": target,
                    "covered": covered,
                    "interval_width": upper - lower,
                    "correction": correction,
                    "available_score_count": len(scores),
                    "released_scores_this_step": released_count,
                    "pending_label_count": len(pending) + int(delayed),
                }
            )
            if delayed:
                heapq.heappush(pending, (int(available_at.iloc[index].value), index, raw_score))
            else:
                scores.append(raw_score)
        result = pd.DataFrame(rows)
        result["rolling_coverage_25"] = result["covered"].rolling(25, min_periods=1).mean()
        return result
