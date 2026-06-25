from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = ("q10", "q50", "q90")


def _validate_predictions(predictions: pd.DataFrame) -> np.ndarray:
    missing = set(_REQUIRED_COLUMNS).difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing quantile columns: {sorted(missing)}")
    values = predictions.loc[:, _REQUIRED_COLUMNS].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Predictions are empty")
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain non-finite values")
    return values


def _finite_sample_level(n: int, target_coverage: float) -> float:
    if n <= 0:
        raise ValueError("Calibration sample size must be positive")
    return min(float(np.ceil((n + 1) * target_coverage) / n), 1.0)


@dataclass
class ConformalQuantileCalibrator:
    """Split conformal correction for lower/upper quantile predictions.

    The class reports marginal offline coverage under the calibration protocol. It does not
    claim conditional coverage for individual routes or arbitrary distribution shift.
    """

    target_coverage: float = 0.80
    correction_: float | None = None
    calibration_size_: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_coverage) < 1.0:
            raise ValueError("target_coverage must be strictly between 0 and 1")
        self.target_coverage = float(self.target_coverage)

    def fit(self, predictions: pd.DataFrame, y_true: np.ndarray) -> ConformalQuantileCalibrator:
        values = _validate_predictions(predictions)
        target = np.asarray(y_true, dtype=float).reshape(-1)
        if len(target) != len(values):
            raise ValueError(
                f"Prediction/target length mismatch: {len(values)} predictions, {len(target)} targets"
            )
        if not np.isfinite(target).all():
            raise ValueError("Calibration targets contain non-finite values")
        lower, upper = values[:, 0], values[:, 2]
        scores = np.maximum(lower - target, target - upper)
        level = _finite_sample_level(len(scores), self.target_coverage)
        self.correction_ = float(np.quantile(scores, level, method="higher"))
        self.calibration_size_ = int(len(scores))
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if self.correction_ is None:
            raise RuntimeError("Calibrator has not been fit")
        if not np.isfinite(self.correction_):
            raise RuntimeError("Calibrator correction is non-finite")
        values = _validate_predictions(predictions)
        calibrated = predictions.copy()
        median = np.maximum(values[:, 1], 0.0)
        lower = np.minimum(np.maximum(values[:, 0] - self.correction_, 0.0), median)
        upper = np.maximum(np.maximum(values[:, 2] + self.correction_, 0.0), median)
        calibrated.loc[:, _REQUIRED_COLUMNS] = np.column_stack([lower, median, upper])
        return calibrated

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str) -> ConformalQuantileCalibrator:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value
