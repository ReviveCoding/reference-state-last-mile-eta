from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from reference_eta.features.tabular import TabularFeatureEncoder


def _safe_edges(values: pd.Series) -> np.ndarray:
    array = values.astype(float).to_numpy()
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Binning values must be finite and non-empty")
    edges = np.unique(np.quantile(array, [0.0, 0.33, 0.67, 1.0]))
    if len(edges) < 2:
        value = float(array[0])
        edges = np.array([value - 1.0, value + 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _validate_target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in frame.columns:
        raise ValueError(f"Missing target column: {target}")
    values = frame[target].astype(float)
    if frame.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Training target must be finite and non-empty")
    if (values < 0.0).any():
        raise ValueError("ETA targets cannot be negative")
    return values


class CohortMedianRegressor:
    def __init__(self) -> None:
        self.workload_edges_: np.ndarray | None = None
        self.phase_edges_: np.ndarray | None = None
        self.medians_: dict[tuple[object, ...], float] = {}

    def fit(self, frame: pd.DataFrame, target: str) -> CohortMedianRegressor:
        y = _validate_target(frame, target)
        self.workload_edges_ = _safe_edges(frame["initial_workload"])
        self.phase_edges_ = np.array([-np.inf, 0.33, 0.67, np.inf])
        work_bin = np.digitize(frame["initial_workload"], self.workload_edges_[1:-1])
        phase_bin = np.digitize(frame["route_phase"], self.phase_edges_[1:-1])
        enriched = frame.assign(_work_bin=work_bin, _phase_bin=phase_bin, _target=y)
        self.medians_.clear()
        for (city, workload_bin, phase), group in enriched.groupby(
            ["city", "_work_bin", "_phase_bin"]
        ):
            self.medians_[("fine", city, int(workload_bin), int(phase))] = float(
                group["_target"].median()
            )
        for city, group in enriched.groupby("city"):
            self.medians_[("city", city)] = float(group["_target"].median())
        self.medians_[("global",)] = float(enriched["_target"].median())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.workload_edges_ is None or self.phase_edges_ is None:
            raise RuntimeError("CohortMedianRegressor has not been fit")
        predictions: list[float] = []
        for _, row in frame.iterrows():
            workload_bin = int(
                np.digitize([row["initial_workload"]], self.workload_edges_[1:-1])[0]
            )
            phase = int(np.digitize([row["route_phase"]], self.phase_edges_[1:-1])[0])
            fine = self.medians_.get(("fine", row["city"], workload_bin, phase))
            city = self.medians_.get(("city", row["city"]))
            value = (
                fine
                if fine is not None
                else city
                if city is not None
                else self.medians_[("global",)]
            )
            predictions.append(max(float(value), 0.0))
        return np.asarray(predictions, dtype=float)


@dataclass
class LightGBMConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_estimators < 1 or self.num_leaves < 2 or self.min_child_samples < 1:
            raise ValueError("LightGBM tree parameters are invalid")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


class LightGBMPointModel:
    def __init__(self, config: LightGBMConfig, *, include_rcot: bool) -> None:
        self.config = config
        self.encoder = TabularFeatureEncoder(include_rcot=include_rcot)
        self.model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            num_leaves=config.num_leaves,
            min_child_samples=config.min_child_samples,
            random_state=config.random_state,
            verbosity=-1,
            n_jobs=1,
        )
        self.fitted_ = False

    def fit(self, frame: pd.DataFrame, target: str) -> LightGBMPointModel:
        y = _validate_target(frame, target)
        features = self.encoder.fit_transform(frame)
        self.model.fit(features, y)
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("LightGBMPointModel has not been fit")
        return np.maximum(self.model.predict(self.encoder.transform(frame)), 0.0)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> LightGBMPointModel:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value


class QuantileLightGBMModel:
    def __init__(
        self,
        config: LightGBMConfig,
        *,
        include_rcot: bool,
        quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
    ) -> None:
        if tuple(sorted(quantiles)) != quantiles or len(quantiles) != 3:
            raise ValueError("quantiles must contain exactly three ordered values")
        if any(not 0.0 < quantile < 1.0 for quantile in quantiles):
            raise ValueError("quantiles must lie in (0, 1)")
        self.config = config
        self.quantiles = quantiles
        self.encoder = TabularFeatureEncoder(include_rcot=include_rcot)
        self.models: dict[float, LGBMRegressor] = {}

    def fit(self, frame: pd.DataFrame, target: str) -> QuantileLightGBMModel:
        y = _validate_target(frame, target)
        features = self.encoder.fit_transform(frame)
        self.models.clear()
        for quantile in self.quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                random_state=self.config.random_state + int(quantile * 100),
                verbosity=-1,
                n_jobs=1,
            )
            model.fit(features, y)
            self.models[quantile] = model
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(self.models) != set(self.quantiles):
            raise RuntimeError("QuantileLightGBMModel has not been fit")
        features = self.encoder.transform(frame)
        raw = np.column_stack(
            [self.models[quantile].predict(features) for quantile in self.quantiles]
        )
        raw = np.maximum(raw, 0.0)
        ordered = np.sort(raw, axis=1)
        return pd.DataFrame(ordered, columns=["q10", "q50", "q90"], index=frame.index)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> QuantileLightGBMModel:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value
