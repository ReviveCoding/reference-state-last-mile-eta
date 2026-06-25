from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_NUMERIC_FEATURES = [
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

RCOT_NUMERIC_FEATURES = [
    "rcot_minutes",
    "progress_gap",
    "pace_ratio",
    "reference_support",
    "reference_dispersion",
    "reference_ood_probability",
    "rcot_trust",
]


class TabularFeatureEncoder:
    def __init__(self, *, include_rcot: bool = True) -> None:
        self.include_rcot = include_rcot
        self.columns_: list[str] | None = None
        self.known_cities_: list[str] | None = None

    @property
    def numeric_features(self) -> list[str]:
        return BASE_NUMERIC_FEATURES + (RCOT_NUMERIC_FEATURES if self.include_rcot else [])

    def _validate(self, frame: pd.DataFrame) -> None:
        required = set(self.numeric_features + ["city"])
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing tabular features: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Tabular feature frame is empty")
        numeric = frame[self.numeric_features].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("Tabular numeric features contain non-finite values")
        if frame["city"].isna().any() or (frame["city"].astype(str).str.len() == 0).any():
            raise ValueError("city must be non-null and non-empty")

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.known_cities_ is None:
            raise RuntimeError("Feature encoder has not been fit")
        working = frame[self.numeric_features + ["city"]].copy()
        known = set(self.known_cities_)
        working["city"] = (
            working["city"]
            .astype(str)
            .where(working["city"].astype(str).isin(known), "__UNKNOWN__")
        )
        categories = [*self.known_cities_, "__UNKNOWN__"]
        working["city"] = pd.Categorical(working["city"], categories=categories)
        return pd.get_dummies(working, columns=["city"], dtype=float)

    def fit(self, frame: pd.DataFrame) -> TabularFeatureEncoder:
        self._validate(frame)
        self.known_cities_ = sorted(frame["city"].astype(str).unique().tolist())
        encoded = self._encode(frame)
        self.columns_ = list(encoded.columns)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("Feature encoder has not been fit")
        self._validate(frame)
        encoded = self._encode(frame)
        return encoded.reindex(columns=self.columns_, fill_value=0.0).astype(float)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str) -> TabularFeatureEncoder:
        encoder = joblib.load(path)
        if not isinstance(encoder, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(encoder).__name__}")
        return encoder
