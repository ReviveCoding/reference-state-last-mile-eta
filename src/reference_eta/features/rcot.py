from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


@dataclass
class _ReferenceCurve:
    elapsed: np.ndarray
    progress: np.ndarray
    inverse_progress: np.ndarray
    inverse_elapsed: np.ndarray
    support_rows: int
    support_groups: int
    dispersion_minutes: float

    def expected_progress(self, elapsed_minutes: float) -> float:
        return float(np.interp(elapsed_minutes, self.elapsed, self.progress))

    def elapsed_at_progress(self, observed_progress: float) -> float:
        if len(self.inverse_progress) == 1:
            return float(self.inverse_elapsed[0])
        return float(np.interp(observed_progress, self.inverse_progress, self.inverse_elapsed))


class ReferenceOperationalTimeTransformer:
    """Train-only support-aware reference operational time (RCOT) transformer.

    Support is measured with distinct courier-day groups, rather than correlated snapshot rows.
    ``fit_transform_cross_fitted`` prevents a training route from serving as its own reference.
    The fitted transformer is then refit on the complete training partition for future data.
    """

    feature_columns = [
        "initial_workload",
        "task_density",
        "aoi_transition_burden",
        "query_hour",
        "elapsed_minutes",
        "observed_progress",
    ]
    group_columns = ["courier_id", "work_date"]

    def __init__(
        self,
        *,
        min_cohort_rows: int = 20,
        min_cohort_groups: int = 5,
        support_shrinkage: float = 12.0,
        max_dispersion_minutes: float = 45.0,
    ) -> None:
        self.min_cohort_rows = int(min_cohort_rows)
        self.min_cohort_groups = int(min_cohort_groups)
        self.support_shrinkage = float(support_shrinkage)
        self.max_dispersion_minutes = float(max_dispersion_minutes)
        self.curves_: dict[tuple[object, ...], _ReferenceCurve] = {}
        self.workload_edges_: np.ndarray | None = None
        self.density_edges_: np.ndarray | None = None
        self.density_median_: float | None = None
        self.transition_median_: float | None = None
        self.fitted_: bool = False

    def _new_like(self) -> ReferenceOperationalTimeTransformer:
        return type(self)(
            min_cohort_rows=self.min_cohort_rows,
            min_cohort_groups=self.min_cohort_groups,
            support_shrinkage=self.support_shrinkage,
            max_dispersion_minutes=self.max_dispersion_minutes,
        )

    @staticmethod
    def _safe_quantile_edges(series: pd.Series) -> np.ndarray:
        values = series.astype(float).to_numpy()
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("RCOT binning features must be finite and non-empty")
        edges = np.quantile(values, [0.0, 0.33, 0.67, 1.0])
        edges = np.unique(edges)
        if len(edges) < 2:
            value = float(values[0])
            edges = np.array([value - 1.0, value + 1.0])
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @staticmethod
    def _bin(value: float, edges: np.ndarray) -> int:
        return int(np.clip(np.digitize([value], edges[1:-1], right=False)[0], 0, len(edges) - 2))

    def _regime(self, row: pd.Series) -> str:
        density = float(row["task_density"])
        transition = float(row["aoi_transition_burden"])
        assert self.density_median_ is not None and self.transition_median_ is not None
        if transition > max(0.55, self.transition_median_ * 1.15):
            return "high_transition"
        if density >= self.density_median_:
            return "dense_service"
        return "sparse_travel"

    def _keys(self, row: pd.Series) -> Iterable[tuple[object, ...]]:
        assert self.workload_edges_ is not None and self.density_edges_ is not None
        workload_bin = self._bin(float(row["initial_workload"]), self.workload_edges_)
        density_bin = self._bin(float(row["task_density"]), self.density_edges_)
        hour_bucket = int(float(row["query_hour"]) // 3)
        regime = self._regime(row)
        city = str(row["city"])
        yield ("fine", city, hour_bucket, workload_bin, density_bin, regime)
        yield ("city_workload", city, workload_bin, regime)
        yield ("city", city, regime)
        yield ("global", regime)
        yield ("global", "all")

    @classmethod
    def _fit_curve(cls, group: pd.DataFrame) -> _ReferenceCurve:
        ordered = group.sort_values("elapsed_minutes")
        elapsed = ordered["elapsed_minutes"].astype(float).to_numpy()
        observed_progress = ordered["observed_progress"].astype(float).to_numpy()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        fitted_progress = np.asarray(iso.fit_transform(elapsed, observed_progress), dtype=float)

        # Multiple rows can share an elapsed time or an isotonic plateau. Median aggregation
        # avoids choosing the first edge of a plateau as the inverse operational time.
        forward = pd.DataFrame({"elapsed": elapsed, "progress": fitted_progress})
        forward = forward.groupby("elapsed", as_index=False, sort=True)["progress"].median()
        forward_progress = np.maximum.accumulate(forward["progress"].to_numpy(dtype=float))

        inverse = pd.DataFrame(
            {
                "progress": np.round(fitted_progress, 12),
                "elapsed": elapsed,
            }
        )
        inverse = inverse.groupby("progress", as_index=False, sort=True)["elapsed"].median()
        inverse_progress = inverse["progress"].to_numpy(dtype=float)
        inverse_elapsed = np.maximum.accumulate(inverse["elapsed"].to_numpy(dtype=float))

        group_count = int(group[cls.group_columns].drop_duplicates().shape[0])
        curve = _ReferenceCurve(
            elapsed=forward["elapsed"].to_numpy(dtype=float),
            progress=forward_progress,
            inverse_progress=inverse_progress,
            inverse_elapsed=inverse_elapsed,
            support_rows=len(group),
            support_groups=group_count,
            dispersion_minutes=0.0,
        )
        mapped_elapsed = np.interp(observed_progress, inverse_progress, inverse_elapsed)
        residual = elapsed - mapped_elapsed
        q25, q75 = np.quantile(residual, [0.25, 0.75])
        curve.dispersion_minutes = float(max(q75 - q25, 1.0))
        return curve

    def fit(self, train: pd.DataFrame) -> ReferenceOperationalTimeTransformer:
        missing = set(self.feature_columns + ["city", *self.group_columns]).difference(
            train.columns
        )
        if missing:
            raise ValueError(f"Missing RCOT columns: {sorted(missing)}")
        if train.empty:
            raise ValueError("RCOT cannot be fit on an empty frame")

        self.workload_edges_ = self._safe_quantile_edges(train["initial_workload"])
        self.density_edges_ = self._safe_quantile_edges(train["task_density"])
        self.density_median_ = float(train["task_density"].median())
        self.transition_median_ = float(train["aoi_transition_burden"].median())

        expanded = train.copy()
        expanded["_regime"] = expanded.apply(self._regime, axis=1)
        expanded["_workload_bin"] = expanded["initial_workload"].map(
            lambda value: self._bin(float(value), self.workload_edges_)
        )
        expanded["_density_bin"] = expanded["task_density"].map(
            lambda value: self._bin(float(value), self.density_edges_)
        )
        expanded["_hour_bucket"] = (expanded["query_hour"].astype(float) // 3).astype(int)

        group_specs = {
            "fine": ["city", "_hour_bucket", "_workload_bin", "_density_bin", "_regime"],
            "city_workload": ["city", "_workload_bin", "_regime"],
            "city": ["city", "_regime"],
            "global": ["_regime"],
        }
        self.curves_.clear()
        for level, columns in group_specs.items():
            for values, group in expanded.groupby(columns, observed=True):
                values = values if isinstance(values, tuple) else (values,)
                unique_groups = group[self.group_columns].drop_duplicates().shape[0]
                sufficiently_supported = (
                    len(group) >= self.min_cohort_rows and unique_groups >= self.min_cohort_groups
                )
                if sufficiently_supported or level in {"city", "global"}:
                    self.curves_[(level, *values)] = self._fit_curve(group)
        self.curves_[("global", "all")] = self._fit_curve(expanded)
        self.fitted_ = True
        return self

    def _validate_transform_frame(self, frame: pd.DataFrame) -> None:
        missing = set(self.feature_columns + ["city"]).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing RCOT transform columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("RCOT transform frame is empty")
        numeric = frame[self.feature_columns].astype(float).to_numpy()
        if not np.isfinite(numeric).all():
            raise ValueError("RCOT transform features must be finite")
        if (frame["elapsed_minutes"].astype(float) < 0.0).any():
            raise ValueError("elapsed_minutes cannot be negative")
        if not frame["observed_progress"].astype(float).between(0.0, 1.0).all():
            raise ValueError("observed_progress must be between 0 and 1")

    def _neutral_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        elapsed = frame["elapsed_minutes"].astype(float).to_numpy()
        neutral = pd.DataFrame(
            {
                "rcot_minutes": np.maximum(elapsed, 0.0),
                "progress_gap": np.zeros(len(frame)),
                "pace_ratio": np.ones(len(frame)),
                "reference_support": np.zeros(len(frame)),
                "reference_support_groups": np.zeros(len(frame)),
                "reference_support_rows": np.zeros(len(frame)),
                "reference_dispersion": np.full(len(frame), self.max_dispersion_minutes),
                "reference_ood_probability": np.ones(len(frame)),
                "rcot_trust": np.zeros(len(frame)),
                "reference_level": ["temporal_warmup"] * len(frame),
                "reference_regime": ["unavailable"] * len(frame),
            }
        )
        return pd.concat([frame.reset_index(drop=True), neutral], axis=1)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("RCOT transformer has not been fit")
        self._validate_transform_frame(frame)

        output = frame.copy()
        results: list[dict[str, float | str]] = []
        for _, row in frame.iterrows():
            curve = None
            selected_key: tuple[object, ...] | None = None
            for key in self._keys(row):
                candidate = self.curves_.get(key)
                if candidate is not None:
                    curve = candidate
                    selected_key = key
                    break
            if curve is None or selected_key is None:
                raise RuntimeError("Global RCOT fallback is missing")
            elapsed = float(row["elapsed_minutes"])
            progress = float(row["observed_progress"])
            expected = curve.expected_progress(elapsed)
            rcot = curve.elapsed_at_progress(progress)
            support = curve.support_groups / (curve.support_groups + self.support_shrinkage)
            dispersion_scaled = min(curve.dispersion_minutes / self.max_dispersion_minutes, 1.0)
            progress_std = float(np.std(curve.progress))
            ood_distance = float(abs(progress - expected) / max(0.10, progress_std + 0.05))
            ood_probability = float(1.0 - np.exp(-max(ood_distance - 1.0, 0.0)))
            trust = float(support * np.exp(-dispersion_scaled) * (1.0 - ood_probability))
            results.append(
                {
                    "rcot_minutes": max(rcot, 0.0),
                    "progress_gap": progress - expected,
                    "pace_ratio": max(rcot, 0.0) / max(elapsed, 1.0),
                    "reference_support": support,
                    "reference_support_groups": float(curve.support_groups),
                    "reference_support_rows": float(curve.support_rows),
                    "reference_dispersion": curve.dispersion_minutes,
                    "reference_ood_probability": ood_probability,
                    "rcot_trust": trust,
                    "reference_level": str(selected_key[0]),
                    "reference_regime": self._regime(row),
                }
            )
        return pd.concat([output.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    def fit_transform_cross_fitted(
        self,
        train: pd.DataFrame,
        *,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Create forward-chained RCOT features, then fit on all training rows.

        Training dates are divided into chronological blocks. Each block is transformed only
        with strictly earlier dates. The first block has no valid reference history and receives
        neutral, zero-trust RCOT features. This prevents both same-route self-reference and
        future-reference leakage while retaining every training row.
        """

        missing = set(self.group_columns).difference(train.columns)
        if missing:
            raise ValueError(f"Missing RCOT cross-fit group columns: {sorted(missing)}")
        if train.empty:
            raise ValueError("RCOT cross-fit frame is empty")
        self._validate_transform_frame(train)
        ordered_dates = np.array(sorted(train["work_date"].astype(str).unique()))
        if len(ordered_dates) < 2:
            raise ValueError("Temporal RCOT cross-fitting requires at least two work dates")
        split_count = max(2, min(int(n_splits), len(ordered_dates)))
        date_blocks = [block for block in np.array_split(ordered_dates, split_count) if len(block)]
        transformed_parts: list[pd.DataFrame] = []

        for block_index, held_out_dates in enumerate(date_blocks):
            held_mask = train["work_date"].astype(str).isin(set(held_out_dates.tolist()))
            held_positions = np.flatnonzero(held_mask.to_numpy())
            held_frame = train.iloc[held_positions]
            if block_index == 0:
                transformed = self._neutral_features(held_frame)
            else:
                prior_dates = np.concatenate(date_blocks[:block_index]).tolist()
                fit_mask = train["work_date"].astype(str).isin(set(prior_dates))
                fold_transformer = self._new_like().fit(train.loc[fit_mask])
                transformed = fold_transformer.transform(held_frame)
            transformed["_original_position"] = held_positions
            transformed_parts.append(transformed)

        transformed = pd.concat(transformed_parts, ignore_index=True)
        transformed = transformed.sort_values("_original_position", kind="stable")
        transformed = transformed.drop(columns="_original_position").reset_index(drop=True)
        if len(transformed) != len(train):
            raise RuntimeError("Temporal RCOT cross-fitting lost training rows")
        self.fit(train)
        return transformed

    def save(self, path: str) -> None:
        from pathlib import Path

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str) -> ReferenceOperationalTimeTransformer:
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
