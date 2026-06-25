from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_numeric_values(series: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.isna() & ~series.isna()
    if invalid.any():
        raise ValueError(f"{name} contains nonnumeric values")
    values = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    return values


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    reference_values = _finite_numeric_values(reference, name="reference")
    current_values = _finite_numeric_values(current, name="current")
    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference_values, bins=edges)
    cur_counts, _ = np.histogram(current_values, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def build_drift_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    if train.empty or test.empty:
        raise ValueError("train and test frames must be nonempty")
    if not features or len(features) != len(set(features)):
        raise ValueError("features must be a nonempty list without duplicates")
    missing = set(features).difference(train.columns).union(set(features).difference(test.columns))
    if missing:
        raise ValueError(f"Missing drift features: {sorted(missing)}")

    rows = []
    for feature in features:
        train_values = _finite_numeric_values(train[feature], name=f"train.{feature}")
        test_values = _finite_numeric_values(test[feature], name=f"test.{feature}")
        rows.append(
            {
                "feature": feature,
                "psi": population_stability_index(train[feature], test[feature]),
                "train_mean": float(np.mean(train_values)) if len(train_values) else float("nan"),
                "test_mean": float(np.mean(test_values)) if len(test_values) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
