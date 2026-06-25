from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_vector(values: np.ndarray, *, name: str, expected_length: int) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if len(vector) != expected_length:
        raise ValueError(f"{name} length does not match frame length")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def clustered_mae_difference_ci(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    challenger_pred: np.ndarray,
    *,
    cluster_columns: tuple[str, str] = ("courier_id", "work_date"),
    n_bootstrap: int = 250,
    seed: int = 42,
) -> dict[str, float]:
    if frame.empty:
        raise ValueError("frame cannot be empty")
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not cluster_columns or len(cluster_columns) != len(set(cluster_columns)):
        raise ValueError("cluster_columns must be nonempty and unique")
    missing = set(cluster_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing bootstrap cluster columns: {sorted(missing)}")
    if frame[list(cluster_columns)].isna().any().any():
        raise ValueError("Bootstrap cluster columns cannot contain missing values")

    y = _finite_vector(y_true, name="y_true", expected_length=len(frame))
    baseline = _finite_vector(baseline_pred, name="baseline_pred", expected_length=len(frame))
    challenger = _finite_vector(challenger_pred, name="challenger_pred", expected_length=len(frame))

    working = frame[list(cluster_columns)].copy()
    working["y_true"] = y
    working["baseline"] = baseline
    working["challenger"] = challenger
    clusters = list(working.groupby(list(cluster_columns), sort=False, dropna=False))
    if not clusters:
        raise ValueError("No bootstrap clusters were available")

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        sample = pd.concat([clusters[index][1] for index in chosen], ignore_index=True)
        baseline_mae = np.mean(np.abs(sample["y_true"] - sample["baseline"]))
        challenger_mae = np.mean(np.abs(sample["y_true"] - sample["challenger"]))
        differences.append(float(challenger_mae - baseline_mae))
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "challenger_minus_baseline_mae": float(np.mean(differences)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }
