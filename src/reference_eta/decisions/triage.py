from __future__ import annotations

import numpy as np
import pandas as pd

_LOGISTIC_90_LOG_ODDS = float(np.log(9.0))


def _finite_vector(
    values: np.ndarray, *, name: str, expected_length: int | None = None
) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if expected_length is not None and len(vector) != expected_length:
        raise ValueError(f"{name} length does not match the expected length")
    if len(vector) == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return vector


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_tail_thresholds(train: pd.DataFrame, target: str) -> dict[tuple[str, int], float]:
    required = {"city", "route_phase", target}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"Missing threshold columns: {sorted(missing)}")
    if train.empty:
        raise ValueError("Threshold training data cannot be empty")
    target_values = pd.to_numeric(train[target], errors="coerce")
    phases = pd.to_numeric(train["route_phase"], errors="coerce")
    if not np.isfinite(target_values.to_numpy(dtype=float)).all() or (target_values < 0.0).any():
        raise ValueError("Tail target must be finite and nonnegative")
    if not np.isfinite(phases.to_numpy(dtype=float)).all() or not phases.between(0.0, 1.0).all():
        raise ValueError("route_phase must be finite and within [0, 1]")
    if train["city"].isna().any() or train["city"].astype(str).str.strip().eq("").any():
        raise ValueError("city cannot be missing or blank")

    enriched = train.copy()
    enriched[target] = target_values
    enriched["_phase_bin"] = np.digitize(phases, [0.33, 0.67])
    thresholds: dict[tuple[str, int], float] = {}
    for (city, phase), group in enriched.groupby(["city", "_phase_bin"]):
        thresholds[(str(city), int(phase))] = float(group[target].quantile(0.90))
    thresholds[("__global__", -1)] = float(enriched[target].quantile(0.90))
    return thresholds


def threshold_for_rows(frame: pd.DataFrame, thresholds: dict[tuple[str, int], float]) -> np.ndarray:
    required = {"city", "route_phase"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing threshold lookup columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Threshold lookup frame cannot be empty")
    global_key = ("__global__", -1)
    if global_key not in thresholds or not np.isfinite(float(thresholds[global_key])):
        raise ValueError("Threshold mapping requires a finite global fallback")
    phases = pd.to_numeric(frame["route_phase"], errors="coerce")
    if not np.isfinite(phases.to_numpy(dtype=float)).all() or not phases.between(0.0, 1.0).all():
        raise ValueError("route_phase must be finite and within [0, 1]")

    values: list[float] = []
    global_threshold = float(thresholds[global_key])
    for position, (_, row) in enumerate(frame.iterrows()):
        phase = int(np.digitize([phases.iloc[position]], [0.33, 0.67])[0])
        threshold = float(thresholds.get((str(row["city"]), phase), global_threshold))
        if not np.isfinite(threshold):
            raise ValueError("Threshold mapping contains a nonfinite value")
        values.append(threshold)
    return np.asarray(values, dtype=float)


def derive_tail_scores(
    predictions: pd.DataFrame,
    thresholds: np.ndarray,
    trust: np.ndarray,
) -> pd.DataFrame:
    """Derive coherent tail probability and expected excess from q10/q50/q90.

    A logistic predictive distribution is moment-matched to the reported 10th, 50th,
    and 90th percentiles. For a logistic distribution, q90-q10 = 2*log(9)*scale.
    """

    required = {"q10", "q50", "q90"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing quantile columns: {sorted(missing)}")
    values = predictions[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Predictions must be nonempty and finite")
    q10, q50, q90 = values.T
    if ((q10 > q50) | (q50 > q90)).any():
        raise ValueError("Quantile predictions must satisfy q10 <= q50 <= q90")

    thresholds = _finite_vector(thresholds, name="thresholds", expected_length=len(values))
    trust = _finite_vector(trust, name="trust", expected_length=len(values))
    if ((trust < 0.0) | (trust > 1.0)).any():
        raise ValueError("trust must be within [0, 1]")
    scale = np.maximum((q90 - q10) / (2.0 * _LOGISTIC_90_LOG_ODDS), 1e-3)
    standardized = (q50 - thresholds) / scale
    probability = _stable_sigmoid(standardized)

    # For X ~ Logistic(mu, scale), E[(X-t)+] = scale*log(1+exp((mu-t)/scale)).
    unconditional_expected_excess = scale * np.logaddexp(0.0, standardized)
    conditional_excess = unconditional_expected_excess / np.maximum(probability, 1e-12)
    risk = unconditional_expected_excess

    # A constant per-review cost cannot change a fixed-capacity ranking. It is applied in
    # the intervention sensitivity simulator rather than subtracted from every row here.
    adjusted = risk * trust
    return pd.DataFrame(
        {
            "tail_probability": probability,
            "expected_excess_minutes": conditional_excess,
            "tail_risk": risk,
            "reliability_adjusted_priority": adjusted,
        },
        index=predictions.index,
    )


def capacity_metrics(
    y_true: np.ndarray,
    thresholds: np.ndarray,
    scores: np.ndarray,
    capacities: list[float],
) -> list[dict[str, float]]:
    y_true = _finite_vector(y_true, name="y_true")
    thresholds = _finite_vector(thresholds, name="thresholds", expected_length=len(y_true))
    scores = _finite_vector(scores, name="scores", expected_length=len(y_true))
    if not capacities or len(capacities) != len(set(map(float, capacities))):
        raise ValueError("capacities must be nonempty and unique")
    if any(not 0.0 < float(capacity) <= 1.0 for capacity in capacities):
        raise ValueError("Every capacity must be in (0, 1]")

    true_tail = y_true > thresholds
    excess = np.maximum(y_true - thresholds, 0.0)
    total_tail = int(true_tail.sum())
    total_excess = float(excess.sum())
    order = np.argsort(-scores, kind="stable")
    rows: list[dict[str, float]] = []
    for capacity in capacities:
        k = max(1, int(np.ceil(len(y_true) * float(capacity))))
        selected = order[:k]
        captured_tail = int(true_tail[selected].sum())
        captured_excess = float(excess[selected].sum())
        rows.append(
            {
                "capacity": float(capacity),
                "selected": float(k),
                "precision_at_capacity": captured_tail / k,
                "tail_recall_at_capacity": captured_tail / max(total_tail, 1),
                "excess_minutes_capture": captured_excess / max(total_excess, 1e-9),
                "false_review_burden": float((~true_tail[selected]).sum()),
            }
        )
    return rows
