from __future__ import annotations

import numpy as np
import pandas as pd


def intervention_sensitivity(
    *,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    policy_scores: dict[str, np.ndarray],
    capacity: float,
    effectiveness_grid: list[float],
    action_cost: float,
) -> pd.DataFrame:
    """Evaluate assumption-transparent utility and regret over action effectiveness.

    Utility is simulated, not causal evidence. The effect parameter is explicitly varied and
    actual predictive triage metrics are reported separately.
    """

    target = np.asarray(y_true, dtype=float).reshape(-1)
    threshold_values = np.asarray(thresholds, dtype=float).reshape(-1)
    if len(target) == 0 or len(target) != len(threshold_values):
        raise ValueError("y_true and thresholds must be nonempty and have the same length")
    if not np.isfinite(target).all() or not np.isfinite(threshold_values).all():
        raise ValueError("y_true and thresholds must be finite")
    if not 0.0 < float(capacity) <= 1.0:
        raise ValueError("capacity must be in (0, 1]")
    if float(action_cost) < 0.0:
        raise ValueError("action_cost must be nonnegative")
    if not effectiveness_grid:
        raise ValueError("effectiveness_grid must not be empty")
    if any(not 0.0 <= float(value) <= 1.0 for value in effectiveness_grid):
        raise ValueError("effectiveness values must be between 0 and 1")
    if not policy_scores:
        raise ValueError("policy_scores must not be empty")

    excess = np.maximum(target - threshold_values, 0.0)
    k = max(1, int(np.ceil(len(target) * capacity)))
    rows: list[dict[str, float | str]] = []
    validated_scores: dict[str, np.ndarray] = {}
    for policy, scores in policy_scores.items():
        values = np.asarray(scores, dtype=float).reshape(-1)
        if len(values) != len(target) or not np.isfinite(values).all():
            raise ValueError(f"Policy {policy!r} has invalid scores")
        validated_scores[policy] = values

    for effectiveness in effectiveness_grid:
        scenario_rows: list[dict[str, float | str]] = []
        for policy, scores in validated_scores.items():
            selected = np.argsort(-scores, kind="stable")[:k]
            captured_excess = float(excess[selected].sum())
            utility = float(effectiveness) * captured_excess - float(action_cost) * k
            break_even = float(action_cost) * k / max(captured_excess, 1e-9)
            scenario_rows.append(
                {
                    "policy": policy,
                    "effectiveness": float(effectiveness),
                    "capacity": float(capacity),
                    "selected": float(k),
                    "captured_excess_minutes": captured_excess,
                    "simulated_utility": utility,
                    "break_even_effectiveness": break_even,
                }
            )
        best = max(float(row["simulated_utility"]) for row in scenario_rows)
        for row in scenario_rows:
            row["regret"] = best - float(row["simulated_utility"])
            rows.append(row)
    return pd.DataFrame(rows)
