from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reference_eta.decisions.sensitivity import intervention_sensitivity
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import interval_metrics, point_metrics, slice_metrics
from reference_eta.monitoring.drift import build_drift_report, population_stability_index


def test_intervention_sensitivity_rewards_better_ranking_and_reports_regret() -> None:
    frame = intervention_sensitivity(
        y_true=np.array([30.0, 5.0, 20.0, 1.0]),
        thresholds=np.array([10.0, 10.0, 10.0, 10.0]),
        policy_scores={
            "good": np.array([4.0, 1.0, 3.0, 2.0]),
            "bad": np.array([1.0, 4.0, 2.0, 3.0]),
        },
        capacity=0.25,
        effectiveness_grid=[0.0, 0.5, 1.0],
        action_cost=2.0,
    )
    assert set(frame["policy"]) == {"good", "bad"}
    at_full_effect = frame[frame["effectiveness"] == 1.0].set_index("policy")
    assert at_full_effect.loc["good", "captured_excess_minutes"] == 20.0
    assert at_full_effect.loc["good", "simulated_utility"] == 18.0
    assert at_full_effect.loc["good", "regret"] == 0.0
    assert at_full_effect.loc["bad", "regret"] > 0.0


def test_intervention_sensitivity_rejects_nonfinite_policy_scores() -> None:
    with pytest.raises(ValueError, match="invalid scores"):
        intervention_sensitivity(
            y_true=np.array([1.0, 2.0]),
            thresholds=np.array([1.0, 1.0]),
            policy_scores={"bad": np.array([np.nan, 1.0])},
            capacity=0.5,
            effectiveness_grid=[0.5],
            action_cost=0.0,
        )


def test_clustered_bootstrap_is_seed_reproducible_and_detects_better_challenger() -> None:
    frame = pd.DataFrame(
        {
            "courier_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "work_date": ["d1", "d1", "d2", "d2", "d3", "d3"],
        }
    )
    y_true = np.array([1.0, 2.0, 10.0, 11.0, 20.0, 21.0])
    baseline = y_true + 4.0
    challenger = y_true + 1.0
    first = clustered_mae_difference_ci(
        frame, y_true, baseline, challenger, n_bootstrap=50, seed=17
    )
    second = clustered_mae_difference_ci(
        frame, y_true, baseline, challenger, n_bootstrap=50, seed=17
    )
    assert first == second
    assert first["challenger_minus_baseline_mae"] < 0.0
    assert first["ci_upper"] < 0.0


def test_point_interval_and_slice_metrics_have_known_values() -> None:
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    point = point_metrics(y_true, y_pred)
    assert point["mae"] == pytest.approx(2.0 / 3.0)
    assert point["signed_bias"] == pytest.approx(0.0)

    quantiles = pd.DataFrame(
        {
            "q10": [0.0, 2.0, 4.0],
            "q50": [1.0, 3.0, 5.0],
            "q90": [2.0, 4.0, 6.0],
        }
    )
    interval = interval_metrics(y_true, quantiles)
    assert interval["coverage"] == 1.0
    assert interval["mean_interval_width"] == 2.0
    assert interval["quantile_crossing_rate"] == 0.0

    frame = pd.DataFrame({"city": ["A", "A", "B"]})
    records = slice_metrics(frame, y_true, y_pred, slice_column="city")
    assert {record["value"] for record in records} == {"A", "B"}
    assert sum(record["count"] for record in records) == 3


def test_drift_report_orders_shifted_feature_first() -> None:
    train = pd.DataFrame(
        {
            "stable": np.linspace(0.0, 1.0, 200),
            "shifted": np.linspace(0.0, 1.0, 200),
        }
    )
    test = pd.DataFrame(
        {
            "stable": np.linspace(0.0, 1.0, 200),
            "shifted": np.linspace(3.0, 4.0, 200),
        }
    )
    report = build_drift_report(train, test, ["stable", "shifted"])
    assert report.iloc[0]["feature"] == "shifted"
    assert report.iloc[0]["psi"] > report.iloc[1]["psi"]
    assert population_stability_index(train["stable"], test["stable"]) == pytest.approx(0.0)
