import numpy as np
import pandas as pd

from reference_eta.evaluation.release import evaluate_rcot_promotion, evaluate_system_release


def test_promotion_requires_gain_and_slice_safety() -> None:
    frame = pd.DataFrame({"city": ["A", "A", "B", "B"]})
    y = np.array([10.0, 20.0, 30.0, 40.0])
    baseline = np.array([12.0, 22.0, 32.0, 42.0])
    challenger = np.array([11.0, 21.0, 31.0, 41.0])
    thresholds = np.array([15.0, 15.0, 35.0, 35.0])
    decision = evaluate_rcot_promotion(frame, y, baseline, challenger, thresholds)
    assert decision.promote


def test_system_gate_rejects_bad_coverage() -> None:
    predictions = pd.DataFrame({"q10": [1.0] * 100, "q50": [2.0] * 100, "q90": [3.0] * 100})
    decision = evaluate_system_release(
        interval_coverage=0.4,
        target_coverage=0.8,
        quantile_crossing_rate=0.0,
        predictions=predictions,
        test_rows=100,
        champion_mae=2.0,
        business_baseline_mae=3.0,
        decision_excess_capture=0.5,
        random_excess_capture=0.1,
        test_tail_events=10,
        coverage_tolerance=0.1,
    )
    assert not decision["pass"]
    assert not decision["checks"]["coverage_within_tolerance"]


def test_system_gate_rejects_bad_predictive_or_decision_quality() -> None:
    predictions = pd.DataFrame({"q10": [1.0] * 100, "q50": [2.0] * 100, "q90": [3.0] * 100})
    decision = evaluate_system_release(
        interval_coverage=0.8,
        target_coverage=0.8,
        quantile_crossing_rate=0.0,
        predictions=predictions,
        test_rows=100,
        champion_mae=3.2,
        business_baseline_mae=3.0,
        decision_excess_capture=0.1,
        random_excess_capture=0.2,
        test_tail_events=0,
    )
    assert not decision["pass"]
    assert not decision["checks"]["champion_predictive_quality"]
    assert not decision["checks"]["decision_noninferior_to_random"]
    assert not decision["checks"]["minimum_test_tail_events"]
