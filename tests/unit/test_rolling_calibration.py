import numpy as np
import pandas as pd

from reference_eta.models.rolling_calibration import RollingConformalReplay


def test_rolling_replay_updates_without_crossing() -> None:
    predictions = pd.DataFrame({"q10": [8, 9, 10], "q50": [10, 11, 12], "q90": [12, 13, 14]})
    result = RollingConformalReplay(target_coverage=0.8, window_size=5).replay(
        np.array([0.0, 1.0, 2.0]), predictions, np.array([11.0, 20.0, 13.0])
    )
    assert len(result) == 3
    assert (result["lower"] <= result["median"]).all()
    assert (result["median"] <= result["upper"]).all()
    assert result["rolling_coverage_25"].between(0, 1).all()


def test_rolling_replay_rejects_mismatched_inputs() -> None:
    import pytest

    predictions = pd.DataFrame({"q10": [1.0], "q50": [2.0], "q90": [3.0]})
    with pytest.raises(ValueError):
        RollingConformalReplay(target_coverage=0.8, window_size=5).replay(
            np.array([0.0]), predictions, np.array([2.0, 3.0])
        )
    with pytest.raises(ValueError):
        RollingConformalReplay(target_coverage=0.0, window_size=5)


def test_delayed_labels_are_not_released_before_availability() -> None:
    predictions = pd.DataFrame(
        {"q10": [8.0, 8.0, 8.0], "q50": [10.0, 10.0, 10.0], "q90": [12.0, 12.0, 12.0]}
    )
    prediction_times = pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:10", "2026-01-01 10:10"])
    label_times = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 11:00", "2026-01-01 12:00"])
    result = RollingConformalReplay(target_coverage=0.8, window_size=10).replay(
        np.array([0.0]),
        predictions,
        np.array([20.0, 11.0, 10.0]),
        prediction_times=prediction_times,
        label_available_times=label_times,
    )
    assert result.loc[0, "available_score_count"] == 1
    assert result.loc[1, "available_score_count"] == 1
    # At 10:10 the first 10:00 label becomes available, but the 11:00 label does not.
    assert result.loc[2, "available_score_count"] == 2
    assert result.loc[2, "released_scores_this_step"] == 1
