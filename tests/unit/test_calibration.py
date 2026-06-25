import numpy as np
import pandas as pd

from reference_eta.models.calibration import ConformalQuantileCalibrator


def test_conformal_calibration_preserves_quantile_order() -> None:
    predictions = pd.DataFrame({"q10": [8, 18, 28], "q50": [10, 20, 30], "q90": [12, 22, 32]})
    y = np.array([15.0, 20.0, 25.0])
    calibrated = (
        ConformalQuantileCalibrator(target_coverage=0.8).fit(predictions, y).transform(predictions)
    )
    assert np.allclose(calibrated["q50"], predictions["q50"])
    assert (calibrated["q10"] <= calibrated["q50"]).all()
    assert (calibrated["q50"] <= calibrated["q90"]).all()
    assert (calibrated["q10"] >= 0).all()


def test_conformal_rejects_invalid_inputs() -> None:
    import pytest

    with pytest.raises(ValueError):
        ConformalQuantileCalibrator(target_coverage=1.0)
    calibrator = ConformalQuantileCalibrator(target_coverage=0.8)
    predictions = pd.DataFrame({"q10": [1.0], "q50": [2.0], "q90": [3.0]})
    with pytest.raises(ValueError):
        calibrator.fit(predictions, np.array([2.0, 3.0]))
    with pytest.raises(ValueError):
        calibrator.fit(predictions.assign(q90=np.nan), np.array([2.0]))
