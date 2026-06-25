import numpy as np
import pandas as pd

from reference_eta.decisions.triage import derive_tail_scores


def test_tail_scores_match_logistic_quantiles() -> None:
    predictions = pd.DataFrame({"q10": [10.0], "q50": [20.0], "q90": [30.0]})
    result = derive_tail_scores(
        predictions,
        thresholds=np.array([20.0]),
        trust=np.array([1.0]),
    )
    assert result.iloc[0]["tail_probability"] == 0.5
    assert result.iloc[0]["expected_excess_minutes"] > 0.0
    assert result.iloc[0]["tail_risk"] > 0.0


def test_tail_risk_is_unconditional_expected_excess_not_double_probability() -> None:
    predictions = pd.DataFrame({"q10": [10.0], "q50": [20.0], "q90": [30.0]})
    result = derive_tail_scores(
        predictions,
        thresholds=np.array([20.0]),
        trust=np.array([0.5]),
    ).iloc[0]
    # At the median threshold, P(tail)=0.5 and conditional severity is twice
    # the unconditional expected excess.
    assert np.isclose(result["tail_probability"], 0.5)
    assert np.isclose(
        result["tail_risk"], result["tail_probability"] * result["expected_excess_minutes"]
    )
    assert np.isclose(result["reliability_adjusted_priority"], 0.5 * result["tail_risk"])


def test_tail_scores_reject_crossed_quantiles_and_invalid_trust() -> None:
    import pytest

    crossed = pd.DataFrame({"q10": [10.0], "q50": [9.0], "q90": [30.0]})
    with pytest.raises(ValueError, match="q10"):
        derive_tail_scores(crossed, np.array([20.0]), np.array([1.0]))
    valid = pd.DataFrame({"q10": [10.0], "q50": [20.0], "q90": [30.0]})
    with pytest.raises(ValueError, match="trust"):
        derive_tail_scores(valid, np.array([20.0]), np.array([1.1]))
