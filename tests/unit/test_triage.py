import numpy as np
import pandas as pd

from reference_eta.decisions.triage import capacity_metrics, derive_tail_scores


def test_capacity_metrics_respect_capacity() -> None:
    predictions = pd.DataFrame({"q10": [5, 10, 20], "q50": [10, 20, 40], "q90": [20, 40, 80]})
    thresholds = np.array([30.0, 30.0, 30.0])
    scores = derive_tail_scores(predictions, thresholds, np.ones(3))
    rows = capacity_metrics(
        np.array([10.0, 35.0, 70.0]), thresholds, scores["tail_risk"].to_numpy(), [1 / 3]
    )
    assert rows[0]["selected"] == 1
    assert rows[0]["precision_at_capacity"] == 1.0
