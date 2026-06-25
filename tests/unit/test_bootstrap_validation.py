import numpy as np
import pandas as pd
import pytest

from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci


def test_bootstrap_rejects_length_mismatch() -> None:
    frame = pd.DataFrame({"courier_id": ["c1", "c2"], "work_date": ["d1", "d2"]})
    with pytest.raises(ValueError, match="length"):
        clustered_mae_difference_ci(
            frame,
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            n_bootstrap=20,
        )
