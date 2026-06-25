import pandas as pd
import pytest

from reference_eta.monitoring.drift import build_drift_report, population_stability_index


def test_drift_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="bins"):
        population_stability_index(pd.Series([1.0]), pd.Series([1.0]), bins=1)
    with pytest.raises(ValueError, match="Missing drift features"):
        build_drift_report(pd.DataFrame({"x": [1.0]}), pd.DataFrame({"x": [2.0]}), ["y"])
