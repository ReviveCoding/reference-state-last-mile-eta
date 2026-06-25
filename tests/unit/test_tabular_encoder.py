import numpy as np
import pandas as pd
import pytest

from reference_eta.features.tabular import BASE_NUMERIC_FEATURES, TabularFeatureEncoder


def _frame(city: str = "Boston") -> pd.DataFrame:
    row = {feature: 1.0 for feature in BASE_NUMERIC_FEATURES}
    row["city"] = city
    return pd.DataFrame([row])


def test_tabular_encoder_marks_unseen_city_explicitly() -> None:
    encoder = TabularFeatureEncoder(include_rcot=False).fit(_frame("Boston"))
    transformed = encoder.transform(_frame("Seattle"))
    assert transformed.iloc[0]["city___UNKNOWN__"] == 1.0
    assert transformed.iloc[0]["city_Boston"] == 0.0


def test_tabular_encoder_rejects_nonfinite_features() -> None:
    frame = _frame()
    frame.loc[0, BASE_NUMERIC_FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        TabularFeatureEncoder(include_rcot=False).fit(frame)
