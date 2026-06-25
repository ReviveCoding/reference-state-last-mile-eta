import numpy as np

from reference_eta.data.synthetic import generate_synthetic_delivery_data
from reference_eta.features.rcot import ReferenceOperationalTimeTransformer


def test_cross_fitted_rcot_is_complete_and_refits_final_transformer() -> None:
    data = generate_synthetic_delivery_data(
        n_courier_days=24,
        start_date="2026-01-01",
        cities=["Atlanta", "Boston"],
        min_tasks=8,
        max_tasks=10,
        snapshot_stride=3,
        seed=9,
    )
    transformer = ReferenceOperationalTimeTransformer(
        min_cohort_rows=4,
        min_cohort_groups=2,
    )
    transformed = transformer.fit_transform_cross_fitted(data.snapshots, n_splits=4)
    assert len(transformed) == len(data.snapshots)
    assert np.isfinite(transformed["rcot_minutes"]).all()
    assert transformer.fitted_
    future = transformer.transform(data.snapshots.iloc[:2])
    assert len(future) == 2
