import numpy as np

from reference_eta.data.split import grouped_temporal_split
from reference_eta.data.synthetic import generate_synthetic_delivery_data
from reference_eta.features.rcot import ReferenceOperationalTimeTransformer


def test_rcot_outputs_are_bounded_and_supported() -> None:
    data = generate_synthetic_delivery_data(
        n_courier_days=50,
        start_date="2026-01-01",
        cities=["Atlanta", "Boston"],
        min_tasks=10,
        max_tasks=14,
        snapshot_stride=3,
        seed=11,
    )
    split = grouped_temporal_split(
        data.snapshots,
        train_fraction=0.6,
        validation_fraction=0.15,
        calibration_fraction=0.1,
        test_fraction=0.15,
    )
    transformer = ReferenceOperationalTimeTransformer(min_cohort_rows=8).fit(split.train)
    transformed = transformer.transform(split.test)
    assert np.isfinite(transformed["rcot_minutes"]).all()
    assert transformed["reference_support"].between(0, 1).all()
    assert transformed["rcot_trust"].between(0, 1).all()
    assert (transformed["reference_dispersion"] > 0).all()
