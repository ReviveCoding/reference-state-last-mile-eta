from reference_eta.data.split import grouped_temporal_split
from reference_eta.data.synthetic import generate_synthetic_delivery_data
from reference_eta.features.rcot import ReferenceOperationalTimeTransformer


def test_reference_curves_do_not_change_when_test_outcomes_change() -> None:
    data = generate_synthetic_delivery_data(
        n_courier_days=45,
        start_date="2026-01-01",
        cities=["Atlanta", "Boston"],
        min_tasks=8,
        max_tasks=12,
        snapshot_stride=3,
        seed=5,
    )
    split = grouped_temporal_split(
        data.snapshots,
        train_fraction=0.6,
        validation_fraction=0.15,
        calibration_fraction=0.1,
        test_fraction=0.15,
    )
    transformer = ReferenceOperationalTimeTransformer(min_cohort_rows=8).fit(split.train)
    before = transformer.transform(split.test)["rcot_minutes"].to_numpy()
    tampered = split.test.copy()
    tampered["target_route_remaining_minutes"] = 999999.0
    after = transformer.transform(tampered)["rcot_minutes"].to_numpy()
    assert (before == after).all()
