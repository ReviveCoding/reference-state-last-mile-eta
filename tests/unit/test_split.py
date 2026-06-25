from reference_eta.data.split import grouped_temporal_split
from reference_eta.data.synthetic import generate_synthetic_delivery_data


def test_grouped_temporal_split_has_no_courier_day_overlap() -> None:
    data = generate_synthetic_delivery_data(
        n_courier_days=40,
        start_date="2026-01-01",
        cities=["Atlanta", "Boston"],
        min_tasks=8,
        max_tasks=10,
        snapshot_stride=3,
        seed=2,
    )
    split = grouped_temporal_split(
        data.snapshots,
        train_fraction=0.6,
        validation_fraction=0.15,
        calibration_fraction=0.1,
        test_fraction=0.15,
    )
    parts = [split.train, split.validation, split.calibration, split.test]
    groups = [set(zip(part["courier_id"], part["work_date"], strict=True)) for part in parts]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            assert groups[i].isdisjoint(groups[j])
