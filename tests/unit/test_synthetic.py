from reference_eta.data.synthetic import generate_synthetic_delivery_data


def test_synthetic_generation_is_deterministic() -> None:
    kwargs = dict(
        n_courier_days=20,
        start_date="2026-01-01",
        cities=["Atlanta", "Boston"],
        min_tasks=8,
        max_tasks=12,
        snapshot_stride=3,
        seed=7,
    )
    first = generate_synthetic_delivery_data(**kwargs)
    second = generate_synthetic_delivery_data(**kwargs)
    assert first.snapshots.equals(second.snapshots)
    assert first.pending_tasks.equals(second.pending_tasks)
    assert (first.snapshots["target_route_remaining_minutes"] > 0).all()
    assert (first.pending_tasks.groupby("snapshot_id")["target_next"].sum() == 1).all()


def test_synthetic_generation_rejects_unrunnable_shape() -> None:
    import pytest

    with pytest.raises(ValueError, match="n_courier_days"):
        generate_synthetic_delivery_data(
            n_courier_days=4,
            start_date="2026-01-01",
            cities=["Boston"],
            min_tasks=8,
            max_tasks=12,
            snapshot_stride=3,
            seed=7,
        )
