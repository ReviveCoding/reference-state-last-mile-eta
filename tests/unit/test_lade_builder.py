import pandas as pd

from reference_eta.data.lade import build_closed_set_snapshots


def test_lade_closed_set_builder_uses_visible_tasks() -> None:
    base = pd.Timestamp("2026-01-01 08:00:00")
    frame = pd.DataFrame(
        {
            "courier_id": ["c1"] * 8,
            "city": ["Boston"] * 8,
            "accept_time": [base + pd.Timedelta(minutes=i * 2) for i in range(8)],
            "delivery_time": [base + pd.Timedelta(minutes=20 + i * 10) for i in range(8)],
            "latitude": [42.0 + i * 0.001 for i in range(8)],
            "longitude": [-71.0 - i * 0.001 for i in range(8)],
            "task_id": [f"t{i}" for i in range(8)],
            "work_date": ["2026-01-01"] * 8,
        }
    )
    data = build_closed_set_snapshots(frame, snapshot_stride=2)
    assert not data.snapshots.empty
    assert (data.snapshots["target_route_remaining_minutes"] > 0).all()
    assert (data.pending_tasks.groupby("snapshot_id")["target_next"].sum() == 1).all()


def test_lade_builder_rejects_invalid_direct_input() -> None:
    base = pd.Timestamp("2026-01-01 08:00:00")
    frame = pd.DataFrame(
        {
            "courier_id": ["c1"] * 4,
            "city": ["Boston"] * 4,
            "accept_time": [base] * 4,
            "delivery_time": [base + pd.Timedelta(minutes=10 + i) for i in range(4)],
            "latitude": [95.0, 42.0, 42.0, 42.0],
            "longitude": [-71.0] * 4,
        }
    )
    import pytest

    with pytest.raises(ValueError, match="latitude"):
        build_closed_set_snapshots(frame)
