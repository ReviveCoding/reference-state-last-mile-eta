import pandas as pd

from reference_eta.data.lade import build_closed_set_snapshots


def _frame(future_shift: int) -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01 08:00:00")
    delivery = [base + pd.Timedelta(minutes=20 + i * 10) for i in range(8)]
    for index in range(1, len(delivery)):
        delivery[index] += pd.Timedelta(minutes=future_shift)
    return pd.DataFrame(
        {
            "courier_id": ["c1"] * 8,
            "city": ["Boston"] * 8,
            "accept_time": [base + pd.Timedelta(minutes=i) for i in range(8)],
            "delivery_time": delivery,
            "latitude": [42.0 + i * 0.001 for i in range(8)],
            "longitude": [-71.0 - i * 0.001 for i in range(8)],
            "task_id": [f"t{i}" for i in range(8)],
            "time_window_end": [base + pd.Timedelta(hours=4)] * 8,
            "work_date": ["2026-01-01"] * 8,
        }
    )


def test_future_delivery_time_does_not_leak_into_pending_task_features() -> None:
    original = build_closed_set_snapshots(_frame(0), snapshot_stride=2)
    shifted = build_closed_set_snapshots(_frame(30), snapshot_stride=2)
    original_first = original.pending_tasks[
        original.pending_tasks["snapshot_id"] == "LADE-00000000"
    ]
    shifted_first = shifted.pending_tasks[shifted.pending_tasks["snapshot_id"] == "LADE-00000000"]
    feature_columns = [
        "task_id",
        "distance_to_current",
        "delta_x",
        "delta_y",
        "service_burden",
        "package_count",
        "same_aoi",
        "time_window_slack",
        "task_density",
    ]
    pd.testing.assert_frame_equal(
        original_first[feature_columns].reset_index(drop=True),
        shifted_first[feature_columns].reset_index(drop=True),
    )
    assert (
        original.snapshots.iloc[0]["target_route_remaining_minutes"]
        != shifted.snapshots.iloc[0]["target_route_remaining_minutes"]
    )
