import pandas as pd

from reference_eta.models.hsg_eta import CONTEXT_FEATURES, TASK_FEATURES, SnapshotTaskDataset


def _snapshots() -> pd.DataFrame:
    row = {feature: 1.0 for feature in CONTEXT_FEATURES}
    row.update({"snapshot_id": "s1", "target_route_remaining_minutes": 20.0})
    return pd.DataFrame([row])


def _task(task_id: str, distance: float, actual_rank: int, target_next: int) -> dict[str, object]:
    row: dict[str, object] = {feature: 1.0 for feature in TASK_FEATURES}
    row.update(
        {
            "snapshot_id": "s1",
            "task_id": task_id,
            "distance_to_current": distance,
            "actual_rank": actual_rank,
            "target_next": target_next,
        }
    )
    return row


def test_hsg_order_uses_observable_distance_not_actual_rank() -> None:
    tasks = pd.DataFrame(
        [
            _task("future-first", 8.0, actual_rank=0, target_next=1),
            _task("nearby", 1.0, actual_rank=2, target_next=0),
            _task("middle", 2.0, actual_rank=1, target_next=0),
        ]
    )
    dataset = SnapshotTaskDataset(_snapshots(), tasks, max_tasks=3)
    item = dataset[0]
    # Sorted by observable distance, so the future-next task is at index 2, not index 0.
    assert item["route_target"].item() == 2


def test_hsg_truncation_does_not_force_future_target_into_input() -> None:
    tasks = pd.DataFrame(
        [
            _task("future-first", 8.0, actual_rank=0, target_next=1),
            _task("nearby", 1.0, actual_rank=2, target_next=0),
            _task("middle", 2.0, actual_rank=1, target_next=0),
        ]
    )
    dataset = SnapshotTaskDataset(_snapshots(), tasks, max_tasks=2)
    assert dataset[0]["route_target"].item() == -100
