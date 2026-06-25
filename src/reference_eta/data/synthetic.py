from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticData:
    snapshots: pd.DataFrame
    pending_tasks: pd.DataFrame


_CITY_SPEED = {
    "Atlanta": 25.0,
    "Boston": 18.0,
    "Chicago": 21.0,
}


def _nearest_neighbor_order(coords: np.ndarray, rng: np.random.Generator) -> list[int]:
    remaining = set(range(len(coords)))
    current = np.array([0.0, 0.0])
    order: list[int] = []
    while remaining:
        candidates = np.array(sorted(remaining), dtype=int)
        distances = np.linalg.norm(coords[candidates] - current, axis=1)
        noise = rng.normal(0.0, 0.12, size=len(candidates))
        score = distances * (1.0 + noise)
        chosen = int(candidates[int(np.argmin(score))])
        order.append(chosen)
        remaining.remove(chosen)
        current = coords[chosen]
    return order


def _route_density(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    distances[distances == 0.0] = np.nan
    nearest = np.nanmin(distances, axis=1)
    return float(1.0 / (np.nanmean(nearest) + 0.2))


def generate_synthetic_delivery_data(
    *,
    n_courier_days: int,
    start_date: str,
    cities: Iterable[str],
    min_tasks: int,
    max_tasks: int,
    snapshot_stride: int,
    seed: int,
) -> SyntheticData:
    """Generate event-time delivery snapshots with no future-derived input features.

    Targets use realized future durations, while all model features are observable at the
    snapshot query time or are static task attributes.
    """

    if n_courier_days < 8:
        raise ValueError("n_courier_days must be at least 8 for temporal splitting")
    cities = [str(city).strip() for city in cities]
    if not cities or any(not city for city in cities):
        raise ValueError("cities must contain at least one nonblank city")
    if min_tasks < 4 or max_tasks < min_tasks:
        raise ValueError("Require 4 <= min_tasks <= max_tasks")
    if snapshot_stride < 1:
        raise ValueError("snapshot_stride must be at least 1")
    try:
        start = pd.Timestamp(start_date)
    except (TypeError, ValueError) as error:
        raise ValueError("start_date must be parseable as a timestamp") from error
    if pd.isna(start):
        raise ValueError("start_date must be a valid timestamp")

    rng = np.random.default_rng(seed)
    snapshot_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    for day_index in range(n_courier_days):
        date = start + pd.Timedelta(days=day_index)
        city = str(rng.choice(cities))
        courier_id = f"{city[:3].upper()}-{int(rng.integers(0, 32)):03d}"
        n_tasks = int(rng.integers(min_tasks, max_tasks + 1))
        start_hour = float(np.clip(rng.normal(8.3, 0.7), 6.5, 10.5))
        route_start = date + pd.Timedelta(hours=start_hour)
        weather_severity = float(np.clip(rng.beta(2.0, 7.0) + 0.15 * (date.month in (1, 2)), 0, 1))
        peak_factor = 1.0 + 0.20 * (7.5 <= start_hour <= 9.5)
        base_speed = _CITY_SPEED.get(city, 21.0)
        congestion_index = float(np.clip(peak_factor + rng.normal(0, 0.08), 0.8, 1.5))

        n_centers = int(rng.integers(2, 5))
        centers = rng.normal(0, 3.0, size=(n_centers, 2))
        cluster_ids = rng.integers(0, n_centers, size=n_tasks)
        coords = centers[cluster_ids] + rng.normal(0, 0.85, size=(n_tasks, 2))
        service_burden = np.clip(rng.lognormal(mean=1.25, sigma=0.35, size=n_tasks), 1.5, 12.0)
        package_count = rng.integers(1, 6, size=n_tasks)
        time_window_end = rng.uniform(150, 520, size=n_tasks)
        order = _nearest_neighbor_order(coords, rng)

        travel_minutes: list[float] = []
        finish_minutes: list[float] = []
        estimated_workload: list[float] = []
        current = np.array([0.0, 0.0])
        cumulative = 0.0
        for _rank, task_idx in enumerate(order):
            distance = float(np.linalg.norm(coords[task_idx] - current))
            current_hour = start_hour + cumulative / 60.0
            dynamic_congestion = congestion_index + 0.12 * (16.0 <= current_hour <= 19.0)
            travel = distance / max(base_speed / dynamic_congestion, 5.0) * 60.0
            travel *= 1.0 + 0.30 * weather_severity
            travel *= float(rng.lognormal(mean=0.0, sigma=0.12))
            service = float(service_burden[task_idx] * rng.lognormal(mean=0.0, sigma=0.15))
            stall = (
                float(rng.exponential(4.0))
                if rng.random() < (0.03 + 0.08 * weather_severity)
                else 0.0
            )
            cumulative += travel + service + stall
            travel_minutes.append(travel)
            finish_minutes.append(cumulative)
            estimated_workload.append(
                distance / max(base_speed, 5.0) * 60.0 + service_burden[task_idx]
            )
            current = coords[task_idx]

        total_workload = float(sum(estimated_workload))
        density = _route_density(coords)
        aoi_transitions = sum(
            int(cluster_ids[order[i]] != cluster_ids[order[i - 1]]) for i in range(1, len(order))
        )
        aoi_transition_burden = aoi_transitions / max(n_tasks - 1, 1)

        completed_candidates = list(range(1, max(n_tasks - 1, 2), snapshot_stride))
        if not completed_candidates:
            completed_candidates = [1]

        for completed_count in completed_candidates:
            if completed_count >= n_tasks:
                continue
            current_rank = completed_count - 1
            query_elapsed = float(finish_minutes[current_rank])
            query_time = route_start + pd.Timedelta(minutes=query_elapsed)
            completed_workload = float(sum(estimated_workload[:completed_count]))
            remaining_workload = float(total_workload - completed_workload)
            progress = completed_workload / max(total_workload, 1e-6)
            route_phase = completed_count / n_tasks
            recent_start = max(0, completed_count - 3)
            prior_finish = finish_minutes[recent_start - 1] if recent_start > 0 else 0.0
            recent_elapsed = max(query_elapsed - prior_finish, 1e-3)
            recent_workload = float(sum(estimated_workload[recent_start:completed_count]))
            recent_pace = recent_workload / recent_elapsed
            current_coord = coords[order[current_rank]]
            remaining_indices = order[completed_count:]
            remaining_coords = coords[remaining_indices]
            remaining_spread = float(
                np.mean(np.linalg.norm(remaining_coords - remaining_coords.mean(axis=0), axis=1))
            )
            snapshot_id = f"S-{day_index:05d}-{completed_count:03d}"
            route_remaining = float(finish_minutes[-1] - query_elapsed)
            next_completion = float(finish_minutes[completed_count] - query_elapsed)
            query_hour = query_time.hour + query_time.minute / 60.0

            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "courier_id": courier_id,
                    "work_date": date.date().isoformat(),
                    "query_time": query_time.isoformat(),
                    "city": city,
                    "query_hour": query_hour,
                    "elapsed_minutes": query_elapsed,
                    "completed_task_count": completed_count,
                    "remaining_task_count": n_tasks - completed_count,
                    "initial_task_count": n_tasks,
                    "completed_workload": completed_workload,
                    "remaining_workload": remaining_workload,
                    "initial_workload": total_workload,
                    "observed_progress": progress,
                    "route_phase": route_phase,
                    "recent_pace": recent_pace,
                    "task_density": density,
                    "remaining_spread": remaining_spread,
                    "aoi_transition_burden": aoi_transition_burden,
                    "weather_severity": weather_severity,
                    "congestion_proxy": congestion_index,
                    "trajectory_missingness": float(rng.binomial(1, 0.03)),
                    "target_next_minutes": next_completion,
                    "target_route_remaining_minutes": route_remaining,
                }
            )

            current_cluster = cluster_ids[order[current_rank]]
            for local_rank, task_idx in enumerate(remaining_indices):
                delta = coords[task_idx] - current_coord
                distance = float(np.linalg.norm(delta))
                slack = float(time_window_end[task_idx] - query_elapsed)
                task_rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "task_id": f"T-{day_index:05d}-{task_idx:03d}",
                        "actual_rank": local_rank,
                        "target_next": int(local_rank == 0),
                        "distance_to_current": distance,
                        "delta_x": float(delta[0]),
                        "delta_y": float(delta[1]),
                        "service_burden": float(service_burden[task_idx]),
                        "package_count": int(package_count[task_idx]),
                        "same_aoi": int(cluster_ids[task_idx] == current_cluster),
                        "time_window_slack": slack,
                        "task_density": density,
                    }
                )

    snapshots = (
        pd.DataFrame(snapshot_rows).sort_values(["work_date", "query_time"]).reset_index(drop=True)
    )
    pending_tasks = (
        pd.DataFrame(task_rows).sort_values(["snapshot_id", "actual_rank"]).reset_index(drop=True)
    )
    return SyntheticData(snapshots=snapshots, pending_tasks=pending_tasks)
