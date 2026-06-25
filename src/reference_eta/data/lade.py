from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reference_eta.data.synthetic import SyntheticData

LADE_REQUIRED_COLUMNS = {
    "courier_id",
    "city",
    "accept_time",
    "delivery_time",
    "latitude",
    "longitude",
}
_EARTH_RADIUS_KM = 6371.0088


def _normalized_lade_frame(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of the normalized LaDe schema."""

    missing = LADE_REQUIRED_COLUMNS.difference(deliveries.columns)
    if missing:
        raise ValueError(
            "LaDe adapter requires normalized columns "
            f"{sorted(LADE_REQUIRED_COLUMNS)}; missing {sorted(missing)}"
        )
    if deliveries.empty:
        raise ValueError("Normalized LaDe data cannot be empty")

    frame = deliveries.copy()
    for column in ("accept_time", "delivery_time"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    if frame[["accept_time", "delivery_time"]].isna().any().any():
        raise ValueError("accept_time and delivery_time cannot be missing")
    if "time_window_end" in frame.columns:
        original_window_missing = frame["time_window_end"].isna()
        parsed_window = pd.to_datetime(frame["time_window_end"], errors="coerce")
        if (parsed_window.isna() & ~original_window_missing).any():
            raise ValueError("time_window_end contains invalid timestamps")
        frame["time_window_end"] = parsed_window
    if (frame["delivery_time"] < frame["accept_time"]).any():
        raise ValueError("delivery_time precedes accept_time in normalized LaDe data")

    coordinates = frame[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ValueError("latitude and longitude must be finite numeric values")
    frame[["latitude", "longitude"]] = coordinates
    if not frame["latitude"].between(-90.0, 90.0).all():
        raise ValueError("latitude is outside [-90, 90]")
    if not frame["longitude"].between(-180.0, 180.0).all():
        raise ValueError("longitude is outside [-180, 180]")

    for column in ("courier_id", "city"):
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{column} cannot be missing or blank")
        frame[column] = values.astype(str)

    if "work_date" not in frame.columns:
        frame["work_date"] = frame["delivery_time"].dt.date.astype(str)
    work_dates = frame["work_date"].astype("string")
    if work_dates.isna().any() or work_dates.str.strip().eq("").any():
        raise ValueError("work_date cannot be missing or blank")
    frame["work_date"] = work_dates.astype(str)
    if "task_id" not in frame.columns:
        frame["task_id"] = [f"lade-task-{index}" for index in range(len(frame))]
    task_ids = frame["task_id"].astype("string")
    if task_ids.isna().any() or task_ids.str.strip().eq("").any():
        raise ValueError("task_id cannot be missing or blank")
    frame["task_id"] = task_ids.astype(str)
    duplicated = frame.duplicated(["courier_id", "work_date", "task_id"], keep=False)
    if duplicated.any():
        raise ValueError("task_id must be unique within courier_id + work_date")

    optional_numeric_rules: dict[str, tuple[float | None, float | None]] = {
        "service_minutes": (0.0, None),
        "package_count": (0.0, None),
        "weather_severity": (0.0, 1.0),
        "congestion_proxy": (0.0, None),
        "trajectory_missingness": (0.0, 1.0),
    }
    for column, (lower, upper) in optional_numeric_rules.items():
        if column not in frame.columns:
            continue
        original_missing = frame[column].isna()
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() & ~original_missing
        if invalid.any() or not np.isfinite(numeric.dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite numeric values when present")
        present = numeric.dropna()
        if column in {"package_count", "congestion_proxy"} and (present <= 0.0).any():
            raise ValueError(f"{column} must be positive")
        if lower is not None and (present < lower).any():
            raise ValueError(f"{column} must be >= {lower}")
        if column == "package_count" and not np.allclose(present, np.round(present)):
            raise ValueError("package_count must contain integer values")
        if upper is not None and (numeric.dropna() > upper).any():
            raise ValueError(f"{column} must be <= {upper}")
        frame[column] = numeric

    return frame


def load_lade_delivery_csv(path: str | Path) -> pd.DataFrame:
    """Load an explicitly normalized LaDe delivery file without column guessing."""

    return _normalized_lade_frame(pd.read_csv(path))


def _haversine_km(a: np.ndarray, b: np.ndarray) -> float:
    lat1, lon1 = np.radians(a.astype(float))
    lat2, lon2 = np.radians(b.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0))))


def _local_xy_km(coord: np.ndarray, origin: np.ndarray) -> np.ndarray:
    lat, lon = coord.astype(float)
    origin_lat, origin_lon = origin.astype(float)
    dy = np.radians(lat - origin_lat) * _EARTH_RADIUS_KM
    mean_lat = np.radians((lat + origin_lat) / 2.0)
    dx = np.radians(lon - origin_lon) * _EARTH_RADIUS_KM * np.cos(mean_lat)
    return np.array([dx, dy], dtype=float)


def _pairwise_haversine(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    distances = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = _haversine_km(coords[i], coords[j])
            distances[i, j] = value
            distances[j, i] = value
    return distances


def _density(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    pairwise = _pairwise_haversine(coords)
    pairwise[pairwise == 0.0] = np.nan
    nearest = np.nanmin(pairwise, axis=1)
    return float(1.0 / (np.nanmean(nearest) + 0.1))


def _path_distance_km(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    return float(
        sum(_haversine_km(coords[index - 1], coords[index]) for index in range(1, len(coords)))
    )


def _nearest_neighbor_path_distance_km(current: np.ndarray, coords: np.ndarray) -> float:
    if len(coords) == 0:
        return 0.0
    remaining = set(range(len(coords)))
    position = current.astype(float)
    total = 0.0
    while remaining:
        selected = min(remaining, key=lambda index: _haversine_km(position, coords[index]))
        total += _haversine_km(position, coords[selected])
        position = coords[selected]
        remaining.remove(selected)
    return float(total)


def _service_minutes(frame: pd.DataFrame, default_service_minutes: float) -> np.ndarray:
    if "service_minutes" not in frame.columns:
        return np.full(len(frame), default_service_minutes, dtype=float)
    values = pd.to_numeric(frame["service_minutes"], errors="coerce").fillna(
        default_service_minutes
    )
    return np.maximum(values.to_numpy(dtype=float), 0.0)


def _finite_scalar(value: object, default: float, *, minimum: float | None = None) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        parsed = default
    if not np.isfinite(parsed):
        parsed = default
    if minimum is not None:
        parsed = max(parsed, minimum)
    return float(parsed)


def build_closed_set_snapshots(
    deliveries: pd.DataFrame,
    *,
    snapshot_stride: int = 3,
    default_service_minutes: float = 5.0,
    assumed_speed_kmh: float = 15.0,
) -> SyntheticData:
    """Build leakage-safe LaDe snapshots for the closed-visible-task-set track.

    At each query, only tasks accepted by that time and not yet completed are visible.
    Later accepted tasks are excluded from features and labels. Actual delivery timestamps
    are used only for target duration and route-order labels, never as time-window features.
    """

    if snapshot_stride < 1:
        raise ValueError("snapshot_stride must be at least 1")
    if not np.isfinite(default_service_minutes) or default_service_minutes < 0.0:
        raise ValueError("default_service_minutes must be finite and nonnegative")
    if not np.isfinite(assumed_speed_kmh) or assumed_speed_kmh <= 0.0:
        raise ValueError("assumed_speed_kmh must be finite and positive")

    frame = _normalized_lade_frame(deliveries)

    snapshot_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    snapshot_counter = 0

    for (courier_id, work_date), group in frame.groupby(["courier_id", "work_date"], sort=True):
        group = group.sort_values("delivery_time").reset_index(drop=True)
        if len(group) < 4:
            continue
        day_start = min(group["accept_time"].min(), group["delivery_time"].min())
        for completed_index in range(0, len(group) - 1, snapshot_stride):
            query_time = group.iloc[completed_index]["delivery_time"]
            completed = (
                group[group["delivery_time"] <= query_time].sort_values("delivery_time").copy()
            )
            visible = group[
                (group["accept_time"] <= query_time) & (group["delivery_time"] > query_time)
            ].copy()
            if visible.empty:
                continue

            current = completed.iloc[-1][["latitude", "longitude"]].to_numpy(dtype=float)
            completed_coords = completed[["latitude", "longitude"]].to_numpy(dtype=float)
            visible_coords = visible[["latitude", "longitude"]].to_numpy(dtype=float)
            completed_services = _service_minutes(completed, default_service_minutes)
            visible_services = _service_minutes(visible, default_service_minutes)

            completed_travel_minutes = (
                _path_distance_km(completed_coords) / assumed_speed_kmh * 60.0
            )
            remaining_travel_minutes = (
                _nearest_neighbor_path_distance_km(current, visible_coords)
                / assumed_speed_kmh
                * 60.0
            )
            completed_workload = float(completed_travel_minutes + completed_services.sum())
            remaining_workload = float(remaining_travel_minutes + visible_services.sum())
            initial_workload = completed_workload + remaining_workload
            completed_count = len(completed)
            total_count = completed_count + len(visible)
            observed_progress = completed_workload / max(initial_workload, 1e-6)
            elapsed_minutes = float((query_time - day_start).total_seconds() / 60.0)

            recent = completed.tail(min(3, len(completed)))
            if len(recent) >= 2:
                recent_coords = recent[["latitude", "longitude"]].to_numpy(dtype=float)
                recent_services = _service_minutes(recent.iloc[1:], default_service_minutes)
                recent_workload = (
                    _path_distance_km(recent_coords) / assumed_speed_kmh * 60.0
                    + recent_services.sum()
                )
                recent_elapsed = (
                    recent["delivery_time"].iloc[-1] - recent["delivery_time"].iloc[0]
                ).total_seconds() / 60.0
                recent_pace = float(recent_workload / max(recent_elapsed, 1e-3))
            else:
                recent_pace = float(completed_workload / max(elapsed_minutes, 1.0))

            local_visible = np.vstack([_local_xy_km(coord, current) for coord in visible_coords])
            remaining_spread = float(
                np.mean(np.linalg.norm(local_visible - local_visible.mean(axis=0), axis=1))
            )
            task_density = _density(visible_coords)
            city = str(group.iloc[0]["city"])
            snapshot_id = f"LADE-{snapshot_counter:08d}"
            snapshot_counter += 1
            next_finish = visible["delivery_time"].min()
            route_finish = visible["delivery_time"].max()
            query_hour = query_time.hour + query_time.minute / 60.0

            if "aoi_id" in completed.columns and len(completed) >= 2:
                aoi_values = completed["aoi_id"].astype(str).to_numpy()
                transitions = np.sum(aoi_values[1:] != aoi_values[:-1])
                aoi_transition_burden = float(transitions / max(len(aoi_values) - 1, 1))
                current_aoi = str(completed.iloc[-1]["aoi_id"])
            else:
                aoi_transition_burden = 0.0
                current_aoi = ""

            last_completed = completed.iloc[-1]
            weather_severity = _finite_scalar(last_completed.get("weather_severity", 0.0), 0.0)
            congestion_proxy = _finite_scalar(
                last_completed.get("congestion_proxy", 1.0), 1.0, minimum=1e-3
            )
            trajectory_missingness = _finite_scalar(
                last_completed.get("trajectory_missingness", 0.0), 0.0
            )

            snapshot_rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "courier_id": str(courier_id),
                    "work_date": str(work_date),
                    "query_time": query_time.isoformat(),
                    "city": city,
                    "query_hour": query_hour,
                    "elapsed_minutes": elapsed_minutes,
                    "completed_task_count": completed_count,
                    "remaining_task_count": len(visible),
                    "initial_task_count": total_count,
                    "completed_workload": completed_workload,
                    "remaining_workload": remaining_workload,
                    "initial_workload": initial_workload,
                    "observed_progress": observed_progress,
                    "route_phase": completed_count / max(total_count, 1),
                    "recent_pace": recent_pace,
                    "task_density": task_density,
                    "remaining_spread": remaining_spread,
                    "aoi_transition_burden": aoi_transition_burden,
                    "weather_severity": float(np.clip(weather_severity, 0.0, 1.0)),
                    "congestion_proxy": max(congestion_proxy, 1e-3),
                    "trajectory_missingness": float(np.clip(trajectory_missingness, 0.0, 1.0)),
                    "target_next_minutes": (next_finish - query_time).total_seconds() / 60.0,
                    "target_route_remaining_minutes": (route_finish - query_time).total_seconds()
                    / 60.0,
                }
            )

            visible = visible.sort_values("delivery_time").reset_index(drop=True)
            for rank, row in visible.iterrows():
                coord = row[["latitude", "longitude"]].to_numpy(dtype=float)
                delta_xy = _local_xy_km(coord, current)
                if "time_window_end" in row.index and pd.notna(row["time_window_end"]):
                    time_window_slack = float(
                        (pd.Timestamp(row["time_window_end"]) - query_time).total_seconds() / 60.0
                    )
                else:
                    time_window_slack = 0.0
                same_aoi = int(
                    bool(current_aoi)
                    and "aoi_id" in row.index
                    and str(row["aoi_id"]) == current_aoi
                )
                task_rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "task_id": str(row["task_id"]),
                        "actual_rank": int(rank),
                        "target_next": int(rank == 0),
                        "distance_to_current": _haversine_km(current, coord),
                        "delta_x": float(delta_xy[0]),
                        "delta_y": float(delta_xy[1]),
                        "service_burden": _finite_scalar(
                            row.get("service_minutes", default_service_minutes),
                            default_service_minutes,
                            minimum=0.0,
                        ),
                        "package_count": int(
                            max(_finite_scalar(row.get("package_count", 1), 1.0), 1.0)
                        ),
                        "same_aoi": same_aoi,
                        "time_window_slack": time_window_slack,
                        "task_density": task_density,
                    }
                )

    if not snapshot_rows:
        raise ValueError(
            "No valid closed-set snapshots could be constructed from normalized LaDe data"
        )
    return SyntheticData(
        snapshots=pd.DataFrame(snapshot_rows)
        .sort_values(["work_date", "query_time"])
        .reset_index(drop=True),
        pending_tasks=pd.DataFrame(task_rows)
        .sort_values(["snapshot_id", "actual_rank"])
        .reset_index(drop=True),
    )
