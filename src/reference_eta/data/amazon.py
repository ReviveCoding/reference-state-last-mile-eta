from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AmazonRouteReplayData:
    route_id: str
    station_id: str
    departure_time: pd.Timestamp
    stops: pd.DataFrame
    travel_times_seconds: pd.DataFrame
    recorded_order: list[str] | None


def load_amazon_route_json(path: str | Path) -> dict[str, Any]:
    """Load one official-format Amazon challenge JSON object."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Amazon route artifact must be a JSON object")
    return payload


def summarize_route_payload(payload: dict[str, Any]) -> dict[str, int]:
    routes = len(payload)
    nested_stops = 0
    for route in payload.values():
        if isinstance(route, dict) and isinstance(route.get("stops"), dict):
            nested_stops += len(route["stops"])
    return {"routes": routes, "nested_stops": nested_stops}


def _timestamp(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return pd.to_datetime(text, utc=True, errors="raise")


def _finite_nonnegative(value: object, *, name: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def normalize_amazon_route(
    route_id: str,
    *,
    route_data: dict[str, Any],
    package_data: dict[str, Any],
    travel_times: dict[str, Any],
    actual_sequences: dict[str, Any] | None = None,
) -> AmazonRouteReplayData:
    """Normalize one official-format challenge route for offline replay.

    Arrival timestamps are not invented. Package time windows, planned service seconds,
    stop geometry, historical average travel times, and the observed order are preserved.
    Multiple package windows at one stop are intersected; conflicts are explicitly flagged.
    """

    if not str(route_id).strip():
        raise ValueError("route_id cannot be blank")
    if not all(isinstance(value, dict) for value in (route_data, package_data, travel_times)):
        raise TypeError("Amazon route, package, and travel-time artifacts must be mappings")
    if route_id not in route_data or route_id not in package_data or route_id not in travel_times:
        raise KeyError(f"Route {route_id!r} is missing from one or more required Amazon artifacts")
    route = route_data[route_id]
    packages_by_stop = package_data[route_id]
    matrix_payload = travel_times[route_id]
    if not isinstance(route, dict) or not isinstance(route.get("stops"), dict):
        raise ValueError("route_data entry must contain a stops object")
    stops_payload = route["stops"]
    if not isinstance(packages_by_stop, dict) or not isinstance(matrix_payload, dict):
        raise ValueError("Package and travel-time route entries must be objects")
    extra_package_stops = set(packages_by_stop).difference(stops_payload)
    if extra_package_stops:
        raise ValueError(f"Package payload references unknown stops: {sorted(extra_package_stops)}")
    stop_ids = list(stops_payload)
    if not stop_ids:
        raise ValueError("Amazon route contains no stops")

    station_candidates = [
        stop_id for stop_id, stop in stops_payload.items() if str(stop.get("type", "")) == "Station"
    ]
    if len(station_candidates) != 1:
        raise ValueError("Amazon route must contain exactly one Station stop")
    station_id = station_candidates[0]
    station = stops_payload[station_id]
    origin_lat = float(station["lat"])
    origin_lng = float(station["lng"])
    departure = pd.to_datetime(
        f"{route['date_YYYY_MM_DD']} {route['departure_time_utc']}", utc=True, errors="raise"
    )

    stop_rows: list[dict[str, Any]] = []
    for stop_id, stop in stops_payload.items():
        lat = float(stop["lat"])
        lng = float(stop["lng"])
        if not np.isfinite([lat, lng]).all() or not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
            raise ValueError(f"Invalid coordinates for stop {stop_id}")
        mean_lat = np.radians((lat + origin_lat) / 2.0)
        x_km = (lng - origin_lng) * 111.320 * np.cos(mean_lat)
        y_km = (lat - origin_lat) * 110.574

        stop_packages = packages_by_stop.get(stop_id, {})
        if not isinstance(stop_packages, dict):
            raise ValueError(f"Package payload for stop {stop_id} must be an object")
        service_seconds = 0.0
        volume_cm3 = 0.0
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []
        for package_id, package in stop_packages.items():
            if not isinstance(package, dict):
                raise ValueError(f"Package {package_id} at stop {stop_id} must be an object")
            service_seconds += _finite_nonnegative(
                package.get("planned_service_time_seconds", 0.0),
                name=f"planned_service_time_seconds[{package_id}]",
            )
            dimensions = package.get("dimensions", {}) or {}
            if not isinstance(dimensions, dict):
                raise ValueError(f"dimensions[{package_id}] must be an object")
            depth = _finite_nonnegative(
                dimensions.get("depth_cm", 0.0), name=f"depth_cm[{package_id}]"
            )
            height = _finite_nonnegative(
                dimensions.get("height_cm", 0.0), name=f"height_cm[{package_id}]"
            )
            width = _finite_nonnegative(
                dimensions.get("width_cm", 0.0), name=f"width_cm[{package_id}]"
            )
            volume_cm3 += depth * height * width
            window = package.get("time_window", {}) or {}
            if not isinstance(window, dict):
                raise ValueError(f"time_window[{package_id}] must be an object")
            start = _timestamp(window.get("start_time_utc"))
            end = _timestamp(window.get("end_time_utc"))
            if start is not None:
                starts.append(start)
            if end is not None:
                ends.append(end)

        window_start = max(starts) if starts else None
        window_end = min(ends) if ends else None
        window_start_minutes = (
            max((window_start - departure).total_seconds() / 60.0, 0.0)
            if window_start is not None
            else 0.0
        )
        window_end_minutes = (
            (window_end - departure).total_seconds() / 60.0 if window_end is not None else np.inf
        )
        window_conflict = bool(
            window_start is not None and window_end is not None and window_start > window_end
        )
        stop_rows.append(
            {
                "stop_id": str(stop_id),
                "lat": lat,
                "lng": lng,
                "x": float(x_km),
                "y": float(y_km),
                "type": str(stop.get("type", "")),
                "zone_id": str(stop.get("zone_id", "")),
                "package_count": int(len(stop_packages)),
                "package_volume_cm3": float(volume_cm3),
                "service_minutes": 0.0 if stop_id == station_id else service_seconds / 60.0,
                "window_start": float(window_start_minutes),
                "window_end": float(window_end_minutes),
                "window_conflict": window_conflict,
            }
        )
    stops = pd.DataFrame(stop_rows).set_index("stop_id", drop=False)

    matrix = pd.DataFrame(index=stop_ids, columns=stop_ids, dtype=float)
    for source in stop_ids:
        if source not in matrix_payload or not isinstance(matrix_payload[source], dict):
            raise ValueError(f"Missing travel-time row for stop {source}")
        for destination in stop_ids:
            if destination not in matrix_payload[source]:
                raise ValueError(f"Missing travel time {source}->{destination}")
            value = float(matrix_payload[source][destination])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid travel time {source}->{destination}: {value}")
            matrix.loc[source, destination] = value

    recorded_order: list[str] | None = None
    if actual_sequences is not None:
        route_sequence = actual_sequences.get(route_id, {})
        mapping = route_sequence.get("actual") if isinstance(route_sequence, dict) else None
        if not isinstance(mapping, dict):
            raise ValueError(f"Missing actual sequence mapping for route {route_id}")
        if set(mapping) != set(stop_ids):
            raise ValueError("Actual sequence stop set does not match route_data stop set")
        try:
            ranks = {str(stop): int(rank) for stop, rank in mapping.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("Actual sequence ranks must be integers") from error
        if sorted(ranks.values()) != list(range(len(stop_ids))):
            raise ValueError("Actual sequence ranks must be unique and contiguous from zero")
        recorded_order = [stop for stop, _ in sorted(ranks.items(), key=lambda item: item[1])]
        if recorded_order[0] != station_id:
            raise ValueError("Recorded Amazon sequence must begin at the Station stop")

    return AmazonRouteReplayData(
        route_id=route_id,
        station_id=station_id,
        departure_time=departure,
        stops=stops,
        travel_times_seconds=matrix,
        recorded_order=recorded_order,
    )
