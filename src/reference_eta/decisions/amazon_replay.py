from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from reference_eta.data.amazon import AmazonRouteReplayData


@dataclass(frozen=True)
class RouteEvaluation:
    distance: float
    completion_minutes: float
    time_window_violations: int
    total_lateness_minutes: float


@dataclass(frozen=True)
class AmazonMatrixEvaluation:
    travel_minutes: float
    completion_minutes: float
    time_window_violations: int
    total_lateness_minutes: float


def _travel_minutes(a: np.ndarray, b: np.ndarray, speed_units_per_hour: float) -> float:
    return float(np.linalg.norm(a - b) / max(speed_units_per_hour, 1e-6) * 60.0)


def evaluate_order(
    stops: pd.DataFrame,
    order: list[int],
    *,
    speed_units_per_hour: float = 20.0,
) -> RouteEvaluation:
    current = np.array([0.0, 0.0])
    elapsed = 0.0
    distance = 0.0
    violations = 0
    lateness = 0.0
    for index in order:
        row = stops.iloc[index]
        coord = np.array([float(row["x"]), float(row["y"])])
        leg_distance = float(np.linalg.norm(coord - current))
        distance += leg_distance
        elapsed += _travel_minutes(current, coord, speed_units_per_hour)
        window_start = float(row.get("window_start", 0.0))
        window_end = float(row.get("window_end", np.inf))
        if elapsed < window_start:
            elapsed = window_start
        if elapsed > window_end:
            violations += 1
            lateness += elapsed - window_end
        elapsed += float(row.get("service_minutes", 5.0))
        current = coord
    return RouteEvaluation(distance, elapsed, violations, lateness)


def nearest_neighbor_order(stops: pd.DataFrame) -> list[int]:
    remaining = set(range(len(stops)))
    current = np.array([0.0, 0.0])
    order: list[int] = []
    while remaining:
        chosen = min(
            remaining,
            key=lambda index: np.linalg.norm(
                stops.iloc[index][["x", "y"]].to_numpy(dtype=float) - current
            ),
        )
        order.append(chosen)
        current = stops.iloc[chosen][["x", "y"]].to_numpy(dtype=float)
        remaining.remove(chosen)
    return order


def time_window_aware_order(
    stops: pd.DataFrame,
    *,
    speed_units_per_hour: float = 20.0,
    lateness_penalty: float = 5.0,
) -> list[int]:
    remaining = set(range(len(stops)))
    current = np.array([0.0, 0.0])
    elapsed = 0.0
    order: list[int] = []
    while remaining:
        current_coord = current.copy()
        elapsed_now = elapsed

        def score(
            index: int,
            current_coord: np.ndarray = current_coord,
            elapsed_now: float = elapsed_now,
        ) -> float:
            row = stops.iloc[index]
            coord = row[["x", "y"]].to_numpy(dtype=float)
            travel = _travel_minutes(current_coord, coord, speed_units_per_hour)
            arrival = elapsed_now + travel
            lateness = max(arrival - float(row.get("window_end", np.inf)), 0.0)
            return travel + lateness_penalty * lateness

        chosen = min(remaining, key=score)
        row = stops.iloc[chosen]
        coord = row[["x", "y"]].to_numpy(dtype=float)
        elapsed += _travel_minutes(current, coord, speed_units_per_hour)
        elapsed = max(elapsed, float(row.get("window_start", 0.0)))
        elapsed += float(row.get("service_minutes", 5.0))
        current = coord
        order.append(chosen)
        remaining.remove(chosen)
    return order


def two_opt_local_search(
    stops: pd.DataFrame,
    initial_order: list[int],
    *,
    speed_units_per_hour: float = 20.0,
    violation_penalty: float = 250.0,
    lateness_penalty: float = 5.0,
    max_passes: int = 8,
) -> list[int]:
    def objective(order: list[int]) -> float:
        evaluation = evaluate_order(stops, order, speed_units_per_hour=speed_units_per_hour)
        return (
            evaluation.completion_minutes
            + violation_penalty * evaluation.time_window_violations
            + lateness_penalty * evaluation.total_lateness_minutes
        )

    best = list(initial_order)
    best_score = objective(best)
    for _ in range(max_passes):
        improved = False
        for i in range(0, max(len(best) - 1, 0)):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                candidate_score = objective(candidate)
                if candidate_score + 1e-9 < best_score:
                    best, best_score = candidate, candidate_score
                    improved = True
        if not improved:
            break
    return best


def demo_replay(seed: int = 42, n_stops: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stops = pd.DataFrame(
        {
            "x": rng.normal(0, 3, n_stops),
            "y": rng.normal(0, 3, n_stops),
            "service_minutes": rng.uniform(3, 9, n_stops),
            "window_start": rng.uniform(0, 120, n_stops),
        }
    )
    stops["window_end"] = stops["window_start"] + rng.uniform(90, 220, n_stops)
    orders = {
        "recorded_proxy": list(range(n_stops)),
        "nearest_neighbor": nearest_neighbor_order(stops),
        "time_window_greedy": time_window_aware_order(stops),
    }
    orders["time_window_two_opt"] = two_opt_local_search(stops, orders["time_window_greedy"])
    rows = []
    for strategy, order in orders.items():
        evaluation = evaluate_order(stops, order)
        rows.append({"strategy": strategy, **evaluation.__dict__})
    return pd.DataFrame(rows)


def _matrix_delivery_order(route: AmazonRouteReplayData, order: list[str]) -> list[str]:
    if not isinstance(route, AmazonRouteReplayData):
        raise TypeError("route must be AmazonRouteReplayData")
    delivery_stops = set(route.stops.index) - {route.station_id}
    cleaned = [str(stop) for stop in order if str(stop) != route.station_id]
    if len(cleaned) != len(set(cleaned)) or set(cleaned) != delivery_stops:
        raise ValueError("Order must contain every non-station stop exactly once")
    return cleaned


def evaluate_amazon_matrix_order(
    route: AmazonRouteReplayData,
    order: list[str],
) -> AmazonMatrixEvaluation:
    """Evaluate an order using official historical average travel-time seconds."""

    cleaned = _matrix_delivery_order(route, order)
    current = route.station_id
    elapsed = 0.0
    travel_minutes = 0.0
    violations = 0
    lateness = 0.0
    for stop_id in cleaned:
        leg = float(route.travel_times_seconds.loc[current, stop_id]) / 60.0
        travel_minutes += leg
        elapsed += leg
        row = route.stops.loc[stop_id]
        elapsed = max(elapsed, float(row["window_start"]))
        if bool(row["window_conflict"]) or elapsed > float(row["window_end"]):
            violations += 1
            if np.isfinite(float(row["window_end"])):
                lateness += max(elapsed - float(row["window_end"]), 0.0)
        elapsed += float(row["service_minutes"])
        current = stop_id
    elapsed += float(route.travel_times_seconds.loc[current, route.station_id]) / 60.0
    travel_minutes += float(route.travel_times_seconds.loc[current, route.station_id]) / 60.0
    return AmazonMatrixEvaluation(travel_minutes, elapsed, violations, lateness)


def amazon_nearest_neighbor_order(route: AmazonRouteReplayData) -> list[str]:
    remaining = set(route.stops.index) - {route.station_id}
    current = route.station_id
    order: list[str] = []
    while remaining:
        chosen = min(
            remaining, key=lambda stop: float(route.travel_times_seconds.loc[current, stop])
        )
        order.append(str(chosen))
        remaining.remove(chosen)
        current = chosen
    return order


def amazon_time_window_order(
    route: AmazonRouteReplayData,
    *,
    lateness_penalty: float = 5.0,
) -> list[str]:
    remaining = set(route.stops.index) - {route.station_id}
    current = route.station_id
    elapsed = 0.0
    order: list[str] = []
    while remaining:
        current_stop = current
        elapsed_now = elapsed

        def score(
            stop_id: str,
            current_stop: str = current_stop,
            elapsed_now: float = elapsed_now,
        ) -> float:
            travel = float(route.travel_times_seconds.loc[current_stop, stop_id]) / 60.0
            arrival = elapsed_now + travel
            end = float(route.stops.loc[stop_id, "window_end"])
            lateness = max(arrival - end, 0.0) if np.isfinite(end) else 0.0
            return travel + lateness_penalty * lateness

        chosen = min(remaining, key=score)
        elapsed += float(route.travel_times_seconds.loc[current, chosen]) / 60.0
        elapsed = max(elapsed, float(route.stops.loc[chosen, "window_start"]))
        elapsed += float(route.stops.loc[chosen, "service_minutes"])
        order.append(str(chosen))
        remaining.remove(chosen)
        current = chosen
    return order


def amazon_two_opt(
    route: AmazonRouteReplayData,
    initial_order: list[str],
    *,
    violation_penalty: float = 250.0,
    lateness_penalty: float = 5.0,
    max_passes: int = 8,
) -> list[str]:
    best = _matrix_delivery_order(route, initial_order)

    def objective(order: list[str]) -> float:
        result = evaluate_amazon_matrix_order(route, order)
        return (
            result.completion_minutes
            + violation_penalty * result.time_window_violations
            + lateness_penalty * result.total_lateness_minutes
        )

    best_score = objective(best)
    for _ in range(max_passes):
        improved = False
        for i in range(max(len(best) - 1, 0)):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                score = objective(candidate)
                if score + 1e-9 < best_score:
                    best, best_score = candidate, score
                    improved = True
        if not improved:
            break
    return best


def replay_normalized_amazon_route(route: AmazonRouteReplayData) -> pd.DataFrame:
    strategies: dict[str, list[str]] = {
        "historical_travel_nearest": amazon_nearest_neighbor_order(route),
    }
    strategies["time_window_greedy"] = amazon_time_window_order(route)
    strategies["time_window_two_opt"] = amazon_two_opt(route, strategies["time_window_greedy"])
    if route.recorded_order is not None:
        strategies["recorded_sequence"] = route.recorded_order
    rows = []
    for strategy, order in strategies.items():
        result = evaluate_amazon_matrix_order(route, order)
        rows.append({"route_id": route.route_id, "strategy": strategy, **result.__dict__})
    return pd.DataFrame(rows)
