from reference_eta.data.amazon import normalize_amazon_route
from reference_eta.decisions.amazon_replay import replay_normalized_amazon_route


def test_official_shaped_amazon_route_normalizes_and_replays() -> None:
    route_id = "R"
    route_data = {
        route_id: {
            "station_code": "D",
            "date_YYYY_MM_DD": "2018-01-01",
            "departure_time_utc": "08:00:00",
            "stops": {
                "S": {"lat": 40.0, "lng": -74.0, "type": "Station", "zone_id": "D"},
                "A": {"lat": 40.01, "lng": -74.0, "type": "Dropoff", "zone_id": "A"},
                "B": {"lat": 40.02, "lng": -74.0, "type": "Dropoff", "zone_id": "B"},
            },
        }
    }

    def package(end: str) -> dict[str, object]:
        return {
            "scan_status": "DELIVERED",
            "time_window": {"start_time_utc": "2018-01-01 08:00:00", "end_time_utc": end},
            "planned_service_time_seconds": 60,
            "dimensions": {"depth_cm": 1, "height_cm": 1, "width_cm": 1},
        }

    package_data = {
        route_id: {
            "S": {},
            "A": {"P1": package("2018-01-01 10:00:00")},
            "B": {"P2": package("2018-01-01 11:00:00")},
        }
    }
    travel_times = {
        route_id: {
            "S": {"S": 0, "A": 300, "B": 600},
            "A": {"S": 300, "A": 0, "B": 300},
            "B": {"S": 600, "A": 300, "B": 0},
        }
    }
    actual = {route_id: {"actual": {"S": 0, "A": 1, "B": 2}}}
    route = normalize_amazon_route(
        route_id,
        route_data=route_data,
        package_data=package_data,
        travel_times=travel_times,
        actual_sequences=actual,
    )
    report = replay_normalized_amazon_route(route)
    assert set(report["strategy"]) == {
        "historical_travel_nearest",
        "time_window_greedy",
        "time_window_two_opt",
        "recorded_sequence",
    }
    assert (report["completion_minutes"] > 0).all()
