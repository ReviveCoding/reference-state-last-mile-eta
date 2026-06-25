from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/amazon_sample"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_id = "RouteID_SAMPLE"
    route_data = {
        route_id: {
            "station_code": "DST1",
            "date_YYYY_MM_DD": "2018-07-23",
            "departure_time_utc": "08:00:00",
            "executor_capacity_cm3": 1000000,
            "route_score": "High",
            "stops": {
                "ST": {"lat": 40.0, "lng": -74.0, "type": "Station", "zone_id": "DEPOT"},
                "AA": {"lat": 40.01, "lng": -74.00, "type": "Dropoff", "zone_id": "A-1"},
                "AB": {"lat": 40.00, "lng": -73.99, "type": "Dropoff", "zone_id": "A-1"},
                "AC": {"lat": 40.02, "lng": -73.98, "type": "Dropoff", "zone_id": "B-1"},
            },
        }
    }
    package_data = {
        route_id: {
            "ST": {},
            "AA": {
                "P1": {
                    "scan_status": "DELIVERED",
                    "time_window": {
                        "start_time_utc": "2018-07-23 08:00:00",
                        "end_time_utc": "2018-07-23 10:00:00",
                    },
                    "planned_service_time_seconds": 120,
                    "dimensions": {"depth_cm": 10, "height_cm": 10, "width_cm": 10},
                }
            },
            "AB": {
                "P2": {
                    "scan_status": "DELIVERED",
                    "time_window": {
                        "start_time_utc": "2018-07-23 08:30:00",
                        "end_time_utc": "2018-07-23 11:00:00",
                    },
                    "planned_service_time_seconds": 90,
                    "dimensions": {"depth_cm": 20, "height_cm": 10, "width_cm": 10},
                }
            },
            "AC": {
                "P3": {
                    "scan_status": "DELIVERED",
                    "time_window": {
                        "start_time_utc": "2018-07-23 09:00:00",
                        "end_time_utc": "2018-07-23 12:00:00",
                    },
                    "planned_service_time_seconds": 60,
                    "dimensions": {"depth_cm": 10, "height_cm": 20, "width_cm": 10},
                }
            },
        }
    }
    stops = ["ST", "AA", "AB", "AC"]
    seconds = {
        "ST": {"ST": 0, "AA": 300, "AB": 360, "AC": 720},
        "AA": {"ST": 300, "AA": 0, "AB": 240, "AC": 420},
        "AB": {"ST": 360, "AA": 240, "AB": 0, "AC": 300},
        "AC": {"ST": 720, "AA": 420, "AB": 300, "AC": 0},
    }
    travel_times = {
        route_id: {source: {dest: seconds[source][dest] for dest in stops} for source in stops}
    }
    actual_sequences = {route_id: {"actual": {"ST": 0, "AA": 1, "AB": 2, "AC": 3}}}
    for name, payload in {
        "route_data.json": route_data,
        "package_data.json": package_data,
        "travel_times.json": travel_times,
        "actual_sequences.json": actual_sequences,
    }.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote official-shaped synthetic Amazon sample to {args.output_dir}")


if __name__ == "__main__":
    main()
