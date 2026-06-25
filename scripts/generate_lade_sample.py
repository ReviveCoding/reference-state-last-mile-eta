from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_CITY_CENTERS = {
    "Boston": (42.3601, -71.0589),
    "Chicago": (41.8781, -87.6298),
}


def generate_sample(*, n_days: int = 40, tasks_per_day: int = 10, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2026-01-01 08:00:00")
    for day_index in range(n_days):
        city = "Boston" if day_index % 2 == 0 else "Chicago"
        center_lat, center_lon = _CITY_CENTERS[city]
        work_start = (
            start + pd.Timedelta(days=day_index) + pd.Timedelta(minutes=int(rng.integers(0, 45)))
        )
        courier_id = f"sample-{city.lower()}-{day_index % 7:02d}"
        cumulative = 15.0
        for task_index in range(tasks_per_day):
            accept_time = work_start + pd.Timedelta(minutes=max(task_index - 3, 0) * 3)
            cumulative += float(rng.uniform(7.0, 16.0))
            delivery_time = work_start + pd.Timedelta(minutes=cumulative)
            window_end = work_start + pd.Timedelta(minutes=180 + task_index * 18)
            rows.append(
                {
                    "courier_id": courier_id,
                    "city": city,
                    "accept_time": accept_time.isoformat(),
                    "delivery_time": delivery_time.isoformat(),
                    "latitude": center_lat + float(rng.normal(0.0, 0.012)),
                    "longitude": center_lon + float(rng.normal(0.0, 0.015)),
                    "task_id": f"sample-{day_index:03d}-{task_index:03d}",
                    "service_minutes": float(rng.uniform(3.0, 8.0)),
                    "package_count": int(rng.integers(1, 5)),
                    "aoi_id": f"aoi-{task_index // 3}",
                    "time_window_end": window_end.isoformat(),
                    "weather_severity": float(rng.beta(2.0, 8.0)),
                    "congestion_proxy": float(rng.uniform(0.9, 1.35)),
                    "trajectory_missingness": float(rng.binomial(1, 0.02)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-days", type=int, default=40)
    parser.add_argument("--tasks-per-day", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.n_days < 8 or args.tasks_per_day < 4:
        raise SystemExit("n-days must be >= 8 and tasks-per-day must be >= 4")
    frame = generate_sample(n_days=args.n_days, tasks_per_day=args.tasks_per_day, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} normalized LaDe-style rows to {args.output}")


if __name__ == "__main__":
    main()
