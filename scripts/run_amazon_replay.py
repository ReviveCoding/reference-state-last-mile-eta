from __future__ import annotations

import argparse
from pathlib import Path

from reference_eta.data.amazon import load_amazon_route_json, normalize_amazon_route
from reference_eta.decisions.amazon_replay import replay_normalized_amazon_route

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/amazon_sample")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/amazon_official_shape_replay.csv"
    )
    args = parser.parse_args()
    route_data = load_amazon_route_json(args.input_dir / "route_data.json")
    package_data = load_amazon_route_json(args.input_dir / "package_data.json")
    travel_times = load_amazon_route_json(args.input_dir / "travel_times.json")
    sequences = load_amazon_route_json(args.input_dir / "actual_sequences.json")
    route_id = next(iter(route_data))
    normalized = normalize_amazon_route(
        route_id,
        route_data=route_data,
        package_data=package_data,
        travel_times=travel_times,
        actual_sequences=sequences,
    )
    report = replay_normalized_amazon_route(normalized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
