from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "order_id",
    "city",
    "courier_id",
    "lng",
    "lat",
    "aoi_id",
    "aoi_type",
    "accept_time",
    "delivery_time",
    "ds",
]


def _json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_2022(timestamp: pd.Series) -> pd.Series:
    raw = timestamp.astype("string").str.strip()
    return pd.to_datetime(
        "2022-" + raw,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def _ds_mmdd(ds: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(ds, errors="coerce")
    return numeric.astype("Int64").astype("string").str.zfill(4)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {name: None for name in ("min", "p01", "p05", "p50", "p90", "p95", "p99", "max")}
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-profile raw LaDe-D Parquet files without loading them all at once."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    files = sorted(args.input_dir.glob("delivery_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No delivery_*.parquet files found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    city_rows: Counter[str] = Counter()
    city_couriers: dict[str, set[str]] = {}
    courier_day_counts: Counter[tuple[str, str, str]] = Counter()
    durations: list[np.ndarray] = []
    per_file: list[dict[str, Any]] = []

    total_rows = 0
    invalid_accept = 0
    invalid_delivery = 0
    invalid_coordinates = 0
    negative_duration = 0
    over_12_hours = 0
    over_24_hours = 0
    same_day = 0
    cross_day = 0
    ds_mismatch = 0
    valid_duration_rows = 0

    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        schema_names = set(parquet.schema_arrow.names)
        missing = set(REQUIRED_COLUMNS).difference(schema_names)
        if missing:
            raise ValueError(f"{file_path.name} missing required columns: {sorted(missing)}")

        file_rows = 0
        file_cities: Counter[str] = Counter()
        file_duration_rows = 0

        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=REQUIRED_COLUMNS,
        ):
            frame = batch.to_pandas()
            rows = len(frame)
            total_rows += rows
            file_rows += rows

            city = frame["city"].astype("string").fillna("<missing>").astype(str)
            courier = frame["courier_id"].astype("string").fillna("<missing>").astype(str)
            ds = _ds_mmdd(frame["ds"]).fillna("<missing>").astype(str)

            city_counts = city.value_counts(dropna=False)
            for key, value in city_counts.items():
                city_rows[str(key)] += int(value)
                file_cities[str(key)] += int(value)

            for city_name, values in pd.DataFrame({"city": city, "courier": courier}).groupby(
                "city"
            ):
                city_couriers.setdefault(str(city_name), set()).update(values["courier"].tolist())

            group_counts = (
                pd.DataFrame({"city": city, "courier": courier, "ds": ds})
                .groupby(["city", "courier", "ds"], sort=False)
                .size()
            )
            for key, value in group_counts.items():
                courier_day_counts[(str(key[0]), str(key[1]), str(key[2]))] += int(value)

            accept = _parse_2022(frame["accept_time"])
            delivery = _parse_2022(frame["delivery_time"])

            invalid_accept += int(accept.isna().sum())
            invalid_delivery += int(delivery.isna().sum())

            latitude = pd.to_numeric(frame["lat"], errors="coerce")
            longitude = pd.to_numeric(frame["lng"], errors="coerce")
            valid_coordinate = (
                np.isfinite(latitude.to_numpy(dtype=float, na_value=np.nan))
                & np.isfinite(longitude.to_numpy(dtype=float, na_value=np.nan))
                & latitude.between(-90.0, 90.0).to_numpy()
                & longitude.between(-180.0, 180.0).to_numpy()
            )
            invalid_coordinates += int((~valid_coordinate).sum())

            valid_time = accept.notna() & delivery.notna()
            duration = (delivery[valid_time] - accept[valid_time]).dt.total_seconds() / 60.0
            negative_duration += int((duration < 0.0).sum())

            nonnegative = duration[duration >= 0.0].to_numpy(dtype=float)
            if len(nonnegative):
                durations.append(nonnegative)
                valid_duration_rows += len(nonnegative)
                file_duration_rows += len(nonnegative)
                over_12_hours += int((nonnegative > 12.0 * 60.0).sum())
                over_24_hours += int((nonnegative > 24.0 * 60.0).sum())

            same_day += int((accept[valid_time].dt.date == delivery[valid_time].dt.date).sum())
            cross_day += int((accept[valid_time].dt.date != delivery[valid_time].dt.date).sum())

            accepted_mmdd = accept.dt.strftime("%m%d").astype("string")
            comparable = accepted_mmdd.notna() & (ds != "<missing>")
            ds_mismatch += int((accepted_mmdd[comparable] != ds[comparable]).sum())

        per_file.append(
            {
                "file": file_path.name,
                "rows": file_rows,
                "row_groups": int(parquet.metadata.num_row_groups),
                "cities": dict(sorted(file_cities.items())),
                "valid_nonnegative_duration_rows": file_duration_rows,
            }
        )
        print(f"PROFILED {file_path.name}: rows={file_rows:,}", flush=True)

    all_durations = np.concatenate(durations) if durations else np.array([], dtype=float)
    courier_day_values = np.fromiter(courier_day_counts.values(), dtype=np.int64)

    city_summary = {
        city: {
            "rows": int(city_rows[city]),
            "unique_couriers": int(len(city_couriers.get(city, set()))),
            "courier_days": int(sum(1 for key in courier_day_counts if key[0] == city)),
        }
        for city in sorted(city_rows)
    }

    valid_time_rows = same_day + cross_day
    summary: dict[str, Any] = {
        "status": "PASS",
        "input_dir": str(args.input_dir),
        "files": per_file,
        "total_rows": int(total_rows),
        "cities": city_summary,
        "quality": {
            "invalid_accept_time_rows": int(invalid_accept),
            "invalid_delivery_time_rows": int(invalid_delivery),
            "invalid_coordinate_rows": int(invalid_coordinates),
            "negative_duration_rows": int(negative_duration),
            "valid_nonnegative_duration_rows": int(valid_duration_rows),
            "same_day_delivery_rate": (
                float(same_day / valid_time_rows) if valid_time_rows else None
            ),
            "cross_day_delivery_rate": (
                float(cross_day / valid_time_rows) if valid_time_rows else None
            ),
            "accept_date_vs_ds_mismatch_rows": int(ds_mismatch),
            "duration_over_12h_rate": (
                float(over_12_hours / valid_duration_rows) if valid_duration_rows else None
            ),
            "duration_over_24h_rate": (
                float(over_24_hours / valid_duration_rows) if valid_duration_rows else None
            ),
        },
        "duration_minutes": _quantiles(all_durations),
        "courier_day_order_counts": _quantiles(courier_day_values.astype(float)),
        "recommended_next_step": {
            "pilot": "Normalize a deterministic whole-courier-day subset first; do not sample individual orders.",
            "initial_target_courier_days": 2_000,
            "initial_cities": ["Hangzhou", "Shanghai", "Chongqing"],
            "reason": "Preserves event order and task-set visibility while keeping snapshot construction bounded.",
        },
    }

    output_path = args.output_dir / "lade_raw_profile.json"
    output_path.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print("=" * 100)
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"PROFILE_PATH={output_path}")
    print("PROFILE_COMPLETE")


if __name__ == "__main__":
    main()
