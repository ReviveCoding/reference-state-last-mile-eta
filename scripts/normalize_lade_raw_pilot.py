from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW_COLUMNS = [
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


def _stable_rank(seed: int, city: str, courier_id: object, ds: object) -> str:
    value = f"{seed}|{city}|{courier_id}|{ds}".encode()
    return hashlib.sha256(value).hexdigest()


def _parse_event_time(values: pd.Series, *, anchor_year: int) -> pd.Series:
    text = values.astype("string").str.strip()
    return pd.to_datetime(
        str(anchor_year) + "-" + text,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def _parse_ds(values: pd.Series, *, anchor_year: int) -> pd.Series:
    text = values.astype("Int64").astype("string").str.zfill(4)
    return pd.to_datetime(
        str(anchor_year) + "-" + text.str[:2] + "-" + text.str[2:],
        format="%Y-%m-%d",
        errors="coerce",
    )


def _city_quota(cities: list[str], total: int) -> dict[str, int]:
    if total < len(cities):
        raise ValueError("max-courier-days must be at least the number of selected cities")

    base, remainder = divmod(total, len(cities))
    return {city: base + int(index < remainder) for index, city in enumerate(cities)}


def _iter_raw_batches(path: Path, *, batch_size: int):
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(columns=RAW_COLUMNS, batch_size=batch_size)


def _build_selected_keys(
    files: list[Path],
    *,
    cities: list[str],
    max_courier_days: int,
    seed: int,
    batch_size: int,
) -> dict[str, set[tuple[str, str, int]]]:
    city_set = set(cities)
    candidates: dict[str, set[tuple[str, str, int]]] = defaultdict(set)

    for path in files:
        for batch in _iter_raw_batches(path, batch_size=batch_size):
            frame = batch.to_pandas()
            frame = frame.loc[frame["city"].isin(city_set)]
            if frame.empty:
                continue

            for city, courier_id, ds in frame[["city", "courier_id", "ds"]].itertuples(
                index=False,
                name=None,
            ):
                candidates[str(city)].add((str(city), str(courier_id), int(ds)))

    quota = _city_quota(cities, max_courier_days)
    selected: dict[str, set[tuple[str, str, int]]] = {}

    for city in cities:
        ranked = sorted(
            candidates[city],
            key=lambda key: _stable_rank(seed, key[0], key[1], key[2]),
        )
        if len(ranked) < quota[city]:
            raise RuntimeError(
                f"Only {len(ranked)} courier-days are available for {city}; need {quota[city]}"
            )
        selected[city] = set(ranked[: quota[city]])

    return selected


def _normalize_selected_rows(
    files: list[Path],
    *,
    selected: dict[str, set[tuple[str, str, int]]],
    anchor_year: int,
    max_duration_minutes: float,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    selected_all = set().union(*selected.values())
    kept_parts: list[pd.DataFrame] = []
    counters: Counter[str] = Counter()

    for path in files:
        for batch in _iter_raw_batches(path, batch_size=batch_size):
            frame = batch.to_pandas()
            counters["raw_rows_scanned"] += int(len(frame))

            keys = list(
                zip(
                    frame["city"].astype(str),
                    frame["courier_id"].astype(str),
                    frame["ds"].astype(int),
                    strict=False,
                )
            )
            selected_mask = np.fromiter(
                (key in selected_all for key in keys),
                dtype=bool,
                count=len(frame),
            )
            frame = frame.loc[selected_mask].copy()
            counters["rows_in_selected_courier_days"] += int(len(frame))

            if frame.empty:
                continue

            accept = _parse_event_time(frame["accept_time"], anchor_year=anchor_year)
            delivery = _parse_event_time(frame["delivery_time"], anchor_year=anchor_year)
            ds_date = _parse_ds(frame["ds"], anchor_year=anchor_year)

            finite_coords = np.isfinite(
                frame[["lat", "lng"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            ).all(axis=1)
            valid_dates = accept.notna() & delivery.notna() & ds_date.notna()
            nonnegative = delivery >= accept
            bounded_duration = (delivery - accept).dt.total_seconds().div(
                60.0
            ) <= max_duration_minutes
            delivery_matches_ds = delivery.dt.normalize().eq(ds_date)

            valid = (
                finite_coords & valid_dates & nonnegative & bounded_duration & delivery_matches_ds
            )

            counters["dropped_invalid_coordinate"] += int((~finite_coords).sum())
            counters["dropped_invalid_timestamp"] += int((~valid_dates).sum())
            counters["dropped_negative_duration"] += int((valid_dates & ~nonnegative).sum())
            counters["dropped_duration_over_limit"] += int(
                (valid_dates & nonnegative & ~bounded_duration).sum()
            )
            counters["dropped_delivery_ds_mismatch"] += int(
                (valid_dates & ~delivery_matches_ds).sum()
            )

            frame = frame.loc[valid].copy()
            if frame.empty:
                continue

            accept = accept.loc[valid]
            delivery = delivery.loc[valid]
            ds_date = ds_date.loc[valid]

            normalized = pd.DataFrame(
                {
                    "courier_id": frame["city"].astype(str)
                    + "::"
                    + frame["courier_id"].astype(str),
                    "city": frame["city"].astype(str),
                    "accept_time": accept.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "delivery_time": delivery.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "latitude": pd.to_numeric(frame["lat"], errors="coerce"),
                    "longitude": pd.to_numeric(frame["lng"], errors="coerce"),
                    "task_id": frame["city"].astype(str) + "::" + frame["order_id"].astype(str),
                    "work_date": ds_date.dt.strftime("%Y-%m-%d"),
                    "aoi_id": frame["aoi_id"].astype("Int64").astype(str),
                    "aoi_type": frame["aoi_type"].astype("Int64").astype(str),
                    "package_count": 1,
                    "raw_duration_minutes": (delivery - accept).dt.total_seconds().div(60.0),
                    "source_ds": frame["ds"].astype(int),
                }
            )

            kept_parts.append(normalized)
            counters["normalized_rows_before_day_filter"] += int(len(normalized))

    if not kept_parts:
        raise RuntimeError("No valid normalized rows remain after raw data quality filters")

    output = pd.concat(kept_parts, ignore_index=True)
    output = output.sort_values(
        ["work_date", "city", "courier_id", "delivery_time", "task_id"],
        kind="stable",
    ).reset_index(drop=True)

    day_sizes = output.groupby(["courier_id", "work_date"], sort=False).size()
    valid_days = day_sizes[day_sizes >= 4].index
    valid_day_index = pd.MultiIndex.from_frame(output[["courier_id", "work_date"]])
    keep_days = valid_day_index.isin(valid_days)

    counters["dropped_rows_from_small_courier_days"] = int((~keep_days).sum())
    output = output.loc[keep_days].reset_index(drop=True)

    if output.empty:
        raise RuntimeError("No courier-days with at least four valid tasks remain")

    report = {
        "selected_courier_days_by_city": {city: int(len(keys)) for city, keys in selected.items()},
        "quality_counts": dict(sorted(counters.items())),
        "normalized_rows": int(len(output)),
        "normalized_courier_days": int(output.groupby(["courier_id", "work_date"]).ngroups),
        "normalized_rows_by_city": {
            str(city): int(count)
            for city, count in output["city"].value_counts().sort_index().items()
        },
        "normalized_courier_days_by_city": {
            str(city): int(count)
            for city, count in output.groupby("city")
            .apply(
                lambda frame: frame.groupby(["courier_id", "work_date"]).ngroups,
                include_groups=False,
            )
            .items()
        },
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize a deterministic whole-courier-day LaDe-D raw Parquet pilot"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Hangzhou", "Shanghai", "Chongqing"],
    )
    parser.add_argument("--max-courier-days", type=int, default=2000)
    parser.add_argument("--max-duration-minutes", type=float, default=720.0)
    parser.add_argument("--anchor-year", type=int, default=2022)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=131072)
    args = parser.parse_args()

    if args.max_courier_days < len(args.cities):
        raise ValueError("max-courier-days must be at least the number of selected cities")
    if args.max_duration_minutes <= 0.0:
        raise ValueError("max-duration-minutes must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    input_dir = args.input_dir.resolve()
    files = sorted(input_dir.glob("delivery_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No delivery_*.parquet files found in {input_dir}")

    selected = _build_selected_keys(
        files,
        cities=list(args.cities),
        max_courier_days=int(args.max_courier_days),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
    )

    normalized, report = _normalize_selected_rows(
        files,
        selected=selected,
        anchor_year=int(args.anchor_year),
        max_duration_minutes=float(args.max_duration_minutes),
        batch_size=int(args.batch_size),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    normalized.to_csv(args.output_csv, index=False)

    full_report = {
        "status": "PASS",
        "input_dir": str(input_dir),
        "output_csv": str(args.output_csv.resolve()),
        "report_path": str(args.report_path.resolve()),
        "cities": list(args.cities),
        "sampling": {
            "method": "deterministic_whole_courier_day_hash_rank",
            "max_courier_days": int(args.max_courier_days),
            "seed": int(args.seed),
        },
        "time_handling": {
            "anchor_year": int(args.anchor_year),
            "work_date_source": "ds (official delivery-date field)",
            "delivery_ds_match_required": True,
            "max_duration_minutes": float(args.max_duration_minutes),
        },
        **report,
    }

    args.report_path.write_text(
        json.dumps(full_report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(full_report, indent=2))
    print("NORMALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
