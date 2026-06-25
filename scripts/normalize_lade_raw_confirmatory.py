from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    os.environ.get(
        "LADE_DATA_ROOT",
        _SCRIPT_REPO_ROOT / "data" / "LaDe-D_dataset" / "data",
    )
)
DEFAULT_CITIES = ("Hangzhou", "Shanghai", "Chongqing")
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


def _stable_rank(seed: int, city: str, courier_id: str, ds: str) -> str:
    value = f"{seed}|{city}|{courier_id}|{ds}".encode()
    return hashlib.sha256(value).hexdigest()


def _canonical_ds(value: object) -> str:
    """Return a stable MMDD key for raw LaDe ds values across Arrow/Pandas dtypes."""
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric) or not numeric.is_integer():
        return ""
    return str(int(numeric))


def _parse_ds(series: pd.Series, year: int) -> pd.Series:
    canonical = series.map(_canonical_ds).astype("string").str.zfill(4)
    return pd.to_datetime(
        str(year) + "-" + canonical.str[:2] + "-" + canonical.str[2:],
        format="%Y-%m-%d",
        errors="coerce",
    )


def _parse_timestamp(series: pd.Series, year: int) -> pd.Series:
    return pd.to_datetime(
        str(year) + "-" + series.astype("string"),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def _excluded_keys(path: Path | None) -> dict[str, set[tuple[str, str]]]:
    if path is None:
        return {}

    frame = pd.read_csv(path, usecols=["courier_id", "work_date"])
    required = {"courier_id", "work_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Exclude CSV is missing required columns: {sorted(missing)}")

    excluded: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for combined, work_date in zip(
        frame["courier_id"].astype(str),
        frame["work_date"].astype(str),
        strict=True,
    ):
        city, separator, courier_id = combined.partition("::")
        if not separator or not city or not courier_id:
            raise ValueError("Exclude CSV courier_id must use '<city>::<courier_id>' format")
        date = pd.Timestamp(work_date)
        ds = str(int(date.strftime("%m%d")))
        excluded[city].add((courier_id, ds))

    return dict(excluded)


def _selected_keys(
    input_dir: Path,
    *,
    cities: tuple[str, ...],
    max_courier_days: int,
    seed: int,
    excluded: dict[str, set[tuple[str, str]]],
) -> dict[str, set[tuple[str, str]]]:
    keys_by_city: dict[str, set[tuple[str, str]]] = {city: set() for city in cities}

    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["city", "courier_id", "ds"],
            batch_size=65_536,
        ):
            frame = batch.to_pandas()
            frame = frame.loc[frame["city"].isin(cities)]
            for city, courier_id, raw_ds in zip(
                frame["city"].astype(str),
                frame["courier_id"].astype(str),
                frame["ds"],
                strict=True,
            ):
                ds = _canonical_ds(raw_ds)
                if ds:
                    keys_by_city[city].add((courier_id, ds))

    per_city = max_courier_days // len(cities)
    remainder = max_courier_days % len(cities)
    selected: dict[str, set[tuple[str, str]]] = {}

    for index, city in enumerate(cities):
        limit = per_city + int(index < remainder)
        eligible = keys_by_city[city].difference(excluded.get(city, set()))
        ranked = sorted(
            eligible,
            key=lambda item: _stable_rank(seed, city, item[0], item[1]),
        )
        selected[city] = set(ranked[:limit])

    return selected


def _normalize_batch(
    frame: pd.DataFrame,
    *,
    selected: dict[str, set[tuple[str, str]]],
    cities: tuple[str, ...],
    anchor_year: int,
    max_duration_minutes: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    frame = frame.loc[frame["city"].isin(cities)].copy()
    if frame.empty:
        return frame, counts

    candidate_keys = [
        (str(city), str(courier_id), _canonical_ds(raw_ds))
        for city, courier_id, raw_ds in zip(
            frame["city"],
            frame["courier_id"],
            frame["ds"],
            strict=True,
        )
    ]
    keep = [
        bool(ds) and (courier_id, ds) in selected.get(city, set())
        for city, courier_id, ds in candidate_keys
    ]
    frame = frame.loc[np.asarray(keep, dtype=bool)].copy()
    counts["rows_in_selected_courier_days"] += len(frame)
    if frame.empty:
        return frame, counts

    frame["accept_dt"] = _parse_timestamp(frame["accept_time"], anchor_year)
    frame["delivery_dt"] = _parse_timestamp(frame["delivery_time"], anchor_year)
    frame["ds_dt"] = _parse_ds(frame["ds"], anchor_year)

    valid_time = frame["accept_dt"].notna() & frame["delivery_dt"].notna() & frame["ds_dt"].notna()
    counts["dropped_invalid_timestamp"] += int((~valid_time).sum())
    frame = frame.loc[valid_time].copy()

    delivery_date_matches_ds = frame["delivery_dt"].dt.normalize() == frame["ds_dt"].dt.normalize()
    counts["dropped_delivery_ds_mismatch"] += int((~delivery_date_matches_ds).sum())
    frame = frame.loc[delivery_date_matches_ds].copy()

    duration = (frame["delivery_dt"] - frame["accept_dt"]).dt.total_seconds() / 60.0
    nonnegative = duration >= 0.0
    counts["dropped_negative_duration"] += int((~nonnegative).sum())
    frame = frame.loc[nonnegative].copy()
    duration = duration.loc[nonnegative]

    under_limit = duration <= max_duration_minutes
    counts["dropped_duration_over_limit"] += int((~under_limit).sum())
    frame = frame.loc[under_limit].copy()

    coordinates = frame[["lat", "lng"]].apply(pd.to_numeric, errors="coerce")
    valid_coordinate = (
        np.isfinite(coordinates.to_numpy(dtype=float)).all(axis=1)
        & coordinates["lat"].between(-90.0, 90.0).to_numpy()
        & coordinates["lng"].between(-180.0, 180.0).to_numpy()
    )
    counts["dropped_invalid_coordinate"] += int((~valid_coordinate).sum())
    frame = frame.loc[valid_coordinate].copy()

    if frame.empty:
        return frame, counts

    normalized = pd.DataFrame(
        {
            "courier_id": frame["city"].astype(str) + "::" + frame["courier_id"].astype(str),
            "city": frame["city"].astype(str),
            "accept_time": frame["accept_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "delivery_time": frame["delivery_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "latitude": pd.to_numeric(frame["lat"], errors="coerce"),
            "longitude": pd.to_numeric(frame["lng"], errors="coerce"),
            "work_date": frame["ds_dt"].dt.strftime("%Y-%m-%d"),
            "task_id": frame["city"].astype(str) + "::" + frame["order_id"].astype(str),
            "aoi_id": frame["aoi_id"].astype(str),
            "aoi_type": pd.to_numeric(frame["aoi_type"], errors="coerce"),
            "package_count": 1,
        }
    )
    return normalized, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an independent deterministic whole-courier-day LaDe-D confirmatory cohort."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-courier-days", type=int, default=2000)
    parser.add_argument("--anchor-year", type=int, default=2022)
    parser.add_argument("--max-duration-minutes", type=float, default=720.0)
    parser.add_argument("--min-courier-day-tasks", type=int, default=4)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument(
        "--exclude-normalized-csv",
        type=Path,
        default=None,
        help="Normalized pilot CSV whose courier-days must be excluded.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_csv = args.output_csv.resolve()
    report_path = args.report_path.resolve()
    cities = tuple(map(str, args.cities))

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.max_courier_days < len(cities):
        raise ValueError("max-courier-days must be at least the number of selected cities")

    excluded = _excluded_keys(
        args.exclude_normalized_csv.resolve() if args.exclude_normalized_csv is not None else None
    )
    selected = _selected_keys(
        input_dir,
        cities=cities,
        max_courier_days=int(args.max_courier_days),
        seed=int(args.seed),
        excluded=excluded,
    )

    frames: list[pd.DataFrame] = []
    quality: dict[str, int] = defaultdict(int)

    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=REQUIRED_COLUMNS,
            batch_size=65_536,
        ):
            normalized, batch_counts = _normalize_batch(
                batch.to_pandas(),
                selected=selected,
                cities=cities,
                anchor_year=int(args.anchor_year),
                max_duration_minutes=float(args.max_duration_minutes),
            )
            for key, value in batch_counts.items():
                quality[key] += int(value)
            if not normalized.empty:
                frames.append(normalized)

    if not frames:
        raise RuntimeError("No rows remained after confirmatory cohort normalization")

    output = pd.concat(frames, ignore_index=True)
    quality["normalized_rows_before_day_filter"] = int(len(output))

    counts = output.groupby(["courier_id", "work_date"], sort=False)["task_id"].transform("size")
    keep_day = counts >= int(args.min_courier_day_tasks)
    quality["dropped_rows_from_small_courier_days"] = int((~keep_day).sum())
    output = output.loc[keep_day].copy()

    output = output.sort_values(
        ["work_date", "courier_id", "delivery_time", "task_id"],
        kind="stable",
    ).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)

    report = {
        "status": "PASS",
        "purpose": "independent confirmatory cohort after exploratory CatBoost city-base signal",
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "exclude_normalized_csv": (
            str(args.exclude_normalized_csv.resolve())
            if args.exclude_normalized_csv is not None
            else None
        ),
        "excluded_courier_days": int(sum(len(values) for values in excluded.values())),
        "seed": int(args.seed),
        "cities": list(cities),
        "sampling": {
            "method": "deterministic_whole_courier_day_hash_rank",
            "max_courier_days": int(args.max_courier_days),
            "selected_courier_days_by_city": {city: len(keys) for city, keys in selected.items()},
        },
        "filters": {
            "delivery_ds_match_required": True,
            "max_duration_minutes": float(args.max_duration_minutes),
            "min_courier_day_tasks": int(args.min_courier_day_tasks),
        },
        "quality_counts": dict(quality),
        "normalized_rows": int(len(output)),
        "normalized_courier_days": int(
            output[["courier_id", "work_date"]].drop_duplicates().shape[0]
        ),
        "normalized_rows_by_city": {
            city: int(count) for city, count in output["city"].value_counts().items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("CONFIRMATORY_NORMALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
