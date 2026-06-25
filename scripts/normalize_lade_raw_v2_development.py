from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
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


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _canonical_ds(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric) or not numeric.is_integer():
        return ""
    return str(int(numeric))


def _stable_rank(seed: int, city: str, courier_id: str, ds: str) -> str:
    payload = f"{seed}|{city}|{courier_id}|{ds}".encode()
    return hashlib.sha256(payload).hexdigest()


def _quota(cities: tuple[str, ...], total: int) -> dict[str, int]:
    if total < len(cities):
        raise ValueError("max-courier-days must be at least number of cities")
    base, remainder = divmod(total, len(cities))
    return {city: base + int(index < remainder) for index, city in enumerate(cities)}


def _parse_ds(values: pd.Series, year: int) -> pd.Series:
    canonical = values.map(_canonical_ds).astype("string").str.zfill(4)
    return pd.to_datetime(
        str(year) + "-" + canonical.str[:2] + "-" + canonical.str[2:],
        format="%Y-%m-%d",
        errors="coerce",
    )


def _parse_event(values: pd.Series, year: int) -> pd.Series:
    return pd.to_datetime(
        str(year) + "-" + values.astype("string").str.strip(),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def _raw_keys_by_city(input_dir: Path, cities: tuple[str, ...]) -> dict[str, set[tuple[str, str]]]:
    keys = {city: set() for city in cities}
    city_set = set(cities)
    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["city", "courier_id", "ds"], batch_size=65_536):
            frame = batch.to_pandas()
            frame = frame.loc[frame["city"].isin(city_set)]
            for city, courier_id, ds in zip(
                frame["city"].astype(str),
                frame["courier_id"].astype(str),
                frame["ds"],
                strict=True,
            ):
                canonical = _canonical_ds(ds)
                if canonical:
                    keys[city].add((courier_id, canonical))
    return keys


def _select(
    keys_by_city: dict[str, set[tuple[str, str]]],
    *,
    cities: tuple[str, ...],
    total: int,
    seed: int,
    exclude: dict[str, set[tuple[str, str]]] | None = None,
) -> dict[str, set[tuple[str, str]]]:
    limits = _quota(cities, total)
    exclude = exclude or {}
    selected: dict[str, set[tuple[str, str]]] = {}
    for city in cities:
        eligible = keys_by_city[city].difference(exclude.get(city, set()))
        ordered = sorted(
            eligible,
            key=lambda item: _stable_rank(seed, city, item[0], item[1]),
        )
        if len(ordered) < limits[city]:
            raise RuntimeError(
                f"Only {len(ordered)} eligible courier-days for {city}; need {limits[city]}"
            )
        selected[city] = set(ordered[: limits[city]])
    return selected


def _merge_key_sets(*parts: dict[str, set[tuple[str, str]]]) -> dict[str, set[tuple[str, str]]]:
    merged: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for part in parts:
        for city, keys in part.items():
            merged[city].update(keys)
    return dict(merged)


def _normalize(
    input_dir: Path,
    *,
    selected: dict[str, set[tuple[str, str]]],
    cities: tuple[str, ...],
    year: int,
    max_duration_minutes: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    selected_all = {
        (city, courier_id, ds) for city, values in selected.items() for courier_id, ds in values
    }
    parts: list[pd.DataFrame] = []
    counts: Counter[str] = Counter()
    city_set = set(cities)

    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=RAW_COLUMNS, batch_size=65_536):
            frame = batch.to_pandas()
            counts["raw_rows_scanned"] += len(frame)
            frame = frame.loc[frame["city"].isin(city_set)].copy()
            if frame.empty:
                continue

            keys = [
                (str(city), str(courier_id), _canonical_ds(ds))
                for city, courier_id, ds in zip(
                    frame["city"], frame["courier_id"], frame["ds"], strict=True
                )
            ]
            keep = np.fromiter((key in selected_all for key in keys), dtype=bool, count=len(frame))
            frame = frame.loc[keep].copy()
            counts["rows_in_selected_courier_days"] += len(frame)
            if frame.empty:
                continue

            frame["accept_dt"] = _parse_event(frame["accept_time"], year)
            frame["delivery_dt"] = _parse_event(frame["delivery_time"], year)
            frame["ds_dt"] = _parse_ds(frame["ds"], year)

            valid_time = frame[["accept_dt", "delivery_dt", "ds_dt"]].notna().all(axis=1)
            counts["dropped_invalid_timestamp"] += int((~valid_time).sum())
            frame = frame.loc[valid_time].copy()
            if frame.empty:
                continue

            ds_match = frame["delivery_dt"].dt.normalize().eq(frame["ds_dt"].dt.normalize())
            counts["dropped_delivery_ds_mismatch"] += int((~ds_match).sum())
            frame = frame.loc[ds_match].copy()
            if frame.empty:
                continue

            duration = (frame["delivery_dt"] - frame["accept_dt"]).dt.total_seconds().div(60.0)
            nonnegative = duration.ge(0.0)
            counts["dropped_negative_duration"] += int((~nonnegative).sum())
            frame = frame.loc[nonnegative].copy()
            duration = duration.loc[nonnegative]
            if frame.empty:
                continue

            bounded = duration.le(max_duration_minutes)
            counts["dropped_duration_over_limit"] += int((~bounded).sum())
            frame = frame.loc[bounded].copy()
            if frame.empty:
                continue

            coordinates = frame[["lat", "lng"]].apply(pd.to_numeric, errors="coerce")
            valid_coordinates = (
                np.isfinite(coordinates.to_numpy(dtype=float)).all(axis=1)
                & coordinates["lat"].between(-90.0, 90.0).to_numpy()
                & coordinates["lng"].between(-180.0, 180.0).to_numpy()
            )
            counts["dropped_invalid_coordinate"] += int((~valid_coordinates).sum())
            frame = frame.loc[valid_coordinates].copy()
            if frame.empty:
                continue

            normalized = pd.DataFrame(
                {
                    "courier_id": frame["city"].astype(str)
                    + "::"
                    + frame["courier_id"].astype(str),
                    "city": frame["city"].astype(str),
                    "accept_time": frame["accept_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "delivery_time": frame["delivery_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "latitude": pd.to_numeric(frame["lat"], errors="coerce"),
                    "longitude": pd.to_numeric(frame["lng"], errors="coerce"),
                    "work_date": frame["ds_dt"].dt.strftime("%Y-%m-%d"),
                    "task_id": frame["city"].astype(str) + "::" + frame["order_id"].astype(str),
                    "aoi_id": frame["aoi_id"].astype(str),
                    "aoi_type": frame["aoi_type"].astype(str),
                    "package_count": 1,
                }
            )
            parts.append(normalized)
            counts["normalized_rows_before_day_filter"] += len(normalized)

    if not parts:
        raise RuntimeError("No valid rows remain after V2 development normalization")

    output = pd.concat(parts, ignore_index=True).sort_values(
        ["work_date", "city", "courier_id", "delivery_time", "task_id"], kind="stable"
    )
    day_size = output.groupby(["courier_id", "work_date"], sort=False).size()
    keep_days = day_size[day_size >= 4].index
    index = pd.MultiIndex.from_frame(output[["courier_id", "work_date"]])
    keep = index.isin(keep_days)
    counts["dropped_rows_from_small_courier_days"] = int((~keep).sum())
    output = output.loc[keep].reset_index(drop=True)
    if output.empty:
        raise RuntimeError("No V2 development courier-days with at least four valid tasks remain")
    return output, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a fully disjoint LaDe-D V2 development cohort."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--max-courier-days", type=int, default=2000)
    parser.add_argument("--anchor-year", type=int, default=2022)
    parser.add_argument("--max-duration-minutes", type=float, default=720.0)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--confirmatory-seed", type=int, default=314159)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_csv = args.output_csv.resolve()
    report_path = args.report_path.resolve()
    cities = tuple(map(str, args.cities))
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if args.max_courier_days < len(cities):
        raise ValueError("max-courier-days must be at least number of cities")

    keys_by_city = _raw_keys_by_city(input_dir, cities)
    pilot = _select(
        keys_by_city,
        cities=cities,
        total=int(args.max_courier_days),
        seed=int(args.pilot_seed),
    )
    confirmatory = _select(
        keys_by_city,
        cities=cities,
        total=int(args.max_courier_days),
        seed=int(args.confirmatory_seed),
        exclude=pilot,
    )
    prior = _merge_key_sets(pilot, confirmatory)
    development = _select(
        keys_by_city,
        cities=cities,
        total=int(args.max_courier_days),
        seed=int(args.seed),
        exclude=prior,
    )

    normalized, counts = _normalize(
        input_dir,
        selected=development,
        cities=cities,
        year=int(args.anchor_year),
        max_duration_minutes=float(args.max_duration_minutes),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_csv, index=False)

    report = {
        "status": "PASS",
        "purpose": "fully disjoint LaDe-D V2 development cohort for feature design only",
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "report_path": str(report_path),
        "cities": list(cities),
        "sampling": {
            "method": "deterministic_whole_courier_day_hash_rank",
            "development_seed": int(args.seed),
            "max_courier_days": int(args.max_courier_days),
            "excluded_selected_courier_days": {
                "pilot": {city: len(values) for city, values in pilot.items()},
                "confirmatory": {city: len(values) for city, values in confirmatory.items()},
            },
            "selected_courier_days_by_city": {
                city: len(values) for city, values in development.items()
            },
        },
        "filters": {
            "delivery_ds_match_required": True,
            "max_duration_minutes": float(args.max_duration_minutes),
            "min_courier_day_tasks": 4,
        },
        "quality_counts": counts,
        "normalized_rows": int(len(normalized)),
        "normalized_courier_days": int(normalized.groupby(["courier_id", "work_date"]).ngroups),
        "normalized_rows_by_city": {
            str(city): int(count)
            for city, count in normalized["city"].value_counts().sort_index().items()
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, indent=2, default=_json_default))
    print("V2_DEVELOPMENT_NORMALIZATION_COMPLETE")


if __name__ == "__main__":
    main()
