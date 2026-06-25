from __future__ import annotations

import argparse
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
DEFAULT_EXCLUDE = (
    _SCRIPT_REPO_ROOT / "artifacts" / "lade_raw_pilot" / "lade_delivery_normalized.csv"
)
CITIES = ("Hangzhou", "Shanghai", "Chongqing")


def canonical_ds(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""
    return str(int(numeric))


def parse_ds(value: object, year: int) -> pd.Timestamp | pd.NaT:
    ds = canonical_ds(value).zfill(4)
    return pd.to_datetime(f"{year}-{ds[:2]}-{ds[2:]}", format="%Y-%m-%d", errors="coerce")


def parse_timestamp(value: object, year: int) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(
        f"{year}-{value}",
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )


def stable_rank(seed: int, city: str, courier_id: str, ds: str) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}|{city}|{courier_id}|{ds}".encode()).hexdigest()


def pilot_excluded(path: Path) -> dict[str, set[tuple[str, str]]]:
    frame = pd.read_csv(path, usecols=["courier_id", "work_date"])
    out: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for combined, work_date in zip(
        frame["courier_id"].astype(str),
        frame["work_date"].astype(str),
        strict=True,
    ):
        city, sep, courier_id = combined.partition("::")
        if not sep:
            raise ValueError(f"Malformed pilot courier_id: {combined!r}")
        ds = str(int(pd.Timestamp(work_date).strftime("%m%d")))
        out[city].add((courier_id, ds))
    return dict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--exclude-csv", type=Path, default=DEFAULT_EXCLUDE)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--max-courier-days", type=int, default=2000)
    parser.add_argument("--anchor-year", type=int, default=2022)
    parser.add_argument("--max-duration-minutes", type=float, default=720.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    exclude_csv = args.exclude_csv.resolve()
    output = args.output.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if not exclude_csv.is_file():
        raise FileNotFoundError(exclude_csv)

    excluded = pilot_excluded(exclude_csv)
    keys_by_city: dict[str, set[tuple[str, str]]] = {city: set() for city in CITIES}

    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["city", "courier_id", "ds"], batch_size=65_536):
            frame = batch.to_pandas()
            frame = frame.loc[frame["city"].isin(CITIES)]
            for city, courier_id, ds in zip(
                frame["city"].astype(str),
                frame["courier_id"].astype(str),
                frame["ds"],
                strict=True,
            ):
                key_ds = canonical_ds(ds)
                if key_ds:
                    keys_by_city[city].add((courier_id, key_ds))

    per_city = args.max_courier_days // len(CITIES)
    remainder = args.max_courier_days % len(CITIES)
    selected: dict[str, set[tuple[str, str]]] = {}
    cohort_counts: dict[str, dict[str, int]] = {}
    for index, city in enumerate(CITIES):
        eligible = keys_by_city[city].difference(excluded.get(city, set()))
        limit = per_city + int(index < remainder)
        ranked = sorted(eligible, key=lambda item: stable_rank(args.seed, city, item[0], item[1]))
        selected[city] = set(ranked[:limit])
        cohort_counts[city] = {
            "raw_courier_days": len(keys_by_city[city]),
            "excluded_pilot_courier_days": len(excluded.get(city, set())),
            "eligible_courier_days": len(eligible),
            "selected_courier_days": len(selected[city]),
        }

    counts: Counter[str] = Counter()
    samples: list[dict[str, object]] = []
    for path in sorted(input_dir.glob("delivery_*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["city", "courier_id", "ds", "accept_time", "delivery_time", "lat", "lng"],
            batch_size=65_536,
        ):
            frame = batch.to_pandas()
            frame = frame.loc[frame["city"].isin(CITIES)].copy()
            counts["rows_city_filter"] += len(frame)
            if frame.empty:
                continue

            keys = [
                (str(city), str(courier_id), canonical_ds(ds))
                for city, courier_id, ds in zip(
                    frame["city"], frame["courier_id"], frame["ds"], strict=True
                )
            ]
            mask = np.fromiter(
                (
                    ds and (courier_id, ds) in selected.get(city, set())
                    for city, courier_id, ds in keys
                ),
                dtype=bool,
                count=len(keys),
            )
            frame = frame.loc[mask].copy()
            counts["rows_selected_key"] += len(frame)
            if frame.empty:
                continue

            for row in frame.head(max(0, 5 - len(samples))).itertuples(index=False):
                samples.append(
                    {
                        "city": str(row.city),
                        "courier_id": str(row.courier_id),
                        "ds_raw": str(row.ds),
                        "ds_canonical": canonical_ds(row.ds),
                        "accept_time": str(row.accept_time),
                        "delivery_time": str(row.delivery_time),
                    }
                )

            accept = pd.to_datetime(
                str(args.anchor_year) + "-" + frame["accept_time"].astype("string"),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            delivery = pd.to_datetime(
                str(args.anchor_year) + "-" + frame["delivery_time"].astype("string"),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )
            ds_text = frame["ds"].map(canonical_ds).astype("string").str.zfill(4)
            ds_date = pd.to_datetime(
                str(args.anchor_year) + "-" + ds_text.str[:2] + "-" + ds_text.str[2:],
                format="%Y-%m-%d",
                errors="coerce",
            )
            valid_time = accept.notna() & delivery.notna() & ds_date.notna()
            counts["rows_valid_timestamps"] += int(valid_time.sum())
            matched = valid_time & (delivery.dt.normalize() == ds_date.dt.normalize())
            counts["rows_delivery_ds_match"] += int(matched.sum())
            duration = (delivery - accept).dt.total_seconds() / 60.0
            duration_ok = matched & duration.ge(0.0) & duration.le(args.max_duration_minutes)
            counts["rows_duration_ok"] += int(duration_ok.sum())
            coords = frame[["lat", "lng"]].apply(pd.to_numeric, errors="coerce")
            coord_ok = (
                np.isfinite(coords.to_numpy(dtype=float)).all(axis=1)
                & coords["lat"].between(-90, 90).to_numpy()
                & coords["lng"].between(-180, 180).to_numpy()
            )
            counts["rows_coordinate_ok"] += int((duration_ok & coord_ok).sum())

    payload = {
        "status": "PASS",
        "purpose": "diagnose confirmatory cohort key matching without writing a cohort",
        "input_dir": str(input_dir),
        "exclude_csv": str(exclude_csv),
        "cohort_counts": cohort_counts,
        "row_counts": dict(counts),
        "selected_row_samples": samples,
        "diagnosis": (
            "rows_selected_key must be positive. If it is zero, the issue is canonical key mismatch. "
            "If rows_selected_key is positive but rows_coordinate_ok is zero, the issue is a filter/parsing condition."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("CONFIRMATORY_DIAGNOSIS_COMPLETE")


if __name__ == "__main__":
    main()
