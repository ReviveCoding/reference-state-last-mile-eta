from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def grouped_temporal_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> TemporalSplit:
    fractions = [train_fraction, validation_fraction, calibration_fraction, test_fraction]
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        raise ValueError("Every split fraction must be strictly between 0 and 1")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0, got {sum(fractions):.6f}")
    if frame.empty:
        raise ValueError("Cannot split an empty frame")
    required = {"work_date", "courier_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")

    if frame[["work_date", "courier_id"]].isna().any().any():
        raise ValueError("work_date and courier_id cannot be missing")
    ordered_dates = sorted(frame["work_date"].astype(str).unique())
    n_dates = len(ordered_dates)
    if n_dates < 8:
        raise ValueError("At least eight distinct work dates are required for temporal splitting")

    cut1 = max(1, int(n_dates * train_fraction))
    cut2 = max(cut1 + 1, int(n_dates * (train_fraction + validation_fraction)))
    cut3 = max(
        cut2 + 1, int(n_dates * (train_fraction + validation_fraction + calibration_fraction))
    )
    cut3 = min(cut3, n_dates - 1)
    date_sets = [
        set(ordered_dates[:cut1]),
        set(ordered_dates[cut1:cut2]),
        set(ordered_dates[cut2:cut3]),
        set(ordered_dates[cut3:]),
    ]
    parts = [frame[frame["work_date"].astype(str).isin(dates)].copy() for dates in date_sets]
    if any(part.empty for part in parts):
        raise ValueError("Temporal split produced an empty partition")

    group_sets = [set(zip(p["courier_id"], p["work_date"].astype(str), strict=True)) for p in parts]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            if group_sets[i].intersection(group_sets[j]):
                raise AssertionError("courier_id + work_date leakage across temporal partitions")

    return TemporalSplit(*parts)
