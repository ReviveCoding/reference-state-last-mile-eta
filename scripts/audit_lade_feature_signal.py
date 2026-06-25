from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "target_route_remaining_minutes"
CATEGORICAL_CANDIDATES = [
    "city",
    "courier_id",
    "reference_level",
    "reference_regime",
]
IDENTIFIER_COLUMNS = {"snapshot_id", "work_date", "query_time", TARGET}


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _load_partition(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}_snapshots.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing partition: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Partition is empty: {path}")
    if TARGET not in frame.columns:
        raise ValueError(f"Missing target {TARGET}: {path}")
    return frame


def _numeric_feature_columns(train: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in train.columns:
        if column in IDENTIFIER_COLUMNS:
            continue
        converted = pd.to_numeric(train[column], errors="coerce")
        if converted.notna().mean() >= 0.99:
            candidates.append(column)
    return candidates


def _numeric_summary(train: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    target = pd.to_numeric(train[TARGET], errors="coerce")
    rows: list[dict[str, object]] = []

    for column in columns:
        values = pd.to_numeric(train[column], errors="coerce")
        valid = values.notna()
        unique_count = int(values.nunique(dropna=True))
        finite = values[valid].to_numpy(dtype=float)
        nonzero_rate = float((finite != 0.0).mean()) if len(finite) else float("nan")
        std = float(np.nanstd(finite)) if len(finite) else float("nan")
        spearman = values.corr(target, method="spearman")

        rows.append(
            {
                "feature": column,
                "train_rows": int(len(train)),
                "missing_rate": float(1.0 - valid.mean()),
                "unique_values": unique_count,
                "nonzero_rate": nonzero_rate,
                "mean": float(np.nanmean(finite)) if len(finite) else float("nan"),
                "std": std,
                "min": float(np.nanmin(finite)) if len(finite) else float("nan"),
                "max": float(np.nanmax(finite)) if len(finite) else float("nan"),
                "spearman_to_target": (float(spearman) if pd.notna(spearman) else None),
                "constant_or_near_constant": bool(unique_count <= 1 or std < 1e-12),
            }
        )

    return rows


def _categorical_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for column in CATEGORICAL_CANDIDATES:
        if column not in train.columns:
            continue

        train_values = train[column].astype(str).fillna("__MISSING__")
        known = set(train_values.unique().tolist())

        for split_name, frame in [("validation", validation), ("test", test)]:
            values = frame[column].astype(str).fillna("__MISSING__")
            unseen = ~values.isin(known)
            rows.append(
                {
                    "feature": column,
                    "split": split_name,
                    "train_unique": int(len(known)),
                    "split_unique": int(values.nunique(dropna=True)),
                    "unseen_rows": int(unseen.sum()),
                    "unseen_rate": float(unseen.mean()),
                }
            )

    return rows


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    target = pd.to_numeric(frame[TARGET], errors="coerce")
    return {
        "rows": int(len(frame)),
        "target_mean": float(target.mean()),
        "target_median": float(target.median()),
        "target_p90": float(target.quantile(0.90)),
    }


def _audit(data_dir: Path, label: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    train = _load_partition(data_dir, "train")
    validation = _load_partition(data_dir, "validation")
    test = _load_partition(data_dir, "test")

    numeric_columns = _numeric_feature_columns(train)
    numeric_rows = _numeric_summary(train, numeric_columns)
    categorical_rows = _categorical_summary(train, validation, test)

    constant_features = [
        row["feature"] for row in numeric_rows if bool(row["constant_or_near_constant"])
    ]
    weak_signal_features = [
        row["feature"]
        for row in numeric_rows
        if row["spearman_to_target"] is not None and abs(float(row["spearman_to_target"])) < 0.01
    ]

    report = {
        "label": label,
        "data_dir": str(data_dir),
        "splits": {
            "train": _split_summary(train),
            "validation": _split_summary(validation),
            "test": _split_summary(test),
        },
        "feature_counts": {
            "all_columns": int(len(train.columns)),
            "numeric_candidate_features": int(len(numeric_columns)),
            "constant_or_near_constant_count": int(len(constant_features)),
        },
        "constant_or_near_constant_features": constant_features,
        "near_zero_spearman_features": weak_signal_features,
        "categorical_generalization": categorical_rows,
    }

    numeric_frame = pd.DataFrame(numeric_rows)
    numeric_frame.insert(0, "cohort", label)
    category_frame = pd.DataFrame(categorical_rows)
    if not category_frame.empty:
        category_frame.insert(0, "cohort", label)

    return report, numeric_frame, category_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit feature signal and categorical generalization across LaDe partitions"
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        type=Path,
        required=True,
        help="Partition directory containing train/validation/test snapshots",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="Label for each --data-dir in the same order",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/lade_feature_signal_audit"),
    )
    args = parser.parse_args()

    if len(args.data_dir) != len(args.label):
        raise ValueError("--data-dir and --label must be provided equally often")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    numeric_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []

    for data_dir, label in zip(args.data_dir, args.label, strict=True):
        report, numeric_frame, category_frame = _audit(data_dir, label)
        reports.append(report)
        numeric_frames.append(numeric_frame)
        if not category_frame.empty:
            category_frames.append(category_frame)

    numeric_output = pd.concat(numeric_frames, ignore_index=True)
    numeric_output.to_csv(output_dir / "numeric_feature_signal.csv", index=False)

    if category_frames:
        category_output = pd.concat(category_frames, ignore_index=True)
    else:
        category_output = pd.DataFrame()
    category_output.to_csv(
        output_dir / "categorical_generalization.csv",
        index=False,
    )

    output = {
        "status": "PASS",
        "purpose": (
            "feature-variance, train-only signal, and categorical "
            "generalization audit before new ETA challenger design"
        ),
        "cohorts": reports,
        "outputs": {
            "numeric_feature_signal_csv": str(output_dir / "numeric_feature_signal.csv"),
            "categorical_generalization_csv": str(output_dir / "categorical_generalization.csv"),
        },
    }
    (output_dir / "feature_signal_audit.json").write_text(
        json.dumps(output, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, default=_json_default))
    print("FEATURE_SIGNAL_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
