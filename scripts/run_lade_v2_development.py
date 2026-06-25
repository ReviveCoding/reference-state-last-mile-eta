from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from reference_eta.data.lade import (
    _density,
    _haversine_km,
    _local_xy_km,
    _nearest_neighbor_path_distance_km,
    _path_distance_km,
    _service_minutes,
    load_lade_delivery_csv,
)
from reference_eta.data.split import grouped_temporal_split
from reference_eta.decisions.triage import fit_tail_thresholds, threshold_for_rows
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import point_metrics

TARGET = "target_route_remaining_minutes"

BASE_STABLE_FEATURES = [
    "query_hour",
    "elapsed_minutes",
    "completed_task_count",
    "remaining_task_count",
    "initial_task_count",
    "completed_workload",
    "remaining_workload",
    "initial_workload",
    "observed_progress",
    "route_phase",
    "recent_pace",
    "task_density",
    "remaining_spread",
    "aoi_transition_burden",
]

V2_EVENT_FEATURES = [
    "completed_delivery_gap_last_minutes",
    "completed_delivery_gap_median_minutes",
    "completed_delivery_gap_p90_minutes",
    "visible_task_age_mean_minutes",
    "visible_task_age_p90_minutes",
    "visible_task_age_max_minutes",
    "visible_accept_span_minutes",
    "visible_accept_count_30m",
    "visible_accept_count_60m",
    "route_accept_count_30m",
    "route_accept_count_60m",
    "route_completion_count_30m",
    "route_completion_count_60m",
    "current_to_pending_min_km",
    "current_to_pending_mean_km",
    "current_to_pending_p90_km",
    "current_to_pending_max_km",
    "current_to_pending_centroid_km",
    "pending_unique_aoi_count",
    "pending_aoi_entropy",
    "pending_same_aoi_fraction",
    "pending_unique_aoi_type_count",
]


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _entropy(values: pd.Series) -> float:
    counts = values.astype(str).value_counts(dropna=False).to_numpy(dtype=float)
    if len(counts) == 0:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())


def _percentile(values: np.ndarray, q: float, fallback: float = 0.0) -> float:
    if len(values) == 0:
        return fallback
    return float(np.percentile(values.astype(float), q))


def _build_v2_snapshots(
    deliveries: pd.DataFrame,
    *,
    snapshot_stride: int,
    default_service_minutes: float,
    assumed_speed_kmh: float,
) -> pd.DataFrame:
    frame = deliveries.copy()
    frame["accept_time"] = pd.to_datetime(frame["accept_time"], errors="raise")
    frame["delivery_time"] = pd.to_datetime(frame["delivery_time"], errors="raise")
    rows: list[dict[str, object]] = []
    counter = 0

    for (courier_id, work_date), group in frame.groupby(["courier_id", "work_date"], sort=True):
        group = group.sort_values("delivery_time", kind="stable").reset_index(drop=True)
        if len(group) < 4:
            continue
        day_start = min(group["accept_time"].min(), group["delivery_time"].min())

        for completed_index in range(0, len(group) - 1, snapshot_stride):
            query_time = group.iloc[completed_index]["delivery_time"]
            completed = (
                group.loc[group["delivery_time"] <= query_time].sort_values("delivery_time").copy()
            )
            visible = group.loc[
                (group["accept_time"] <= query_time) & (group["delivery_time"] > query_time)
            ].copy()
            if visible.empty or completed.empty:
                continue

            current = completed.iloc[-1][["latitude", "longitude"]].to_numpy(dtype=float)
            completed_coords = completed[["latitude", "longitude"]].to_numpy(dtype=float)
            visible_coords = visible[["latitude", "longitude"]].to_numpy(dtype=float)
            completed_services = _service_minutes(completed, default_service_minutes)
            visible_services = _service_minutes(visible, default_service_minutes)

            completed_travel_minutes = (
                _path_distance_km(completed_coords) / assumed_speed_kmh * 60.0
            )
            remaining_travel_minutes = (
                _nearest_neighbor_path_distance_km(current, visible_coords)
                / assumed_speed_kmh
                * 60.0
            )
            completed_workload = float(completed_travel_minutes + completed_services.sum())
            remaining_workload = float(remaining_travel_minutes + visible_services.sum())
            initial_workload = completed_workload + remaining_workload
            completed_count = int(len(completed))
            total_count = int(completed_count + len(visible))
            observed_progress = completed_workload / max(initial_workload, 1e-6)
            elapsed_minutes = float((query_time - day_start).total_seconds() / 60.0)

            recent = completed.tail(min(3, len(completed)))
            if len(recent) >= 2:
                recent_coords = recent[["latitude", "longitude"]].to_numpy(dtype=float)
                recent_services = _service_minutes(recent.iloc[1:], default_service_minutes)
                recent_workload = (
                    _path_distance_km(recent_coords) / assumed_speed_kmh * 60.0
                    + recent_services.sum()
                )
                recent_elapsed = (
                    recent["delivery_time"].iloc[-1] - recent["delivery_time"].iloc[0]
                ).total_seconds() / 60.0
                recent_pace = float(recent_workload / max(recent_elapsed, 1e-3))
            else:
                recent_pace = float(completed_workload / max(elapsed_minutes, 1.0))

            local_visible = np.vstack([_local_xy_km(coord, current) for coord in visible_coords])
            remaining_spread = float(
                np.mean(np.linalg.norm(local_visible - local_visible.mean(axis=0), axis=1))
            )
            task_density = _density(visible_coords)
            city = str(group.iloc[0]["city"])
            query_hour = query_time.hour + query_time.minute / 60.0

            if "aoi_id" in completed.columns and len(completed) >= 2:
                completed_aoi = completed["aoi_id"].astype(str).to_numpy()
                transitions = int(np.sum(completed_aoi[1:] != completed_aoi[:-1]))
                aoi_transition_burden = float(transitions / max(len(completed_aoi) - 1, 1))
                current_aoi = str(completed.iloc[-1]["aoi_id"])
            else:
                aoi_transition_burden = 0.0
                current_aoi = ""

            completed_delivery_times = (
                completed["delivery_time"].sort_values().to_numpy(dtype="datetime64[ns]")
            )
            if len(completed_delivery_times) >= 2:
                completed_gaps = (
                    np.diff(completed_delivery_times).astype("timedelta64[s]").astype(float) / 60.0
                )
            else:
                completed_gaps = np.array([], dtype=float)

            visible_ages = (
                (query_time - visible["accept_time"])
                .dt.total_seconds()
                .div(60.0)
                .clip(lower=0.0)
                .to_numpy(dtype=float)
            )
            visible_accept_span = float(
                (visible["accept_time"].max() - visible["accept_time"].min()).total_seconds() / 60.0
            )
            route_accept_30 = int(
                (
                    (group["accept_time"] <= query_time)
                    & (group["accept_time"] > query_time - pd.Timedelta(minutes=30))
                ).sum()
            )
            route_accept_60 = int(
                (
                    (group["accept_time"] <= query_time)
                    & (group["accept_time"] > query_time - pd.Timedelta(minutes=60))
                ).sum()
            )
            route_done_30 = int(
                (
                    (group["delivery_time"] <= query_time)
                    & (group["delivery_time"] > query_time - pd.Timedelta(minutes=30))
                ).sum()
            )
            route_done_60 = int(
                (
                    (group["delivery_time"] <= query_time)
                    & (group["delivery_time"] > query_time - pd.Timedelta(minutes=60))
                ).sum()
            )
            visible_accept_30 = int(
                (visible["accept_time"] > query_time - pd.Timedelta(minutes=30)).sum()
            )
            visible_accept_60 = int(
                (visible["accept_time"] > query_time - pd.Timedelta(minutes=60)).sum()
            )

            pending_distances = np.asarray(
                [_haversine_km(current, coord) for coord in visible_coords], dtype=float
            )
            centroid = visible_coords.mean(axis=0)
            centroid_distance = _haversine_km(current, centroid)
            pending_aoi = visible.get(
                "aoi_id", pd.Series(["" for _ in range(len(visible))])
            ).astype(str)
            pending_aoi_type = visible.get(
                "aoi_type", pd.Series(["" for _ in range(len(visible))])
            ).astype(str)
            pending_same_aoi = float((pending_aoi == current_aoi).mean()) if current_aoi else 0.0

            snapshot_id = f"LADE-V2-{counter:08d}"
            counter += 1
            next_finish = visible["delivery_time"].min()
            route_finish = visible["delivery_time"].max()

            rows.append(
                {
                    "snapshot_id": snapshot_id,
                    "courier_id": str(courier_id),
                    "work_date": str(work_date),
                    "query_time": query_time.isoformat(),
                    "city": city,
                    "query_hour": query_hour,
                    "elapsed_minutes": elapsed_minutes,
                    "completed_task_count": completed_count,
                    "remaining_task_count": len(visible),
                    "initial_task_count": total_count,
                    "completed_workload": completed_workload,
                    "remaining_workload": remaining_workload,
                    "initial_workload": initial_workload,
                    "observed_progress": observed_progress,
                    "route_phase": completed_count / max(total_count, 1),
                    "recent_pace": recent_pace,
                    "task_density": task_density,
                    "remaining_spread": remaining_spread,
                    "aoi_transition_burden": aoi_transition_burden,
                    "completed_delivery_gap_last_minutes": float(completed_gaps[-1])
                    if len(completed_gaps)
                    else 0.0,
                    "completed_delivery_gap_median_minutes": _percentile(completed_gaps, 50.0),
                    "completed_delivery_gap_p90_minutes": _percentile(completed_gaps, 90.0),
                    "visible_task_age_mean_minutes": float(visible_ages.mean()),
                    "visible_task_age_p90_minutes": _percentile(visible_ages, 90.0),
                    "visible_task_age_max_minutes": float(visible_ages.max()),
                    "visible_accept_span_minutes": visible_accept_span,
                    "visible_accept_count_30m": visible_accept_30,
                    "visible_accept_count_60m": visible_accept_60,
                    "route_accept_count_30m": route_accept_30,
                    "route_accept_count_60m": route_accept_60,
                    "route_completion_count_30m": route_done_30,
                    "route_completion_count_60m": route_done_60,
                    "current_to_pending_min_km": float(pending_distances.min()),
                    "current_to_pending_mean_km": float(pending_distances.mean()),
                    "current_to_pending_p90_km": _percentile(pending_distances, 90.0),
                    "current_to_pending_max_km": float(pending_distances.max()),
                    "current_to_pending_centroid_km": float(centroid_distance),
                    "pending_unique_aoi_count": int(pending_aoi.nunique(dropna=False)),
                    "pending_aoi_entropy": _entropy(pending_aoi),
                    "pending_same_aoi_fraction": pending_same_aoi,
                    "pending_unique_aoi_type_count": int(pending_aoi_type.nunique(dropna=False)),
                    "target_next_minutes": float((next_finish - query_time).total_seconds() / 60.0),
                    "target_route_remaining_minutes": float(
                        (route_finish - query_time).total_seconds() / 60.0
                    ),
                }
            )

    if not rows:
        raise RuntimeError("No V2 snapshots could be constructed")
    return (
        pd.DataFrame(rows)
        .sort_values(["work_date", "query_time"], kind="stable")
        .reset_index(drop=True)
    )


def _encode(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    source = frame[features + ["city"]].copy()
    source[features] = source[features].astype(float)
    return pd.get_dummies(source, columns=["city"], dtype=float)


def _fit_predict(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    *,
    features: list[str],
    train_columns: list[str] | None = None,
) -> tuple[LGBMRegressor, np.ndarray, list[str]]:
    X_train = _encode(train, features)
    columns = list(X_train.columns) if train_columns is None else train_columns
    X_train = X_train.reindex(columns=columns, fill_value=0.0)
    X_eval = _encode(evaluate, features).reindex(columns=columns, fill_value=0.0)
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=30,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(X_train, train[TARGET].to_numpy(dtype=float))
    return model, np.maximum(model.predict(X_eval), 0.0), columns


def _metric_row(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = point_metrics(y, prediction)
    return {key: float(value) for key, value in values.items()}


def _tail_mae(y: np.ndarray, prediction: np.ndarray, thresholds: np.ndarray) -> float:
    mask = y > thresholds
    return float(point_metrics(y[mask], prediction[mask])["mae"]) if mask.any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed V2 LightGBM feature development benchmark."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--snapshot-stride", type=int, default=3)
    parser.add_argument("--default-service-minutes", type=float, default=5.0)
    parser.add_argument("--assumed-speed-kmh", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    artifact_dir = args.artifact_dir.resolve()
    report_dir = args.report_dir.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    deliveries = load_lade_delivery_csv(input_csv)
    snapshots = _build_v2_snapshots(
        deliveries,
        snapshot_stride=int(args.snapshot_stride),
        default_service_minutes=float(args.default_service_minutes),
        assumed_speed_kmh=float(args.assumed_speed_kmh),
    )
    snapshots.to_csv(data_dir / "snapshots_v2_raw.csv", index=False)

    split = grouped_temporal_split(
        snapshots,
        train_fraction=0.60,
        validation_fraction=0.15,
        calibration_fraction=0.10,
        test_fraction=0.15,
    )
    partitions = {
        "train": split.train.reset_index(drop=True),
        "validation": split.validation.reset_index(drop=True),
        "calibration": split.calibration.reset_index(drop=True),
        "test": split.test.reset_index(drop=True),
    }
    for name, frame in partitions.items():
        frame.to_csv(data_dir / f"{name}_snapshots.csv", index=False)

    train = partitions["train"]
    validation = partitions["validation"]
    calibration = partitions["calibration"]
    test = partitions["test"]

    _, baseline_validation, baseline_columns = _fit_predict(
        train, validation, features=BASE_STABLE_FEATURES
    )
    v2_features = BASE_STABLE_FEATURES + V2_EVENT_FEATURES
    _, v2_validation, v2_columns = _fit_predict(train, validation, features=v2_features)

    validation_y = validation[TARGET].to_numpy(dtype=float)
    thresholds = fit_tail_thresholds(train, TARGET)
    validation_thresholds = threshold_for_rows(validation, thresholds)
    validation_base_metrics = _metric_row(validation_y, baseline_validation)
    validation_v2_metrics = _metric_row(validation_y, v2_validation)
    validation_baseline_tail = _tail_mae(validation_y, baseline_validation, validation_thresholds)
    validation_v2_tail = _tail_mae(validation_y, v2_validation, validation_thresholds)

    # Refit each pre-specified model on train+validation before calibration/test.  Validation remains
    # the model-selection record; no candidate sweep is performed in this development cohort.
    train_dev = pd.concat([train, validation], ignore_index=True)
    baseline_model, baseline_calibration, baseline_columns = _fit_predict(
        train_dev, calibration, features=BASE_STABLE_FEATURES
    )
    v2_model, v2_calibration, v2_columns = _fit_predict(
        train_dev, calibration, features=v2_features
    )
    baseline_offset = float(
        np.median(calibration[TARGET].to_numpy(dtype=float) - baseline_calibration)
    )
    v2_offset = float(np.median(calibration[TARGET].to_numpy(dtype=float) - v2_calibration))

    baseline_test_matrix = _encode(test, BASE_STABLE_FEATURES).reindex(
        columns=baseline_columns, fill_value=0.0
    )
    v2_test_matrix = _encode(test, v2_features).reindex(columns=v2_columns, fill_value=0.0)
    baseline_test_raw = np.maximum(baseline_model.predict(baseline_test_matrix), 0.0)
    v2_test_raw = np.maximum(v2_model.predict(v2_test_matrix), 0.0)
    baseline_test = np.maximum(baseline_test_raw + baseline_offset, 0.0)
    v2_test = np.maximum(v2_test_raw + v2_offset, 0.0)

    y_test = test[TARGET].to_numpy(dtype=float)
    test_thresholds = threshold_for_rows(test, thresholds)
    baseline_metrics = _metric_row(y_test, baseline_test)
    v2_metrics = _metric_row(y_test, v2_test)
    baseline_tail = _tail_mae(y_test, baseline_test, test_thresholds)
    v2_tail = _tail_mae(y_test, v2_test, test_thresholds)
    bootstrap = clustered_mae_difference_ci(
        test,
        y_test,
        baseline_test,
        v2_test,
        seed=int(args.seed),
    )

    validation_relative = float(
        (validation_base_metrics["mae"] - validation_v2_metrics["mae"])
        / max(validation_base_metrics["mae"], 1e-9)
    )
    test_relative = float(
        (baseline_metrics["mae"] - v2_metrics["mae"]) / max(baseline_metrics["mae"], 1e-9)
    )
    validation_tail_guardrail = bool(validation_v2_tail <= validation_baseline_tail * 1.02)
    development_promising = bool(validation_relative > 0.0 and validation_tail_guardrail)

    scorecard = pd.DataFrame(
        [
            {
                "model": "lightgbm_v1_clean_baseline",
                "variant": "median_residual_calibrated",
                **baseline_metrics,
                "tail_mae": baseline_tail,
            },
            {
                "model": "lightgbm_v2_event_state",
                "variant": "median_residual_calibrated",
                **v2_metrics,
                "tail_mae": v2_tail,
            },
        ]
    )
    scorecard.to_csv(report_dir / "v2_development_scorecard.csv", index=False)

    validation_scorecard = pd.DataFrame(
        [
            {
                "model": "lightgbm_v1_clean_baseline",
                **validation_base_metrics,
                "tail_mae": validation_baseline_tail,
            },
            {
                "model": "lightgbm_v2_event_state",
                **validation_v2_metrics,
                "tail_mae": validation_v2_tail,
            },
        ]
    )
    validation_scorecard.to_csv(report_dir / "v2_development_validation_scorecard.csv", index=False)

    feature_importance = pd.DataFrame(
        {"feature": v2_columns, "importance": v2_model.feature_importances_}
    ).sort_values("importance", ascending=False, kind="stable")
    feature_importance.to_csv(report_dir / "v2_event_feature_importance.csv", index=False)

    prediction_frame = test[
        ["snapshot_id", "courier_id", "work_date", "query_time", "city", TARGET]
    ].copy()
    prediction_frame["baseline_prediction"] = baseline_test
    prediction_frame["v2_prediction"] = v2_test
    prediction_frame["baseline_absolute_error"] = np.abs(y_test - baseline_test)
    prediction_frame["v2_absolute_error"] = np.abs(y_test - v2_test)
    prediction_frame.to_csv(report_dir / "v2_development_test_predictions.csv", index=False)

    summary = {
        "experiment": "pre_specified_v2_event_state_development",
        "status": "DEVELOPMENT_PROMISING" if development_promising else "DEVELOPMENT_HOLD",
        "claim_supported": False,
        "claim_boundary": "development-only cohort; final claim requires a new untouched whole-courier-day confirmation cohort",
        "model_specification": {
            "baseline_features": BASE_STABLE_FEATURES + ["city_one_hot"],
            "v2_added_features": V2_EVENT_FEATURES,
            "explicitly_excluded": [
                "weather_severity",
                "congestion_proxy",
                "trajectory_missingness",
                "courier_id",
                "rcot feature block",
            ],
            "lightgbm": {
                "objective": "regression_l1",
                "n_estimators": 400,
                "learning_rate": 0.04,
                "num_leaves": 48,
                "min_child_samples": 30,
                "random_state": 42,
                "n_jobs": 1,
            },
            "selection": "pre-specified V2 feature suite; no candidate sweep; validation is development evidence only",
            "point_calibration": "median residual fit on dedicated calibration partition for both models",
        },
        "rows": {name: int(len(frame)) for name, frame in partitions.items()},
        "snapshots": int(len(snapshots)),
        "validation": {
            "baseline": {**validation_base_metrics, "tail_mae": validation_baseline_tail},
            "v2": {**validation_v2_metrics, "tail_mae": validation_v2_tail},
            "relative_mae_improvement": validation_relative,
            "tail_guardrail_pass": validation_tail_guardrail,
        },
        "development_test": {
            "baseline": {**baseline_metrics, "tail_mae": baseline_tail},
            "v2": {**v2_metrics, "tail_mae": v2_tail},
            "relative_mae_improvement": test_relative,
            "bootstrap": bootstrap,
            "tail_guardrail_pass": bool(v2_tail <= baseline_tail * 1.02),
        },
    }
    (report_dir / "v2_development_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    print("V2_DEVELOPMENT_BENCHMARK_COMPLETE")


if __name__ == "__main__":
    main()
