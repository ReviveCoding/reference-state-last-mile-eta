from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reference_eta import __version__
from reference_eta.config import load_config
from reference_eta.data.lade import build_closed_set_snapshots, load_lade_delivery_csv
from reference_eta.data.split import grouped_temporal_split
from reference_eta.data.synthetic import generate_synthetic_delivery_data
from reference_eta.decisions.sensitivity import intervention_sensitivity
from reference_eta.decisions.triage import (
    capacity_metrics,
    derive_tail_scores,
    fit_tail_thresholds,
    threshold_for_rows,
)
from reference_eta.evaluation.bootstrap import clustered_mae_difference_ci
from reference_eta.evaluation.metrics import interval_metrics, point_metrics, slice_metrics
from reference_eta.evaluation.release import evaluate_rcot_promotion, evaluate_system_release
from reference_eta.features.rcot import ReferenceOperationalTimeTransformer
from reference_eta.io import atomic_copy, atomic_write_json
from reference_eta.locking import ExclusiveFileLock
from reference_eta.models.baselines import (
    CohortMedianRegressor,
    LightGBMConfig,
    LightGBMPointModel,
    QuantileLightGBMModel,
)
from reference_eta.models.calibration import ConformalQuantileCalibrator
from reference_eta.models.rolling_calibration import RollingConformalReplay
from reference_eta.monitoring.drift import build_drift_report
from reference_eta.provenance import build_run_provenance
from reference_eta.reporting.report import write_markdown_report

ROOT = Path(__file__).resolve().parents[1]
_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _validated_output_namespace(value: str | None) -> str | None:
    if value is None:
        return None
    if not _NAMESPACE_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            "output_namespace must be a 1-64 character identifier using letters, digits, '.', '_' or '-'"
        )
    return value


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serving_provenance_is_compatible(path: Path) -> bool:
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(provenance, dict):
        return False
    if int(provenance.get("provenance_schema_version", -1)) != 2:
        return False
    project = provenance.get("project")
    return (
        isinstance(project, dict)
        and project.get("name") == "reference-state-last-mile-eta"
        and project.get("version") == __version__
    )


def _serving_bundle_is_complete(bundle_dir: Path, required_names: tuple[str, ...]) -> bool:
    manifest_path = bundle_dir / "artifact_manifest.json"
    if bundle_dir.is_symlink() or not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(records, list) or len(records) != len(required_names):
        return False
    by_name: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not {"path", "size_bytes", "sha256"}.issubset(record):
            return False
        name = str(record["path"])
        if Path(name).name != name or name in by_name:
            return False
        by_name[name] = record
    if set(by_name) != set(required_names):
        return False
    for name, record in by_name.items():
        path = bundle_dir / name
        if not path.is_file() or path.is_symlink():
            return False
        if path.stat().st_size != int(record["size_bytes"]):
            return False
        if _sha256(path) != str(record["sha256"]):
            return False
    return _serving_provenance_is_compatible(bundle_dir / "run_provenance.json")


def _publish_serving_bundle_unlocked(
    artifacts: Path, *, retain: int = 3, fault_stage: str | None = None
) -> dict[str, object]:
    """Publish a complete immutable serving bundle and atomically switch the pointer."""

    required_names = (
        "rcot.joblib",
        "quantile_champion.joblib",
        "cqr_calibrator.joblib",
        "tail_thresholds.joblib",
        "release_decision.json",
        "run_provenance.json",
    )
    sources = [artifacts / name for name in required_names]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot publish serving bundle; missing files: {missing}")
    if not _serving_provenance_is_compatible(artifacts / "run_provenance.json"):
        raise ValueError("Cannot publish serving bundle with incompatible run provenance")

    source_records = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sources
    ]
    bundle_payload = json.dumps(source_records, sort_keys=True, separators=(",", ":"))
    bundle_id = hashlib.sha256(bundle_payload.encode("utf-8")).hexdigest()[:20]
    bundles_root = artifacts / "serving_bundles"
    bundles_root.mkdir(parents=True, exist_ok=True)
    final_dir = bundles_root / bundle_id
    if final_dir.exists() and not _serving_bundle_is_complete(final_dir, required_names):
        shutil.rmtree(final_dir, ignore_errors=True)
    if not final_dir.exists():
        temporary_dir = bundles_root / f".{bundle_id}.{os.getpid()}.tmp"
        shutil.rmtree(temporary_dir, ignore_errors=True)
        temporary_dir.mkdir(parents=True)
        try:
            for source in sources:
                atomic_copy(source, temporary_dir / source.name)
            atomic_write_json(temporary_dir / "artifact_manifest.json", source_records)
            try:
                os.replace(temporary_dir, final_dir)
            except OSError:
                # A concurrent publisher may have completed the identical content-addressed
                # bundle first. Accept it only if it now exists; otherwise propagate the error.
                if not final_dir.is_dir():
                    raise
                shutil.rmtree(temporary_dir, ignore_errors=True)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    manifest_path = final_dir / "artifact_manifest.json"
    if fault_stage == "after_bundle_before_pointer":
        raise RuntimeError("Injected publish failure after bundle completion")
    pointer = {
        "artifact_schema_version": 1,
        "bundle_id": bundle_id,
        "manifest_sha256": _sha256(manifest_path),
    }
    atomic_write_json(artifacts / "current_bundle.json", pointer)

    compatible_bundles: list[Path] = []
    for candidate in bundles_root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        if not _serving_bundle_is_complete(candidate, required_names):
            shutil.rmtree(candidate, ignore_errors=True)
            continue
        compatible_bundles.append(candidate)
    compatible_bundles.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for stale in compatible_bundles[max(int(retain), 1) :]:
        if stale.name != bundle_id:
            shutil.rmtree(stale, ignore_errors=True)
    return pointer


def _publish_serving_bundle(
    artifacts: Path,
    *,
    retain: int = 3,
    fault_stage: str | None = None,
) -> dict[str, object]:
    lock = ExclusiveFileLock(
        artifacts / ".locks" / "publish.lock",
        timeout_seconds=float(os.getenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "5")),
        stale_after_seconds=float(os.getenv("REFERENCE_ETA_LOCK_STALE_SECONDS", "3600")),
        purpose="serving bundle publication",
    )
    with lock:
        return _publish_serving_bundle_unlocked(artifacts, retain=retain, fault_stage=fault_stage)


def _subset_tasks(tasks: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    return tasks[tasks["snapshot_id"].isin(set(snapshots["snapshot_id"]))].copy()


def _clean_run_outputs(artifacts: Path, reports: Path, *, namespaced: bool) -> None:
    """Remove stale outputs without deleting sibling run namespaces."""

    if namespaced:
        shutil.rmtree(artifacts, ignore_errors=True)
        shutil.rmtree(reports, ignore_errors=True)
        return
    for directory in (artifacts, reports):
        directory.mkdir(parents=True, exist_ok=True)
        for child in directory.iterdir():
            if directory == artifacts and child.name in {
                "current_bundle.json",
                "serving_bundles",
                "baseline_source_fingerprint.json",
                "baseline_candidate_metrics.json",
            }:
                continue
            if directory == reports and child.name in {
                "final_release_log.txt",
                "coverage.json",
            }:
                continue
            if child.is_file() or child.name == "data":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()


def _run_manifest_files(
    artifacts: Path,
    reports: Path,
    data_dir: Path,
    *,
    namespaced: bool,
) -> list[Path]:
    if namespaced:
        candidates = [*artifacts.rglob("*"), *reports.rglob("*")]
    else:
        candidates = [
            *[path for path in artifacts.iterdir() if path.is_file()],
            *data_dir.rglob("*"),
            *[path for path in reports.iterdir() if path.is_file()],
        ]
    excluded_names = {"artifact_manifest.json", "final_release_log.txt", "coverage.json"}
    return sorted(path for path in candidates if path.is_file() and path.name not in excluded_names)


def _write_sqlite_outputs(
    db_path: Path,
    snapshots: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        snapshots.to_sql("snapshots", connection, index=False, if_exists="replace")
        predictions.to_sql("predictions", connection, index=False, if_exists="replace")
        sql_files = sorted((ROOT / "sql").glob("*.sql"))
        if not sql_files:
            raise RuntimeError("No SQL artifacts were found")
        for sql_file in sql_files:
            connection.executescript(sql_file.read_text(encoding="utf-8"))
        expected_tables = {
            "snapshots",
            "predictions",
            "snapshot_mart",
            "prediction_reconciliation",
            "monitoring_aggregates",
            "capacity_report",
            "reference_quality",
        }
        existing = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing = expected_tables.difference(existing)
        if missing:
            raise RuntimeError(f"SQL pipeline did not create tables: {sorted(missing)}")
        for table in expected_tables:
            count = int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            if count == 0:
                raise RuntimeError(f"SQL pipeline created an empty required table: {table}")
        connection.commit()
    finally:
        connection.close()


def _run_unlocked(
    config_path: Path, mode: str, output_namespace: str | None = None
) -> dict[str, object]:
    output_namespace = _validated_output_namespace(output_namespace)
    config = load_config(config_path)
    seed = int(config["seed"])
    np.random.seed(seed)
    artifacts_root = ROOT / "artifacts"
    reports_root = ROOT / "reports"
    artifacts = artifacts_root / output_namespace if output_namespace else artifacts_root
    reports = reports_root / output_namespace if output_namespace else reports_root
    data_dir = artifacts / "data"
    _clean_run_outputs(artifacts, reports, namespaced=output_namespace is not None)
    for directory in (artifacts, reports, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    provenance_data_path = (
        (ROOT / str(data_cfg["path"])).resolve()
        if data_cfg["source"] == "lade_normalized"
        else None
    )
    provenance = build_run_provenance(
        root=ROOT,
        config_path=Path(config_path),
        seed=seed,
        data_source=str(data_cfg["source"]),
        data_path=provenance_data_path,
        deterministic_requested=bool(config["advanced"]["deterministic"]),
    )
    atomic_write_json(artifacts / "run_provenance.json", provenance)

    if data_cfg["source"] == "synthetic":
        generated = generate_synthetic_delivery_data(
            n_courier_days=int(data_cfg["n_courier_days"]),
            start_date=str(data_cfg["start_date"]),
            cities=data_cfg["cities"],
            min_tasks=int(data_cfg["min_tasks"]),
            max_tasks=int(data_cfg["max_tasks"]),
            snapshot_stride=int(data_cfg["snapshot_stride"]),
            seed=seed,
        )
    elif data_cfg["source"] == "lade_normalized":
        source_path = Path(data_cfg["path"])
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        generated = build_closed_set_snapshots(
            load_lade_delivery_csv(source_path),
            snapshot_stride=int(data_cfg.get("snapshot_stride", 3)),
            default_service_minutes=float(data_cfg.get("default_service_minutes", 5.0)),
            assumed_speed_kmh=float(data_cfg.get("assumed_speed_kmh", 15.0)),
        )
    else:
        raise ValueError(f"Unsupported data source: {data_cfg['source']}")
    generated.snapshots.to_csv(data_dir / "snapshots_raw.csv", index=False)
    generated.pending_tasks.to_csv(data_dir / "pending_tasks.csv", index=False)

    split_cfg = config["split"]
    split = grouped_temporal_split(
        generated.snapshots,
        train_fraction=float(split_cfg["train"]),
        validation_fraction=float(split_cfg["validation"]),
        calibration_fraction=float(split_cfg["calibration"]),
        test_fraction=float(split_cfg["test"]),
    )

    rcot_cfg = config["rcot"]
    rcot = ReferenceOperationalTimeTransformer(
        min_cohort_rows=int(rcot_cfg["min_cohort_rows"]),
        min_cohort_groups=int(rcot_cfg["min_cohort_groups"]),
        support_shrinkage=float(rcot_cfg["support_shrinkage"]),
        max_dispersion_minutes=float(rcot_cfg["max_dispersion_minutes"]),
    )
    train = rcot.fit_transform_cross_fitted(
        split.train, n_splits=int(rcot_cfg["cross_fit_splits"])
    ).reset_index(drop=True)
    validation = rcot.transform(split.validation).reset_index(drop=True)
    calibration = rcot.transform(split.calibration).reset_index(drop=True)
    test = rcot.transform(split.test).reset_index(drop=True)
    rcot.save(str(artifacts / "rcot.joblib"))

    for name, frame in {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test": test,
    }.items():
        frame.to_csv(data_dir / f"{name}_snapshots.csv", index=False)

    target = "target_route_remaining_minutes"
    model_cfg = LightGBMConfig(
        n_estimators=int(config["model"]["n_estimators"]),
        learning_rate=float(config["model"]["learning_rate"]),
        num_leaves=int(config["model"]["num_leaves"]),
        min_child_samples=int(config["model"]["min_child_samples"]),
        random_state=int(config["model"]["random_state"]),
    )

    cohort = CohortMedianRegressor().fit(train, target)
    point_base = LightGBMPointModel(model_cfg, include_rcot=False).fit(train, target)
    point_rcot = LightGBMPointModel(model_cfg, include_rcot=True).fit(train, target)
    quantile_base = QuantileLightGBMModel(model_cfg, include_rcot=False).fit(train, target)
    quantile_rcot = QuantileLightGBMModel(model_cfg, include_rcot=True).fit(train, target)
    point_base.save(str(artifacts / "point_without_rcot.joblib"))
    point_rcot.save(str(artifacts / "point_with_rcot.joblib"))
    quantile_base.save(str(artifacts / "quantile_without_rcot.joblib"))
    quantile_rcot.save(str(artifacts / "quantile_with_rcot.joblib"))

    validation_y = validation[target].to_numpy(dtype=float)
    validation_quantile_base = quantile_base.predict(validation).reset_index(drop=True)
    validation_quantile_rcot = quantile_rcot.predict(validation).reset_index(drop=True)
    validation_mae_base = point_metrics(validation_y, validation_quantile_base["q50"].to_numpy())[
        "mae"
    ]
    validation_mae_rcot = point_metrics(validation_y, validation_quantile_rcot["q50"].to_numpy())[
        "mae"
    ]
    thresholds = fit_tail_thresholds(train, target)
    joblib.dump(thresholds, artifacts / "tail_thresholds.joblib")
    validation_thresholds = threshold_for_rows(validation, thresholds)
    promotion = evaluate_rcot_promotion(
        validation,
        validation_y,
        validation_quantile_base["q50"].to_numpy(),
        validation_quantile_rcot["q50"].to_numpy(),
        validation_thresholds,
    )
    rcot_promoted = promotion.promote
    quantile_champion = quantile_rcot if rcot_promoted else quantile_base
    quantile_champion.save(str(artifacts / "quantile_champion.joblib"))

    y_test = test[target].to_numpy(dtype=float)
    cohort_pred = cohort.predict(test)
    base_pred = point_base.predict(test)
    rcot_pred = point_rcot.predict(test)
    calibration_raw = quantile_champion.predict(calibration).reset_index(drop=True)
    test_raw = quantile_champion.predict(test).reset_index(drop=True)
    calibrator = ConformalQuantileCalibrator(
        target_coverage=float(config["calibration"]["target_coverage"])
    ).fit(calibration_raw, calibration[target].to_numpy(dtype=float))
    test_quantiles = calibrator.transform(test_raw)
    calibrator.save(str(artifacts / "cqr_calibrator.joblib"))
    calibration_y = calibration[target].to_numpy(dtype=float)
    initial_nonconformity = np.maximum(
        calibration_raw["q10"].to_numpy() - calibration_y,
        calibration_y - calibration_raw["q90"].to_numpy(),
    )
    test_prediction_times = pd.to_datetime(test["query_time"], errors="raise")
    test_label_available_times = test_prediction_times + pd.to_timedelta(
        test[target].to_numpy(dtype=float), unit="m"
    )
    rolling_replay = RollingConformalReplay(
        target_coverage=float(config["calibration"]["target_coverage"]),
        window_size=min(200, max(len(initial_nonconformity), 25)),
    ).replay(
        initial_nonconformity,
        test_raw,
        test[target].to_numpy(dtype=float),
        prediction_times=test_prediction_times,
        label_available_times=test_label_available_times,
    )
    rolling_replay.to_csv(reports / "rolling_calibration_replay.csv", index=False)

    test_quantile_base = quantile_base.predict(test).reset_index(drop=True)
    test_quantile_rcot = quantile_rcot.predict(test).reset_index(drop=True)
    metric_rows = []
    for name, prediction in {
        "conditional_median": cohort_pred,
        "lightgbm_without_rcot": base_pred,
        "lightgbm_with_rcot": rcot_pred,
        "quantile_without_rcot_q50": test_quantile_base["q50"].to_numpy(),
        "quantile_with_rcot_q50": test_quantile_rcot["q50"].to_numpy(),
        "quantile_champion_q50_calibrated": test_quantiles["q50"].to_numpy(),
    }.items():
        metric_rows.append({"model": name, **point_metrics(y_test, prediction)})
    interval_result = interval_metrics(y_test, test_quantiles)
    metric_rows.append({"model": "quantile_interval", **interval_result})

    advanced_result: dict[str, object] | None = None
    if bool(config["advanced"].get("enabled", False)) and mode in {"smoke", "gpu", "all"}:
        try:
            from reference_eta.models.hsg_eta import (
                HSGConfig,
                load_hsg_checkpoint,
                predict_hsg_eta,
                train_hsg_eta,
            )
        except ModuleNotFoundError as error:
            if error.name == "torch":
                raise RuntimeError(
                    "Advanced HSG-ETA training requires the optional 'advanced' or 'gpu' extra"
                ) from error
            raise
        checkpoint = artifacts / "hsg_eta.pt"
        advanced_cfg = HSGConfig(
            hidden_dim=int(config["advanced"]["hidden_dim"]),
            max_tasks=int(config["advanced"]["max_tasks"]),
            deterministic=bool(config["advanced"]["deterministic"]),
        )
        train_tasks = _subset_tasks(generated.pending_tasks, train)
        validation_tasks = _subset_tasks(generated.pending_tasks, validation)
        advanced_result = train_hsg_eta(
            train,
            train_tasks,
            validation,
            validation_tasks,
            config=advanced_cfg,
            epochs=int(config["advanced"]["epochs"]),
            batch_size=int(config["advanced"]["batch_size"]),
            learning_rate=float(config["advanced"]["learning_rate"]),
            amp=bool(config["advanced"]["amp"]),
            seed=seed,
            output_path=checkpoint,
        )
        model, context_scaler, task_scaler, _ = load_hsg_checkpoint(checkpoint)
        hsg_predictions = predict_hsg_eta(
            model,
            test,
            _subset_tasks(generated.pending_tasks, test),
            context_scaler=context_scaler,
            task_scaler=task_scaler,
        )
        hsg_predictions.to_csv(reports / "hsg_test_predictions.csv", index=False)
        available_route = hsg_predictions["route_target_available"].astype(bool)
        route_accuracy = (
            float(hsg_predictions.loc[available_route, "route_top1_correct"].mean())
            if available_route.any()
            else float("nan")
        )
        metric_rows.append(
            {
                "model": "hsg_eta_q50",
                **point_metrics(y_test, hsg_predictions["q50"].to_numpy()),
                "route_top1_accuracy": route_accuracy,
                "route_target_coverage": float(available_route.mean()),
                "mean_route_entropy": float(hsg_predictions["route_entropy"].mean()),
                "mean_route_normalized_entropy": float(
                    hsg_predictions["route_normalized_entropy"].mean()
                ),
                "mean_route_top1_probability": float(
                    hsg_predictions["route_top1_probability"].mean()
                ),
            }
        )
        (reports / "gpu_training_profile.json").write_text(
            json.dumps(advanced_result, indent=2, default=_json_default), encoding="utf-8"
        )

    model_metrics = pd.DataFrame(metric_rows)
    model_metrics.to_csv(reports / "model_scorecard.csv", index=False)

    bootstrap = clustered_mae_difference_ci(
        test,
        y_test,
        base_pred,
        rcot_pred,
        seed=seed,
    )
    (reports / "rcot_bootstrap_ci.json").write_text(
        json.dumps(bootstrap, indent=2, default=_json_default), encoding="utf-8"
    )

    slices = []
    slices.extend(slice_metrics(test, y_test, rcot_pred, slice_column="city"))
    support_band = pd.cut(
        test["reference_support"],
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["low", "medium", "high"],
    )
    test_for_slice = test.assign(reference_support_band=support_band.astype(str))
    slices.extend(
        slice_metrics(test_for_slice, y_test, rcot_pred, slice_column="reference_support_band")
    )
    pd.DataFrame(slices).to_csv(reports / "slice_performance.csv", index=False)

    test_thresholds = threshold_for_rows(test, thresholds)
    trust = test["rcot_trust"].to_numpy(dtype=float)
    tail_scores = derive_tail_scores(
        test_quantiles,
        test_thresholds,
        trust,
    )
    selection_capacity = float(config["decision"]["selection_capacity"])
    capacities = sorted(
        set([float(value) for value in config["decision"]["capacities"]] + [selection_capacity])
    )
    rng = np.random.default_rng(seed)
    policy_scores = {
        "random": rng.random(len(test)),
        "q50_eta": test_quantiles["q50"].to_numpy(),
        "q90_eta": test_quantiles["q90"].to_numpy(),
        "tail_risk": tail_scores["tail_risk"].to_numpy(),
        "reliability_adjusted_priority": tail_scores["reliability_adjusted_priority"].to_numpy(),
    }
    validation_raw = quantile_champion.predict(validation).reset_index(drop=True)
    validation_tail_scores = derive_tail_scores(
        validation_raw,
        validation_thresholds,
        validation["rcot_trust"].to_numpy(dtype=float),
    )
    validation_policy_scores = {
        "q50_eta": validation_raw["q50"].to_numpy(),
        "q90_eta": validation_raw["q90"].to_numpy(),
        "tail_risk": validation_tail_scores["tail_risk"].to_numpy(),
        "reliability_adjusted_priority": validation_tail_scores[
            "reliability_adjusted_priority"
        ].to_numpy(),
    }
    validation_policy_capture = {}
    for policy_name, policy_score in validation_policy_scores.items():
        metric = capacity_metrics(
            validation_y,
            validation_thresholds,
            policy_score,
            [selection_capacity],
        )[0]
        validation_policy_capture[policy_name] = float(metric["excess_minutes_capture"])
    validation_tail_events = int((validation_y > validation_thresholds).sum())
    minimum_validation_tail_events = int(config["decision"]["minimum_validation_tail_events"])
    if validation_tail_events >= minimum_validation_tail_events:
        decision_champion = max(validation_policy_capture, key=validation_policy_capture.get)
        decision_selection_mode = "validation_selected"
    else:
        decision_champion = "tail_risk"
        decision_selection_mode = "prespecified_low_tail_support"
    capacity_rows = []
    for policy, scores in policy_scores.items():
        for row in capacity_metrics(y_test, test_thresholds, scores, capacities):
            capacity_rows.append({"policy": policy, **row})
    capacity_report = pd.DataFrame(capacity_rows)
    capacity_report.to_csv(reports / "capacity_triage.csv", index=False)
    sensitivity = intervention_sensitivity(
        y_true=y_test,
        thresholds=test_thresholds,
        policy_scores=policy_scores,
        capacity=selection_capacity,
        effectiveness_grid=[0.0, 0.10, 0.25, 0.50, 0.75],
        action_cost=float(config["decision"]["review_cost"]),
    )
    sensitivity.to_csv(reports / "intervention_sensitivity.csv", index=False)

    drift_report = build_drift_report(
        train,
        test,
        [
            "remaining_workload",
            "recent_pace",
            "weather_severity",
            "congestion_proxy",
            "rcot_minutes",
            "reference_support",
            "rcot_trust",
        ],
    )
    drift_report.to_csv(reports / "drift_report.csv", index=False)

    predictions = pd.concat(
        [
            test[["snapshot_id"]].reset_index(drop=True),
            test_quantiles.reset_index(drop=True),
            tail_scores.reset_index(drop=True),
        ],
        axis=1,
    )
    predictions["tail_threshold"] = test_thresholds
    predictions["actual_tail"] = (y_test > test_thresholds).astype(int)
    predictions["decision_policy"] = decision_champion
    predictions["decision_priority"] = policy_scores[decision_champion]
    predictions.to_csv(reports / "test_predictions.csv", index=False)

    snapshots_for_sql = test.copy()
    _write_sqlite_outputs(artifacts / "operational_evidence.db", snapshots_for_sql, predictions)

    primary_base = point_metrics(y_test, base_pred)["mae"]
    primary_rcot = point_metrics(y_test, rcot_pred)["mae"]
    relative_improvement = (primary_base - primary_rcot) / max(primary_base, 1e-9)
    decision_test_metric = capacity_metrics(
        y_test,
        test_thresholds,
        policy_scores[decision_champion],
        [selection_capacity],
    )[0]
    random_test_metric = capacity_metrics(
        y_test,
        test_thresholds,
        policy_scores["random"],
        [selection_capacity],
    )[0]
    champion_test_mae = point_metrics(y_test, test_quantiles["q50"].to_numpy(dtype=float))["mae"]
    business_baseline_test_mae = point_metrics(y_test, cohort_pred)["mae"]
    test_tail_events = int((y_test > test_thresholds).sum())
    system_release = evaluate_system_release(
        interval_coverage=float(interval_result["coverage"]),
        target_coverage=float(config["calibration"]["target_coverage"]),
        quantile_crossing_rate=float(interval_result["quantile_crossing_rate"]),
        predictions=test_quantiles,
        test_rows=len(test),
        champion_mae=champion_test_mae,
        business_baseline_mae=business_baseline_test_mae,
        decision_excess_capture=float(decision_test_metric["excess_minutes_capture"]),
        random_excess_capture=float(random_test_metric["excess_minutes_capture"]),
        test_tail_events=test_tail_events,
        coverage_tolerance=float(config["calibration"]["coverage_tolerance"]),
        minimum_test_rows=int(config["calibration"]["minimum_test_rows"]),
        minimum_test_tail_events=int(config["decision"]["minimum_test_tail_events"]),
    )
    system_pass = bool(system_release["pass"])
    gate = (
        "PASS_RCOT_CHAMPION"
        if system_pass and rcot_promoted
        else "PASS_BASELINE_CHAMPION"
        if system_pass
        else "ITERATE_SYSTEM"
    )
    release_decision = {
        "release_gate": gate,
        "system_release": system_release,
        "rcot_promotion_gate": "PROMOTE" if rcot_promoted else "HOLD",
        "rcot_promotion": promotion.as_dict(),
        "validation_q50_mae_without_rcot": validation_mae_base,
        "validation_q50_mae_with_rcot": validation_mae_rcot,
        "quantile_champion": "with_rcot" if rcot_promoted else "without_rcot",
        "decision_champion": decision_champion,
        "decision_selection_mode": decision_selection_mode,
        "decision_selection_capacity": selection_capacity,
        "minimum_validation_tail_events": minimum_validation_tail_events,
        "validation_tail_events": validation_tail_events,
        "validation_policy_excess_capture_at_selection_capacity": validation_policy_capture,
        "artifact_schema_version": 1,
        "serving_policy": {
            "trust_review_threshold": 0.35,
            "support_review_threshold": 0.40,
            "tail_probability_review_threshold": 0.50,
        },
    }
    atomic_write_json(artifacts / "release_decision.json", release_decision, default=_json_default)
    serving_bundle = None
    if output_namespace is None:
        serving_bundle = _publish_serving_bundle(artifacts)
    summary: dict[str, object] = {
        "mode": mode,
        "output_namespace": output_namespace,
        "seed": seed,
        "snapshot_rows": len(generated.snapshots),
        "pending_task_rows": len(generated.pending_tasks),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "rcot_relative_mae_improvement": relative_improvement,
        "rcot_promotion_gate": release_decision["rcot_promotion_gate"],
        "quantile_champion": release_decision["quantile_champion"],
        "decision_champion": decision_champion,
        "interval_coverage": interval_result["coverage"],
        "rolling_replay_coverage": float(rolling_replay["covered"].mean()),
        "release_gate": gate,
        "advanced": advanced_result,
        "serving_bundle": serving_bundle,
        "claim_boundary": (
            "offline synthetic smoke evidence; no actual delay-prevention claim"
            if data_cfg["source"] == "synthetic"
            else "offline normalized LaDe evidence; no actual delay-prevention claim"
        ),
    }
    atomic_write_json(reports / "run_summary.json", summary, default=_json_default)
    write_markdown_report(
        reports / "run_report.md",
        summary=summary,
        model_metrics=model_metrics.fillna(""),
        capacity_metrics=capacity_report,
        drift_report=drift_report,
    )

    manifest = []
    for path in _run_manifest_files(
        artifacts, reports, data_dir, namespaced=output_namespace is not None
    ):
        manifest.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    atomic_write_json(artifacts / "artifact_manifest.json", manifest)
    return summary


def run(config_path: Path, mode: str, output_namespace: str | None = None) -> dict[str, object]:
    validated_namespace = _validated_output_namespace(output_namespace)
    lock_name = validated_namespace or "root"
    lock = ExclusiveFileLock(
        ROOT / "artifacts" / ".locks" / f"{lock_name}.lock",
        timeout_seconds=float(os.getenv("REFERENCE_ETA_LOCK_TIMEOUT_SECONDS", "5")),
        stale_after_seconds=float(os.getenv("REFERENCE_ETA_LOCK_STALE_SECONDS", "3600")),
        purpose=f"pipeline run ({lock_name})",
    )
    with lock:
        return _run_unlocked(config_path, mode, validated_namespace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke.yaml")
    parser.add_argument(
        "--mode",
        choices=["smoke", "baselines", "evaluate", "simulate", "gpu", "all"],
        default="smoke",
    )
    parser.add_argument("--output-namespace", default=None)
    parser.add_argument(
        "--require-release-pass",
        action="store_true",
        help="Exit nonzero unless the system release gate begins with PASS_",
    )
    parser.add_argument(
        "--force-process-exit",
        action="store_true",
        help="Flush output and bypass native-library interpreter shutdown after completion",
    )
    args = parser.parse_args()
    summary = run(args.config, args.mode, output_namespace=args.output_namespace)
    print(json.dumps(summary, indent=2, default=_json_default))
    release_passed = str(summary["release_gate"]).startswith("PASS_")
    if args.force_process_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if (release_passed or not args.require_release_pass) else 1)
    if args.require_release_pass and not release_passed:
        raise SystemExit(f"Release gate did not pass: {summary['release_gate']}")


if __name__ == "__main__":
    main()
