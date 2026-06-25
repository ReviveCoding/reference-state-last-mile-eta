from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REQUIRED_SECTIONS = {
    "seed",
    "data",
    "split",
    "rcot",
    "model",
    "advanced",
    "calibration",
    "decision",
}


def _require_keys(mapping: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys.difference(mapping)
    if missing:
        raise ValueError(f"Missing configuration keys in {context}: {sorted(missing)}")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not np.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{name} must be an integer")
    return parsed


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _require_keys(config, _REQUIRED_SECTIONS, "root")
    _integer(config["seed"], "seed")
    for section in _REQUIRED_SECTIONS - {"seed"}:
        if not isinstance(config[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping")

    split = config["split"]
    _require_keys(split, {"train", "validation", "calibration", "test"}, "split")
    fractions = [
        _finite_float(split[key], f"split.{key}")
        for key in ("train", "validation", "calibration", "test")
    ]
    if any(value <= 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to 1.0")

    data = config["data"]
    _require_keys(data, {"source"}, "data")
    if data["source"] == "synthetic":
        _require_keys(
            data,
            {"n_courier_days", "start_date", "cities", "min_tasks", "max_tasks", "snapshot_stride"},
            "data.synthetic",
        )
        if _integer(data["n_courier_days"], "data.n_courier_days") < 8:
            raise ValueError("Synthetic data requires at least eight courier-days")
        cities = data["cities"]
        if (
            not isinstance(cities, list)
            or not cities
            or any(not str(city).strip() for city in cities)
        ):
            raise ValueError("Synthetic data requires a non-empty list of city names")
        if len(set(map(str, cities))) != len(cities):
            raise ValueError("Synthetic city names must be unique")
        if _integer(data["min_tasks"], "data.min_tasks") < 4 or _integer(
            data["max_tasks"], "data.max_tasks"
        ) < _integer(data["min_tasks"], "data.min_tasks"):
            raise ValueError("Synthetic task bounds are invalid")
        if _integer(data["snapshot_stride"], "data.snapshot_stride") < 1:
            raise ValueError("snapshot_stride must be positive")
    elif data["source"] == "lade_normalized":
        _require_keys(
            data, {"path", "snapshot_stride", "default_service_minutes"}, "data.lade_normalized"
        )
        if _integer(data["snapshot_stride"], "data.snapshot_stride") < 1:
            raise ValueError("snapshot_stride must be positive")
        if _finite_float(data["default_service_minutes"], "data.default_service_minutes") <= 0.0:
            raise ValueError("default_service_minutes must be positive")
        if _finite_float(data.get("assumed_speed_kmh", 15.0), "data.assumed_speed_kmh") <= 0.0:
            raise ValueError("assumed_speed_kmh must be positive")
    else:
        raise ValueError(f"Unsupported data source: {data['source']}")

    rcot = config["rcot"]
    _require_keys(
        rcot,
        {
            "min_cohort_rows",
            "min_cohort_groups",
            "support_shrinkage",
            "max_dispersion_minutes",
            "cross_fit_splits",
        },
        "rcot",
    )
    if (
        _integer(rcot["min_cohort_rows"], "rcot.min_cohort_rows") < 2
        or _integer(rcot["min_cohort_groups"], "rcot.min_cohort_groups") < 2
    ):
        raise ValueError("RCOT cohort minimums must be at least 2")
    if _integer(rcot["cross_fit_splits"], "rcot.cross_fit_splits") < 2:
        raise ValueError("RCOT cross_fit_splits must be at least 2")
    if (
        _finite_float(rcot["support_shrinkage"], "rcot.support_shrinkage") <= 0.0
        or _finite_float(rcot["max_dispersion_minutes"], "rcot.max_dispersion_minutes") <= 0.0
    ):
        raise ValueError("RCOT shrinkage and dispersion limits must be positive")

    model = config["model"]
    _require_keys(
        model,
        {"n_estimators", "learning_rate", "num_leaves", "min_child_samples", "random_state"},
        "model",
    )
    if (
        _integer(model["n_estimators"], "model.n_estimators") < 1
        or _integer(model["num_leaves"], "model.num_leaves") < 2
    ):
        raise ValueError("Model tree parameters are invalid")
    if (
        _finite_float(model["learning_rate"], "model.learning_rate") <= 0.0
        or _integer(model["min_child_samples"], "model.min_child_samples") < 1
    ):
        raise ValueError("Model learning parameters are invalid")

    _integer(model["random_state"], "model.random_state")

    advanced = config["advanced"]
    _require_keys(
        advanced,
        {
            "enabled",
            "epochs",
            "batch_size",
            "hidden_dim",
            "learning_rate",
            "max_tasks",
            "amp",
            "deterministic",
        },
        "advanced",
    )
    if (
        not isinstance(advanced["enabled"], bool)
        or not isinstance(advanced["amp"], bool)
        or not isinstance(advanced["deterministic"], bool)
    ):
        raise ValueError(
            "advanced.enabled, advanced.amp, and advanced.deterministic must be booleans"
        )
    if (
        _integer(advanced["epochs"], "advanced.epochs") < 1
        or _integer(advanced["batch_size"], "advanced.batch_size") < 1
    ):
        raise ValueError("Advanced training epochs and batch_size must be positive")
    if (
        _integer(advanced["hidden_dim"], "advanced.hidden_dim") < 4
        or _integer(advanced["max_tasks"], "advanced.max_tasks") < 2
    ):
        raise ValueError("Advanced model dimensions are invalid")
    if _finite_float(advanced["learning_rate"], "advanced.learning_rate") <= 0.0:
        raise ValueError("Advanced learning_rate must be positive")

    calibration = config["calibration"]
    _require_keys(
        calibration,
        {
            "lower_alpha",
            "upper_alpha",
            "target_coverage",
            "coverage_tolerance",
            "minimum_test_rows",
        },
        "calibration",
    )
    lower_alpha = _finite_float(calibration["lower_alpha"], "calibration.lower_alpha")
    upper_alpha = _finite_float(calibration["upper_alpha"], "calibration.upper_alpha")
    target_coverage = _finite_float(calibration["target_coverage"], "calibration.target_coverage")
    if not (0.0 < lower_alpha < 0.5 < upper_alpha < 1.0):
        raise ValueError("Calibration alphas must satisfy 0 < lower < 0.5 < upper < 1")
    if abs(lower_alpha - 0.10) > 1e-9 or abs(upper_alpha - 0.90) > 1e-9:
        raise ValueError(
            "The current q10/q50/q90 API contract requires lower_alpha=0.10 and upper_alpha=0.90"
        )
    if abs((upper_alpha - lower_alpha) - target_coverage) > 1e-8:
        raise ValueError("target_coverage must equal upper_alpha - lower_alpha")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0, 1)")
    if (
        not 0.0
        < _finite_float(calibration["coverage_tolerance"], "calibration.coverage_tolerance")
        < 0.5
    ):
        raise ValueError("coverage_tolerance must be in (0, 0.5)")
    if _integer(calibration["minimum_test_rows"], "calibration.minimum_test_rows") < 1:
        raise ValueError("minimum_test_rows must be positive")

    decision = config["decision"]
    _require_keys(
        decision,
        {
            "capacities",
            "review_cost",
            "selection_capacity",
            "minimum_validation_tail_events",
            "minimum_test_tail_events",
        },
        "decision",
    )
    if not isinstance(decision["capacities"], list):
        raise ValueError("decision.capacities must be a list")
    capacities = [_finite_float(value, "decision.capacities") for value in decision["capacities"]]
    if not capacities or any(not 0.0 < value <= 1.0 for value in capacities):
        raise ValueError("decision capacities must be non-empty values in (0, 1]")
    if capacities != sorted(set(capacities)):
        raise ValueError("decision capacities must be unique and sorted")
    selection_capacity = _finite_float(
        decision["selection_capacity"], "decision.selection_capacity"
    )
    if not 0.0 < selection_capacity <= 1.0:
        raise ValueError("decision.selection_capacity must be in (0, 1]")
    if (
        _integer(
            decision["minimum_validation_tail_events"],
            "decision.minimum_validation_tail_events",
        )
        < 1
    ):
        raise ValueError("decision.minimum_validation_tail_events must be positive")
    if (
        _integer(
            decision["minimum_test_tail_events"],
            "decision.minimum_test_tail_events",
        )
        < 1
    ):
        raise ValueError("decision.minimum_test_tail_events must be positive")
    if _finite_float(decision["review_cost"], "decision.review_cost") < 0.0:
        raise ValueError("review_cost cannot be negative")

    return config


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return validate_config(config)
