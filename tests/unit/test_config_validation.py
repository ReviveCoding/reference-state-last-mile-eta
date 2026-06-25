import pytest

from reference_eta.config import validate_config


def test_invalid_split_is_rejected() -> None:
    config = {
        "seed": 1,
        "data": {
            "source": "synthetic",
            "n_courier_days": 20,
            "start_date": "2026-01-01",
            "cities": ["Boston"],
            "min_tasks": 8,
            "max_tasks": 10,
            "snapshot_stride": 2,
        },
        "split": {"train": 0.8, "validation": 0.2, "calibration": 0.1, "test": 0.1},
        "rcot": {
            "min_cohort_rows": 4,
            "min_cohort_groups": 2,
            "support_shrinkage": 10,
            "max_dispersion_minutes": 45,
            "cross_fit_splits": 3,
        },
        "model": {},
        "advanced": {},
        "calibration": {"target_coverage": 0.8, "coverage_tolerance": 0.1, "minimum_test_rows": 10},
        "decision": {"capacities": [0.1], "review_cost": 1.0},
    }
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_config(config)


def test_example_configs_satisfy_current_contract() -> None:
    from reference_eta.config import load_config

    load_config("configs/lade_normalized.example.yaml")
    load_config("configs/gpu_full.example.yaml")
