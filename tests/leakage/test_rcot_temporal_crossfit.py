import pandas as pd

from reference_eta.features.rcot import ReferenceOperationalTimeTransformer


def test_temporal_crossfit_uses_no_reference_for_first_block() -> None:
    rows = []
    for day in range(10):
        for step in (1, 2):
            rows.append(
                {
                    "courier_id": f"c{day}",
                    "work_date": f"2026-01-{day + 1:02d}",
                    "city": "A",
                    "initial_workload": 100.0,
                    "task_density": 1.0,
                    "aoi_transition_burden": 0.2,
                    "query_hour": 9.0,
                    "elapsed_minutes": 10.0 * step,
                    "observed_progress": 0.2 * step,
                }
            )
    frame = pd.DataFrame(rows)
    output = ReferenceOperationalTimeTransformer(
        min_cohort_rows=2,
        min_cohort_groups=2,
        support_shrinkage=2.0,
    ).fit_transform_cross_fitted(frame, n_splits=5)
    first_dates = {"2026-01-01", "2026-01-02"}
    warmup = output[output["work_date"].isin(first_dates)]
    later = output[~output["work_date"].isin(first_dates)]
    assert (warmup["reference_support"] == 0.0).all()
    assert (warmup["rcot_trust"] == 0.0).all()
    assert (warmup["reference_level"] == "temporal_warmup").all()
    assert (later["reference_support"] > 0.0).any()
