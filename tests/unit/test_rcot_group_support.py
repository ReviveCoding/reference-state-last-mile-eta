import pandas as pd

from reference_eta.features.rcot import ReferenceOperationalTimeTransformer


def test_rcot_support_counts_distinct_courier_days() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "courier_id": "c1",
                "work_date": "2026-01-01",
                "city": "Boston",
                "initial_workload": 100.0,
                "task_density": 1.0,
                "aoi_transition_burden": 0.1,
                "query_hour": 9.0,
                "elapsed_minutes": 10.0 + index,
                "observed_progress": min(0.05 * index, 0.95),
            }
        )
    frame = pd.DataFrame(rows)
    transformer = ReferenceOperationalTimeTransformer(
        min_cohort_rows=2,
        min_cohort_groups=2,
        support_shrinkage=9.0,
    ).fit(frame)
    transformed = transformer.transform(frame.iloc[[0]])
    assert transformed.iloc[0]["reference_support_groups"] == 1.0
    assert transformed.iloc[0]["reference_support"] == 0.1
