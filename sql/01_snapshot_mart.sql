DROP TABLE IF EXISTS snapshot_mart;
CREATE TABLE snapshot_mart AS
SELECT
    snapshot_id,
    courier_id,
    work_date,
    city,
    query_time,
    completed_task_count,
    remaining_task_count,
    remaining_workload,
    route_phase,
    rcot_minutes,
    reference_support,
    rcot_trust,
    target_route_remaining_minutes
FROM snapshots;
