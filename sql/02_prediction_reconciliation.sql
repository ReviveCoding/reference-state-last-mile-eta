DROP TABLE IF EXISTS prediction_reconciliation;
CREATE TABLE prediction_reconciliation AS
SELECT
    p.snapshot_id,
    s.city,
    s.work_date,
    p.q10,
    p.q50,
    p.q90,
    s.target_route_remaining_minutes AS actual_minutes,
    p.q50 - s.target_route_remaining_minutes AS signed_error,
    CASE
        WHEN s.target_route_remaining_minutes BETWEEN p.q10 AND p.q90 THEN 1
        ELSE 0
    END AS interval_covered
FROM predictions p
JOIN snapshot_mart s USING (snapshot_id);
