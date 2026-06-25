DROP TABLE IF EXISTS capacity_report;
CREATE TABLE capacity_report AS
WITH ranked AS (
    SELECT
        p.*,
        ROW_NUMBER() OVER (ORDER BY decision_priority DESC) AS priority_rank,
        COUNT(*) OVER () AS total_rows
    FROM predictions p
)
SELECT
    snapshot_id,
    decision_policy,
    decision_priority,
    CAST(priority_rank AS REAL) / total_rows AS selected_fraction,
    CASE WHEN actual_tail = 1 THEN 1 ELSE 0 END AS actual_tail
FROM ranked
WHERE CAST(priority_rank AS REAL) / total_rows <= 0.20;
