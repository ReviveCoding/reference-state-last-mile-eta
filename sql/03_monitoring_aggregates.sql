DROP TABLE IF EXISTS monitoring_aggregates;
CREATE TABLE monitoring_aggregates AS
SELECT
    city,
    work_date,
    COUNT(*) AS prediction_count,
    AVG(signed_error) AS mean_signed_error,
    AVG(ABS(signed_error)) AS mean_absolute_error,
    AVG(interval_covered) AS empirical_coverage
FROM prediction_reconciliation
GROUP BY city, work_date;
