DROP TABLE IF EXISTS reference_quality;
CREATE TABLE reference_quality AS
SELECT
    CASE
        WHEN reference_support < 0.40 THEN 'low'
        WHEN reference_support < 0.70 THEN 'medium'
        ELSE 'high'
    END AS support_band,
    COUNT(*) AS row_count,
    AVG(ABS(target_route_remaining_minutes - q50)) AS mae,
    AVG(rcot_trust) AS mean_trust
FROM snapshots
JOIN predictions USING (snapshot_id)
GROUP BY support_band;
