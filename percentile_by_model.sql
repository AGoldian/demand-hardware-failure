WITH data AS (
    SELECT
        model,
        julianday(failure_date) - julianday(start_date) as diff
    FROM failure_info
),
    ranked AS (
        SELECT model,
               diff,
               row_number() over (ORDER BY diff) AS row_num,
               COUNT(*) OVER () AS total_rows
        FROM data
    )
SELECT model, AVG(diff) AS percentile_90
FROM ranked
WHERE row_num <= total_rows * 0.9
GROUP BY model;
