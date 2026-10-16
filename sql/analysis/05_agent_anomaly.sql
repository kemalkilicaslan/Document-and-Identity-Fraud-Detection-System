-- Agent early cancellations (<= 120 days), first-payment default and month-end share vs. branch peers.
WITH fpd AS (
    SELECT application_id, MAX(CASE WHEN installment_no BETWEEN 1 AND 3 AND status = 'UNPAID' THEN 1 ELSE 0 END) AS fpd
    FROM payments GROUP BY application_id
),
agent AS (
    SELECT a.agent_id, a.branch_id,
           COUNT(*) AS applications,
           AVG(CASE WHEN c.cancel_date IS NOT NULL AND c.cancel_date <= CAST(a.app_ts AS DATE) + 120 THEN 1.0 ELSE 0 END) AS early_cancel_rate,
           AVG(COALESCE(p.fpd, 0) * 1.0) AS fpd_rate,
           AVG(CASE WHEN day(a.app_ts) >= 25 THEN 1.0 ELSE 0 END) AS month_end_share,
           SUM(l.is_fraud) AS confirmed_fraud
    FROM applications a
    JOIN contracts c USING (application_id)
    JOIN fraud_labels l USING (application_id)
    LEFT JOIN fpd p USING (application_id)
    GROUP BY a.agent_id, a.branch_id
    HAVING COUNT(*) >= 20
)
SELECT agent_id, branch_id, applications,
       ROUND(100 * early_cancel_rate, 1) AS early_cancel_pct,
       ROUND(100 * fpd_rate, 1)          AS fpd_pct,
       ROUND(100 * month_end_share, 1)   AS month_end_pct,
       ROUND((early_cancel_rate - AVG(early_cancel_rate) OVER ()) / NULLIF(STDDEV_SAMP(early_cancel_rate) OVER (), 0), 2) AS early_cancel_z,
       ROUND(early_cancel_rate - AVG(early_cancel_rate) OVER (PARTITION BY branch_id), 3) AS vs_branch_peers,
       confirmed_fraud
FROM agent
ORDER BY early_cancel_z DESC, agent_id, branch_id;
