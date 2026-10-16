-- Branch fraud rate as a z-score against all branches.
WITH b AS (
    SELECT a.branch_id, br.city,
           COUNT(*) AS applications,
           AVG(l.is_fraud) AS fraud_rate,
           SUM(l.loss_amount) AS loss
    FROM applications a
    JOIN branches br USING (branch_id)
    JOIN fraud_labels l USING (application_id)
    GROUP BY a.branch_id, br.city
)
SELECT branch_id, city, applications,
       ROUND(100 * fraud_rate, 2) AS fraud_rate_pct,
       ROUND(loss / 1e6, 2) AS fraud_loss_m,
       ROUND((fraud_rate - AVG(fraud_rate) OVER ()) / NULLIF(STDDEV_SAMP(fraud_rate) OVER (), 0), 2) AS fraud_rate_z,
       RANK() OVER (ORDER BY fraud_rate DESC) AS risk_rank
FROM b
ORDER BY fraud_rate_z DESC;
