-- Monthly fraud trend by scenario (count and loss).
SELECT CAST(date_trunc('month', a.app_ts) AS DATE) AS app_month,
       l.scenario,
       COUNT(*)                           AS frauds,
       ROUND(SUM(l.loss_amount) / 1e6, 2) AS loss_m
FROM applications a JOIN fraud_labels l USING (application_id)
WHERE l.is_fraud = 1
GROUP BY 1, 2
ORDER BY 1, 2;
