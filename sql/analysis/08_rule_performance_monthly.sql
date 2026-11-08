-- Monthly hit rate and precision per rule.
SELECT CAST(date_trunc('month', f.app_ts) AS DATE) AS app_month,
       r.rule_id,
       COUNT(*)                                     AS alerts,
       ROUND(100.0 * COUNT(*) / ANY_VALUE(m.apps), 2) AS hit_rate_pct,
       ROUND(100.0 * AVG(l.is_fraud), 1)            AS precision_pct
FROM rule_hits r
JOIN application_features f USING (application_id)
JOIN fraud_labels l USING (application_id)
JOIN (SELECT app_month, COUNT(*) AS apps FROM application_features GROUP BY 1) m
  ON m.app_month = f.app_month
GROUP BY 1, 2
ORDER BY 1, 2;
