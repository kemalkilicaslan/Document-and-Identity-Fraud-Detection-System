-- Alerts, precision, recall, loss caught and unique catches per rule.
WITH h AS (
    SELECT r.rule_id, r.application_id, l.is_fraud, l.loss_amount,
           COUNT(*) OVER (PARTITION BY r.application_id) AS rules_on_app
    FROM rule_hits r JOIN fraud_labels l USING (application_id)
),
tot AS (SELECT SUM(is_fraud) AS frauds, SUM(loss_amount) AS loss FROM fraud_labels)
SELECT h.rule_id,
       COUNT(*)                                             AS alerts,
       SUM(h.is_fraud)                                      AS true_positives,
       ROUND(100.0 * AVG(h.is_fraud), 1)                    AS precision_pct,
       ROUND(100.0 * SUM(h.is_fraud) / ANY_VALUE(tot.frauds), 1) AS recall_pct,
       ROUND(100.0 * SUM(h.loss_amount) / ANY_VALUE(tot.loss), 1) AS loss_caught_pct,
       SUM(CASE WHEN h.rules_on_app = 1 AND h.is_fraud = 1 THEN 1 ELSE 0 END) AS unique_tp,
       ROUND((COUNT(*) - SUM(h.is_fraud)) * 1.0 / NULLIF(SUM(h.is_fraud), 0), 1) AS fp_per_tp
FROM h CROSS JOIN tot
GROUP BY h.rule_id
ORDER BY true_positives DESC;
