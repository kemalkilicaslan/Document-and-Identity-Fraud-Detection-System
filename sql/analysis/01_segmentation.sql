-- Fraud rate and loss by channel, product and plan type.
SELECT
    f.channel,
    f.product_type,
    f.plan_type,
    COUNT(*)                                   AS applications,
    SUM(l.is_fraud)                            AS frauds,
    ROUND(100.0 * AVG(l.is_fraud), 2)          AS fraud_rate_pct,
    ROUND(SUM(l.loss_amount) / 1e6, 2)         AS fraud_loss_m,
    ROUND(100.0 * SUM(l.is_fraud) / SUM(SUM(l.is_fraud)) OVER (), 1) AS share_of_all_fraud_pct
FROM application_features f
JOIN fraud_labels l USING (application_id)
GROUP BY f.channel, f.product_type, f.plan_type
ORDER BY fraud_rate_pct DESC;
