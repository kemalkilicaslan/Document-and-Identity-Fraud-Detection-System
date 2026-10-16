-- First-payment default and late-payment rates by application month and channel.
WITH p AS (
    SELECT application_id,
           MAX(CASE WHEN installment_no BETWEEN 1 AND 3 AND status = 'UNPAID' THEN 1 ELSE 0 END) AS fpd,
           AVG(CASE WHEN installment_no >= 1 AND status = 'LATE' THEN 1.0 ELSE 0 END)           AS late_share,
           MAX(installment_no) AS installments_due
    FROM payments GROUP BY application_id
)
SELECT CAST(date_trunc('month', a.app_ts) AS DATE) AS app_month, a.channel,
       COUNT(*)                                AS applications,
       ROUND(100 * AVG(p.fpd), 2)              AS fpd_pct,
       ROUND(100 * AVG(p.late_share), 2)       AS late_payment_pct,
       ROUND(100 * AVG(l.is_fraud), 2)         AS fraud_rate_pct
FROM applications a
JOIN p USING (application_id)
JOIN fraud_labels l USING (application_id)
WHERE p.installments_due >= 3
GROUP BY 1, 2
ORDER BY 1, 2;
