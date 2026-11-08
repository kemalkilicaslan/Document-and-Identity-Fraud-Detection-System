-- Phones, IBANs, own devices and guarantors shared by 3+ identities.
-- Synthetic phone numbers, IBANs and guarantor national ID numbers can coincide with real ones, so only a
-- masked form leaves the database. Device ids are synthetic and stay readable.
-- Masked values are not unique, so link_id is the row key: a sequence number in order of first appearance.
WITH links AS (
    SELECT 'PHONE' AS link_type, phone AS val, tckn, application_id, app_ts, branch_id, agent_id FROM applications
    UNION ALL
    SELECT 'IBAN', iban, tckn, application_id, app_ts, branch_id, agent_id FROM applications
    UNION ALL
    SELECT 'DEVICE', device_id, tckn, application_id, app_ts, branch_id, agent_id FROM applications
    WHERE channel IN ('WEB', 'MOBILE')
    UNION ALL
    SELECT 'GUARANTOR', guarantor_tckn, tckn, application_id, app_ts, branch_id, agent_id FROM applications
    WHERE guarantor_tckn IS NOT NULL
),
g AS (
    SELECT link_type, val,
           COUNT(DISTINCT k.tckn)                         AS identities,
           COUNT(*)                                       AS applications,
           COUNT(DISTINCT k.branch_id)                    AS branches,
           COUNT(DISTINCT k.agent_id)                     AS agents,
           MIN(k.app_ts)                                  AS first_seen,
           MAX(k.app_ts)                                  AS last_seen,
           date_diff('day', MIN(k.app_ts), MAX(k.app_ts)) AS span_days,
           SUM(l.is_fraud)                                AS confirmed_fraud,
           ROUND(SUM(l.loss_amount) / 1e6, 2)             AS fraud_loss_m
    FROM links k JOIN fraud_labels l USING (application_id)
    GROUP BY link_type, val
    HAVING COUNT(DISTINCT k.tckn) >= 3
)
SELECT 'LNK-' || lpad(CAST(ROW_NUMBER() OVER (ORDER BY first_seen, link_type, val) AS VARCHAR), 4, '0') AS link_id,
       link_type,
       CASE link_type
           WHEN 'PHONE'     THEN substr(val, 1, 4) || '*****' || right(val, 2)
           WHEN 'IBAN'      THEN substr(val, 1, 4) || repeat('*', length(val) - 8) || right(val, 4)
           WHEN 'GUARANTOR' THEN substr(val, 1, 3) || '******' || right(val, 2)
           ELSE val
       END AS link_value,
       identities, applications, branches, agents, first_seen, last_seen, span_days, confirmed_fraud, fraud_loss_m
FROM g
ORDER BY identities DESC, span_days, link_id;
