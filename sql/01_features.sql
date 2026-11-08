-- Point-in-time features: every link and velocity count only sees earlier applications.

CREATE OR REPLACE TABLE application_features AS
WITH a AS (
    SELECT ap.*, ip24(ap.ip_address) AS ip_net, ap.channel IN ('WEB', 'MOBILE') AS is_online
    FROM applications ap
),
same_tckn AS (
    SELECT a.application_id, COUNT(*) AS same_tckn_apps_30d,
           COUNT(*) FILTER (WHERE b.branch_id <> a.branch_id) AS same_tckn_other_branch_30d
    FROM a JOIN a b ON b.tckn = a.tckn
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 30 DAY
    GROUP BY 1
),
phone_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS phone_other_tckn_90d
    FROM a JOIN a b ON b.phone = a.phone AND b.tckn <> a.tckn
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 90 DAY
    GROUP BY 1
),
iban_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS iban_other_tckn_90d
    FROM a JOIN a b ON b.iban = a.iban AND b.tckn <> a.tckn
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 90 DAY
    GROUP BY 1
),
address_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS address_other_tckn_30d
    FROM a JOIN a b ON b.address = a.address AND b.tckn <> a.tckn
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 30 DAY
    GROUP BY 1
),
device_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS device_other_tckn_7d
    FROM a JOIN a b ON b.device_id = a.device_id AND b.tckn <> a.tckn
     AND a.is_online AND b.is_online
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 7 DAY
    GROUP BY 1
),
ip_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS ipnet_other_tckn_7d
    FROM a JOIN a b ON b.ip_net = a.ip_net AND b.tckn <> a.tckn
     AND a.is_online AND b.is_online
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 7 DAY
    GROUP BY 1
),
guarantor_link AS (
    SELECT a.application_id, COUNT(DISTINCT b.tckn) AS guarantor_other_apps_180d
    FROM a JOIN a b ON b.guarantor_tckn = a.guarantor_tckn AND b.tckn <> a.tckn
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 180 DAY
    GROUP BY 1
),
agent_hist AS (
    SELECT a.application_id,
           COUNT(*) AS agent_prior_apps_180d,
           COUNT(*) FILTER (WHERE c.cancel_date < CAST(a.app_ts AS DATE)
                             AND c.cancel_date <= CAST(b.app_ts AS DATE) + 120) AS agent_prior_early_cancels_180d,
           COUNT(*) FILTER (WHERE b.app_ts >= a.app_ts - INTERVAL 30 DAY) AS agent_apps_30d
    FROM a JOIN a b ON b.agent_id = a.agent_id
     AND b.app_ts < a.app_ts AND b.app_ts >= a.app_ts - INTERVAL 180 DAY
    JOIN contracts c ON c.application_id = b.application_id
    GROUP BY 1
)
SELECT
    a.application_id, a.app_ts, CAST(date_trunc('month', a.app_ts) AS DATE) AS app_month,
    a.channel, a.branch_id, a.agent_id, a.product_type, a.plan_type, a.occupation,
    a.financing_amount, a.declared_income,
    date_diff('year', a.birth_date, CAST(a.app_ts AS DATE))           AS age,
    NOT tckn_valid(a.tckn)                                            AS tckn_invalid,
    a.guarantor_tckn IS NOT NULL AND NOT tckn_valid(a.guarantor_tckn) AS guarantor_tckn_invalid,
    a.declared_income / r.median_income                               AS income_to_occupation_median,
    a.monthly_installment / NULLIF(a.declared_income, 0)              AS installment_to_income,
    a.down_payment / NULLIF(a.financing_amount, 0)                    AS down_payment_ratio,
    a.down_payment / NULLIF(a.declared_income, 0)                     AS down_payment_months_income,
    a.allocation_month,
    hour(a.app_ts) < 6                                                AS is_night,
    a.is_online,
    regexp_matches(split_part(a.email, '@', 1), '[0-9]{3,}')          AS email_many_digits,
    day(a.app_ts) >= 25                                               AS month_end,
    COALESCE(st.same_tckn_apps_30d, 0)          AS same_tckn_apps_30d,
    COALESCE(st.same_tckn_other_branch_30d, 0)  AS same_tckn_other_branch_30d,
    COALESCE(pl.phone_other_tckn_90d, 0)        AS phone_other_tckn_90d,
    COALESCE(il.iban_other_tckn_90d, 0)         AS iban_other_tckn_90d,
    COALESCE(al.address_other_tckn_30d, 0)      AS address_other_tckn_30d,
    COALESCE(dl.device_other_tckn_7d, 0)        AS device_other_tckn_7d,
    COALESCE(ipl.ipnet_other_tckn_7d, 0)        AS ipnet_other_tckn_7d,
    COALESCE(gl.guarantor_other_apps_180d, 0)   AS guarantor_other_apps_180d,
    COALESCE(ah.agent_prior_apps_180d, 0)       AS agent_prior_apps_180d,
    COALESCE(ah.agent_apps_30d, 0)              AS agent_apps_30d,
    COALESCE(ah.agent_prior_early_cancels_180d, 0) * 1.0
        / NULLIF(ah.agent_prior_apps_180d, 0)   AS agent_early_cancel_rate_180d,
    d.doc_score,
    COALESCE(d.mrz_checkdigit_fail, FALSE)      AS mrz_checkdigit_fail,
    COALESCE(d.mrz_viz_mismatch, FALSE)         AS mrz_viz_mismatch,
    COALESCE(d.date_logic_fail, FALSE)          AS date_logic_fail
FROM a
JOIN occupation_reference r USING (occupation)
JOIN documents d USING (application_id)
LEFT JOIN same_tckn st      USING (application_id)
LEFT JOIN phone_link pl     USING (application_id)
LEFT JOIN iban_link il      USING (application_id)
LEFT JOIN address_link al   USING (application_id)
LEFT JOIN device_link dl    USING (application_id)
LEFT JOIN ip_link ipl       USING (application_id)
LEFT JOIN guarantor_link gl USING (application_id)
LEFT JOIN agent_hist ah     USING (application_id);
