-- Identities that applied more than once within 30 days, and at how many branches.
-- Synthetic national ID numbers are random but checksum-valid, so some can coincide with real people's numbers.
-- Rows are grouped by the full number, but only a masked form (first 3 and last 2 digits) leaves the database.
-- Masked values are not unique, so identity_id is the row key: a sequence number in order of first application,
-- which carries no information about the number itself.
WITH x AS (
    SELECT tckn, application_id, app_ts, branch_id,
           LAG(app_ts) OVER (PARTITION BY tckn ORDER BY app_ts) AS prev_ts
    FROM applications
),
g AS (
    SELECT x.tckn,
           COUNT(*)                                  AS applications,
           COUNT(DISTINCT x.branch_id)               AS branches,
           MIN(x.app_ts)                             AS first_app,
           MAX(x.app_ts)                             AS last_app,
           MIN(date_diff('day', x.prev_ts, x.app_ts)) AS min_gap_days,
           SUM(l.is_fraud)                           AS confirmed_fraud
    FROM x JOIN fraud_labels l USING (application_id)
    GROUP BY x.tckn
    HAVING COUNT(*) >= 2 AND MIN(date_diff('day', x.prev_ts, x.app_ts)) <= 30
)
SELECT 'ID-' || lpad(CAST(ROW_NUMBER() OVER (ORDER BY first_app, tckn) AS VARCHAR), 4, '0') AS identity_id,
       substr(tckn, 1, 3) || '******' || substr(tckn, 10, 2) AS tckn_masked,
       applications, branches, first_app, last_app, min_gap_days, confirmed_fraud
FROM g
ORDER BY branches DESC, applications DESC, identity_id;
