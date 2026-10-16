-- Alerts of the last 30 days, highest hybrid score first, with reasons and owners.
WITH last AS (SELECT MAX(app_ts) AS t FROM applications)
SELECT s.application_id, a.app_ts, a.branch_id, a.agent_id, a.channel, a.product_type,
       ROUND(a.financing_amount / 1e3, 0) AS financing_k,
       ROUND(s.hybrid_score, 3) AS hybrid_score, s.rule_score, s.decision, s.rules_fired,
       (SELECT string_agg(DISTINCT h.owner, ', ' ORDER BY h.owner) FROM rule_hits h WHERE h.application_id = s.application_id) AS owners
FROM application_scores s
JOIN applications a USING (application_id)
CROSS JOIN last
WHERE s.alert AND a.app_ts >= last.t - INTERVAL 30 DAY
ORDER BY s.hybrid_score DESC, s.application_id;
