"""Step 11: monthly performance, PSI drift, KRIs and the candidate-rule back-test."""
import json

import duckdb
import matplotlib
import numpy as np
import pandas as pd

from _bootstrap import ROOT, done, step

from docfraud.config import load_config, path
from docfraud.rules.engine import load_rules, score
from docfraud.scoring.metrics import psi

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DRIFT_FEATURES = ["doc_score", "income_to_occupation_median", "down_payment_months_income",
                  "device_other_tckn_7d", "phone_other_tckn_90d", "agent_early_cancel_rate_180d"]


def status(v, amber, red):
    return "RED" if v >= red else "AMBER" if v >= amber else "GREEN"


if __name__ == "__main__":
    step("11 · Monitoring and early warning")
    cfg = load_config()
    con = duckdb.connect(str(path("db")))
    thr = json.loads((path("models") / "alert_threshold.json").read_text())["hybrid_threshold"]
    df = con.execute("""
        SELECT f.*, s.hybrid_score, s.rule_score, s.alert, l.is_fraud, l.scenario, l.loss_amount
        FROM application_features f JOIN application_scores s USING (application_id)
        JOIN fraud_labels l USING (application_id)
    """).df()
    df["month"] = pd.to_datetime(df.app_month).dt.strftime("%Y-%m")
    base = df[df.month.isin(cfg["scoring"]["baseline_months"])]

    rows = []
    for m, g in df.groupby("month"):
        r = dict(month=m, applications=len(g), frauds=int(g.is_fraud.sum()), alerts=int(g.alert.sum()),
                 alert_rate=g.alert.mean(), precision=g.loc[g.alert, "is_fraud"].mean() if g.alert.any() else np.nan,
                 recall=g.loc[g.is_fraud == 1, "alert"].mean() if g.is_fraud.any() else np.nan,
                 psi_score=psi(base.hybrid_score, g.hybrid_score))
        for f in DRIFT_FEATURES:
            r[f"psi_{f}"] = psi(base[f].astype(float), g[f].astype(float))
        rows.append(r)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(path("tables") / "monitoring_monthly.csv", index=False)
    print(monthly[["month", "applications", "frauds", "alerts", "precision", "recall", "psi_score"]].round(3).to_string(index=False))

    rec = (df[df.is_fraud == 1].groupby(["month", "scenario"]).alert.mean().unstack())
    rec.to_csv(path("tables") / "monitoring_recall_by_scenario.csv")

    kri = []
    agent_rows = []
    for m in sorted(df.month.unique()):
        m_end = (pd.Timestamp(m) + pd.offsets.MonthEnd(0)).date()
        a = con.execute("""
            SELECT a.agent_id, COUNT(*) n,
                   AVG(CASE WHEN c.cancel_date <= ? AND c.cancel_date <= CAST(a.app_ts AS DATE) + 120 THEN 1.0 ELSE 0 END) cr
            FROM applications a JOIN contracts c USING (application_id)
            WHERE CAST(a.app_ts AS DATE) BETWEEN CAST(? AS DATE) - 180 AND ?
            GROUP BY 1 HAVING COUNT(*) >= 15
        """, [m_end, m_end, m_end]).df()
        a["z"] = (a.cr - a.cr.mean()) / (a.cr.std() or 1)
        agent_rows.append(a.assign(month=m))
    agents = pd.concat(agent_rows)
    for m, g in df.groupby("month"):
        online = g[g.is_online]
        kri += [
            dict(month=m, kri="Documents with forgery score >= 0.5 (%)", value=100 * (g.doc_score >= 0.5).mean(), amber=8, red=12),
            dict(month=m, kri="Online applications from a device seen with 2+ identities", value=int((online.device_other_tckn_7d >= 1).sum()), amber=10, red=25),
            dict(month=m, kri="Applications with a shared phone or IBAN",
                 value=int(((g.phone_other_tckn_90d >= 1) | (g.iban_other_tckn_90d >= 1)).sum()), amber=25, red=45),
            dict(month=m, kri="Agents with early-cancel rate >= 20% and z > 3",
                 value=int(((agents.month == m) & (agents.z > 3) & (agents.cr >= 0.20)).sum()), amber=1, red=2),
            dict(month=m, kri="Score PSI vs. baseline", value=float(monthly.set_index("month").at[m, "psi_score"]), amber=0.1, red=0.25),
        ]
    kri = pd.DataFrame(kri)
    kri["status"] = [status(v, a, r) for v, a, r in zip(kri.value, kri.amber, kri.red)]
    kri.to_csv(path("tables") / "monitoring_kri.csv", index=False)
    print("\nlatest KRIs:\n", kri[kri.month == kri.month.max()][["kri", "value", "status"]].round(3).to_string(index=False))

    prod = load_rules(ROOT / "configs" / "rules.yaml")
    cand = load_rules(ROOT / "configs" / "rules_candidate.yaml")
    new_start = cfg["synthetic"]["new_fraud_type_start"]
    rule_thr = prod["decision_bands"]["REVIEW"]
    before, _ = score(con, prod)
    after, cand_hits = score(con, {**prod, "rules": prod["rules"] + cand["rules"]})
    ev = df[["application_id", "app_ts", "is_fraud", "scenario", "loss_amount"]].merge(
        before[["application_id", "rule_score"]], on="application_id").merge(
        after[["application_id", "rule_score"]].rename(columns={"rule_score": "rule_score_new"}), on="application_id")
    win = ev[ev.app_ts >= pd.Timestamp(new_start)]
    res = []
    for name, col in [("Production rules", "rule_score"), ("Production + candidate rules", "rule_score_new")]:
        a = win[col] >= rule_thr
        res.append(dict(rule_set=name, alerts=int(a.sum()), alert_rate=a.mean(),
                        precision=win.loc[a, "is_fraud"].mean(), recall=win.loc[win.is_fraud == 1, col].ge(rule_thr).mean(),
                        device_farm_recall=win.loc[win.scenario == "DEVICE_FARM", col].ge(rule_thr).mean(),
                        loss_recall=win.loc[a, "loss_amount"].sum() / max(1, win.loss_amount.sum())))
    res = pd.DataFrame(res)
    res.to_csv(path("tables") / "monitoring_candidate_rule_backtest.csv", index=False)
    print(f"\ncandidate rule back-test (applications since {new_start}):\n", res.round(3).to_string(index=False))
    ch = cand_hits[cand_hits.rule_id.isin([r["id"] for r in cand["rules"]])].merge(
        df[["application_id", "is_fraud"]], on="application_id")
    ch.groupby("rule_id").is_fraud.agg(["size", "mean"]).rename(columns={"size": "alerts", "mean": "precision"}) \
        .to_csv(path("tables") / "monitoring_candidate_rule_precision.csv")
    con.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for col in rec.columns:
        ax[0].plot(rec.index, rec[col], marker="o", ms=3, label=col)
    ax[0].axvline(pd.Timestamp(new_start).strftime("%Y-%m"), ls="--", c="grey")
    ax[0].set(title="Monthly recall by scenario (hybrid score)", ylabel="Recall")
    ax[0].tick_params(axis="x", rotation=60, labelsize=7)
    ax[0].legend(fontsize=7)
    ax[1].bar(monthly.month, monthly.psi_score, color=["#c0392b" if v >= .25 else "#e67e22" if v >= .1 else "#2e86c1" for v in monthly.psi_score])
    ax[1].set(title="Score PSI (baseline: 2025 H1)", ylabel="PSI")
    ax[1].tick_params(axis="x", rotation=60, labelsize=7)
    fig.tight_layout()
    fig.savefig(path("figures") / "monitoring_recall_psi.png", dpi=150)
    done()
