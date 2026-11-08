"""Step 12: management report and fraud-ring case study (Markdown + HTML)."""
import duckdb
import markdown
import matplotlib
import networkx as nx
import pandas as pd

from _bootstrap import done, step

from docfraud.config import load_config, path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINKS = {"PHONE": "phone", "IBAN": "iban", "ADDRESS": "address", "DEVICE": "device_id"}
ONLINE = ("WEB", "MOBILE")

CSS = """<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;
color:#1f2933;line-height:1.55}table{border-collapse:collapse;margin:1rem 0;font-size:.9rem}th,td{border:1px solid #d5dbe1;
padding:.35rem .6rem;text-align:right}th{background:#f1f4f7}td:first-child,th:first-child{text-align:left}
img{max-width:100%;border:1px solid #e4e7eb;margin:.5rem 0}h1,h2{color:#102a43}code{background:#f1f4f7;padding:0 .25rem}</style>"""


def T(name):
    return pd.read_csv(path("tables") / name)


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def money(v):
    return f"{v / 1e6:,.1f}M TRY"


def md_table(df, floatfmt=3):
    df = df.copy()
    for c in df.select_dtypes("float").columns:
        df[c] = df[c].round(floatfmt)
    return df.to_markdown(index=False)


def write(name, text):
    (path("reports") / f"{name}.md").write_text(text, encoding="utf-8")
    html = markdown.markdown(text, extensions=["tables"])
    (path("reports") / f"{name}.html").write_text(
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{name}</title>{CSS}</head><body>{html}</body></html>",
        encoding="utf-8")


if __name__ == "__main__":
    step("12 · Reports")
    cfg = load_config()
    con = duckdb.connect(str(path("db")), read_only=True)
    fig_dir = path("figures")

    trend = T("sql_09_fraud_trend.csv")
    piv = trend.pivot_table(index="app_month", columns="scenario", values="frauds", fill_value=0)
    ax = piv.plot(kind="bar", stacked=True, figsize=(11, 4), width=0.85, colormap="tab10")
    ax.set(title="Monthly fraud applications by scenario", xlabel="", ylabel="Applications")
    ax.set_xticklabels([m[:7] for m in piv.index], rotation=60, fontsize=7)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "fraud_trend.png", dpi=150)
    plt.close()

    eff = T("rule_effectiveness.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(eff.recall * 100, eff.precision * 100, s=20 + eff.alerts / eff.alerts.max() * 600, alpha=.6)
    for r in eff.itertuples():
        ax.annotate(r.rule_id.split("_")[0], (r.recall * 100, r.precision * 100), fontsize=8)
    ax.set(xlabel="Recall (%)", ylabel="Precision (%)", title="Rule effectiveness (bubble size = alert volume)")
    plt.tight_layout()
    plt.savefig(fig_dir / "rule_effectiveness.png", dpi=150)
    plt.close()

    ring_id = con.execute("""SELECT ring_id FROM fraud_labels WHERE ring_id IS NOT NULL
                             GROUP BY 1 ORDER BY COUNT(*) DESC, SUM(loss_amount) DESC LIMIT 1""").fetchone()[0]
    ring = con.execute(f"""
        SELECT a.application_id, a.app_ts, a.tckn, a.first_name || ' ' || a.last_name AS name, a.branch_id, a.agent_id,
               a.channel, a.phone, a.iban, a.address, a.device_id, a.financing_amount, s.rule_score, s.hybrid_score,
               s.alert, s.rules_fired, c.status, l.loss_amount, d.doc_score
        FROM applications a JOIN fraud_labels l USING (application_id) JOIN application_scores s USING (application_id)
        JOIN contracts c USING (application_id) JOIN documents d USING (application_id)
        WHERE l.ring_id = '{ring_id}' ORDER BY a.app_ts
    """).df()
    G = nx.Graph()
    for r in ring.itertuples():
        G.add_node(r.application_id, kind="app", alert=r.alert)
        for kind, col in LINKS.items():
            val = getattr(r, col)
            if kind == "DEVICE" and r.channel not in ONLINE:
                continue
            if (ring[col] == val).sum() > 1:
                G.add_node(f"{kind}:{val}", kind=kind)
                G.add_edge(r.application_id, f"{kind}:{val}")
    colors = {"app": "#2e86c1", "PHONE": "#e67e22", "IBAN": "#8e44ad", "ADDRESS": "#27ae60", "DEVICE": "#c0392b"}
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=cfg["seed"], k=0.8)
    nx.draw_networkx_edges(G, pos, alpha=.4, ax=ax)
    for kind, col in colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        edge = ["#000" if G.nodes[n].get("alert") else col for n in nodes] if kind == "app" else col
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=col, edgecolors=edge, linewidths=2,
                               node_size=260 if kind == "app" else 160, label=kind, ax=ax)
    nx.draw_networkx_labels(G, pos, {n: n[-4:] for n, d in G.nodes(data=True) if d["kind"] == "app"}, font_size=6, ax=ax)
    ax.legend(fontsize=8)
    ax.set_title(f"{ring_id}: applications and shared attributes (black border = alert raised)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "ring_network.png", dpi=150)
    plt.close()

    tot = con.execute("""SELECT COUNT(*) n, SUM(is_fraud) f, SUM(loss_amount) loss FROM fraud_labels""").fetchone()
    scen = con.execute("""SELECT scenario, COUNT(*) n, SUM(loss_amount) loss FROM fraud_labels WHERE is_fraud = 1
                          GROUP BY 1 ORDER BY loss DESC""").df()
    comp = T("scoring_method_comparison.csv")
    hyb, rul = comp.set_index("method").loc["Hybrid"], comp.set_index("method").loc["Rules"]
    doc = T("doc_model_metrics.csv").set_index("metric").value
    cand = T("monitoring_candidate_rule_backtest.csv")
    kri = T("monitoring_kri.csv")
    last_kri = kri[kri.month == kri.month.max()][["kri", "value", "status"]]
    greedy = T("rule_greedy_order.csv")
    weak = greedy[greedy.added_tp <= max(2, int(0.01 * tot[1]))].rule_id.tolist()
    agents = T("sql_05_agent_anomaly.csv").head(5)
    clusters = T("sql_04_shared_attribute_clusters.csv")
    ocr = T("ocr_check_performance.csv") if (path("tables") / "ocr_check_performance.csv").exists() else pd.DataFrame()
    by_scn = T("scoring_recall_by_scenario.csv")
    cap = cfg["scoring"]["review_capacity_pct"]
    new_start = cfg["synthetic"]["new_fraud_type_start"]

    hard = T("doc_model_by_tampered_field.csv").head(3)
    hard_fields = ", ".join(f"{r.field} ({pct(r.caught, 0)})" for r in hard.itertuples())
    by_ct = T("doc_model_by_ctype.csv").set_index("ctype").recall
    crop_recall, inpaint_recall = pct(by_ct.get("Crop_and_Replace", float("nan"))), pct(by_ct.get("Inpaint_and_Rewrite", float("nan")))
    gain_recall = f"{100 * (hyb.recall - rul.recall):+.1f}"

    report = f"""# Fraud Management Report

*Period: {cfg['synthetic']['start_date']} to {cfg['synthetic']['end_date']} · Data: synthetic applications and SIDTD ID images*

## 1. Executive summary

- **{tot[0]:,} applications**, of which **{int(tot[1]):,} ({pct(tot[1] / tot[0], 2)}) were confirmed fraud**, with a total fraud loss of **{money(tot[2])}**.
- The largest loss comes from **{scen.iloc[0].scenario}** ({money(scen.iloc[0].loss)}), followed by **{scen.iloc[1].scenario}** ({money(scen.iloc[1].loss)}).
- With analysts reviewing **{pct(cap, 0)}** of applications, the hybrid score (rules + model) catches **{pct(hyb.recall)}** of fraud applications and **{pct(hyb.loss_recall)}** of fraud loss in the out-of-time test (rules alone: {pct(rul.recall)} / {pct(rul.loss_recall)}). Alert precision is **{pct(hyb.precision)}**.
- From **{new_start}** a new pattern appears: bursts of applications under different identities from one mobile device (*device farm*). The production rules catch **{pct(cand.iloc[0].device_farm_recall)}** of them; the candidate rules raise this to **{pct(cand.iloc[1].device_farm_recall)}**.
- The document forgery model catches **{pct(doc.get('test_recall_at_threshold', float('nan')))}** of forged documents at the threshold that sends 5% of genuine documents to review (ROC-AUC {doc.get('doc_roc_auc', float('nan')):.3f}).

![Monthly fraud trend](figures/fraud_trend.png)

## 2. Fraud and loss by scenario

{md_table(scen.assign(loss=(scen.loss / 1e6).round(2)).rename(columns={'scenario': 'Scenario', 'n': 'Applications', 'loss': 'Loss (M TRY)'}))}

Recall by scenario in the test period (same capacity):

{md_table(by_scn)}

## 3. Score and threshold performance (out-of-time test)

{md_table(comp[['method', 'roc_auc', 'pr_auc', 'alert_rate', 'precision', 'recall', 'loss_recall']])}

![Capacity curves](figures/capacity_curves.png)

The threshold is set by analyst capacity. At this alert volume the hybrid score's recall differs from the rules by {gain_recall} points; over all thresholds it ranks fraud better (PR-AUC {hyb.pr_auc:.3f} against {rul.pr_auc:.3f}). The model combines signals that stay below every single rule threshold (agent history, income and down-payment ratios), while the rules keep alerts explainable.

## 4. Rule effectiveness

{md_table(eff.rename(columns={'Unnamed: 0': 'rule_id'})[['rule_id', 'alerts', 'tp', 'precision', 'recall', 'loss_caught_share', 'unique_tp', 'fp_per_tp']])}

![Rule effectiveness](figures/rule_effectiveness.png)

- **Rules to review:** {', '.join(weak) if weak else 'none'}. They add almost no fraud on top of the other rules (`rule_greedy_order.csv`) and should be re-tuned, merged or retired.
- Rules with a high `fp_per_tp` should contribute to the score rather than trigger a decline on their own.

## 5. Document and identity checks

{md_table(T('doc_model_metrics.csv'))}

{('Flag rates of the OCR consistency checks on genuine and forged documents:' + chr(10) + chr(10) + md_table(ocr)) if len(ocr) else ''}

**Root cause:** the model misses edits most often in {hard_fields} (`doc_model_by_tampered_field.csv`). Recall is {crop_recall} on Crop_and_Replace and {inpaint_recall} on Inpaint_and_Rewrite forgeries. A pasted field comes from another genuine document, so its texture carries little trace; this type needs a cross-check of the field content against the MRZ or a reference database. The OCR checks flag genuine documents more often than forged ones, so they belong on the analyst screen, not in the rule set.

## 6. Early-warning indicators (latest month)

{md_table(last_kri)}

![Monitoring](figures/monitoring_recall_psi.png)

## 7. Organised fraud and agent risk

- {len(clusters):,} phone, IBAN, device or guarantor values are shared by 3 or more identities. The largest cluster holds {int(clusters.identities.max()) if len(clusters) else 0} identities (`sql_04_shared_attribute_clusters.csv`).
- Agents with the most unusual early-cancellation rates:

{md_table(agents)}

## 8. Recommended actions

| # | Action | Owner | Expected effect |
|---|---|---|---|
| 1 | Promote candidate rule `R16_DEVICE_FARM` | IT Security, Operations | Device-farm recall {pct(cand.iloc[0].device_farm_recall)} → {pct(cand.iloc[1].device_farm_recall)} |
| 2 | Re-screen earlier applications when a phone or IBAN becomes shared | Internal Control | Catches the first members of a ring (see case study) |
| 3 | Defer commission for agents with an early-cancellation z-score above 3 until the cancellation window closes | Sales, HR | Stops commission loss from agent fraud |
| 4 | Verify income documents at the source before allocation when the down payment is high relative to income | Underwriting | Cuts loss in the costliest scenario (income fraud) |
| 5 | Re-tune or retire weak rules ({', '.join(weak[:3]) if weak else '-'}) | Fraud Management | Fewer alerts, higher precision |
| 6 | Require video or in-person identity verification when the document score is high | Operations, Legal | Cuts loss from forged documents |

*All figures come from synthetic data; the method and queries apply to real data unchanged.*
"""
    write("management_report", report)

    first_alert = ring[ring.alert].app_ts.min()
    missed = ring[~ring.alert]
    shared = {k: int(ring[k].duplicated(keep=False).sum()) for k in ("phone", "iban", "address")}
    online = ring[ring.channel.isin(ONLINE)]
    shared["device_id"] = int(online.device_id.duplicated(keep=False).sum())
    first_alert = first_alert.strftime("%Y-%m-%d %H:%M") if pd.notna(first_alert) else "none"
    no_rule = int((missed.rule_score == 0).sum())
    timeline = ring.assign(app_ts=ring.app_ts.dt.strftime("%Y-%m-%d %H:%M"), hybrid_score=ring.hybrid_score.round(3),
                           doc_score=ring.doc_score.round(2), financing_amount=(ring.financing_amount / 1e3).round(0))[
        ["app_ts", "application_id", "branch_id", "agent_id", "channel", "financing_amount", "doc_score",
         "rule_score", "hybrid_score", "alert", "rules_fired", "status"]]
    case = f"""# Case Study: {ring_id}

## Summary

- **{len(ring)} applications** through {ring.branch_id.nunique()} branch(es) and {ring.agent_id.nunique()} agent(s) within {(ring.app_ts.max() - ring.app_ts.min()).days} days.
- Shared attributes: phone on {shared['phone']}, IBAN on {shared['iban']}, address on {shared['address']} and device on {shared['device_id']} applications.
- Total loss: **{money(ring.loss_amount.sum())}**. Contract status: {', '.join(f'{k} {v}' for k, v in ring.status.value_counts().items())}.
- Alerts at application time: **{int(ring.alert.sum())}/{len(ring)}**. First alert: {first_alert}.

![Ring network](figures/ring_network.png)

## Timeline

{md_table(timeline)}

## Root cause

1. **The first applications looked clean.** The shared phone and IBAN rules only look at earlier applications. {int((~ring.head(3).alert).sum())} of the first 3 applications raised no alert because their phone and IBAN had not yet been seen with another identity.
2. **Signals below the thresholds.** No rule fired on {no_rule} of the {len(missed)} missed applications. Their average document score was {missed.doc_score.mean() if len(missed) else 0:.2f}, and declared income was above the occupation median but below the 3x threshold.
3. **One branch, few agents.** Most applications came through a single branch and a small number of agents. The agent rule only looks at cancellation history, so it did not see this pattern.

## Missed applications

{md_table(missed[['application_id', 'app_ts', 'rule_score', 'hybrid_score', 'rules_fired']].assign(hybrid_score=missed.hybrid_score.round(3)))}

## Actions

| Action | Owner |
|---|---|
| Suspend the ring's active contracts before allocation and request identity verification | Operations, Underwriting |
| Investigate the holders of the shared phones and IBANs and the agent; prepare a criminal complaint if confirmed | Internal Control, Legal |
| Re-screen earlier applications when a new link appears | Fraud Management, IT |
| Add a cluster rule: same branch, 30 days, 3+ shared attributes | Fraud Management |
"""
    write("case_study", case)
    con.close()
    print("reports/management_report.html, reports/case_study.html written")
    done()
