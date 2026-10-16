"""Fraud analyst dashboard: streamlit run src/docfraud/dashboard.py (after run_all.sh)."""
from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from docfraud.config import ROOT, path
from docfraud.rules.engine import load_rules

st.set_page_config(page_title="Fraud Monitoring", layout="wide")


@st.cache_resource
def db():
    return duckdb.connect(str(path("db")), read_only=True)


@st.cache_data
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return db().execute(sql, list(params)).df()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    p = path("tables") / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


RULES = {r["id"]: r for r in load_rules(ROOT / "configs" / "rules.yaml")["rules"]}

st.title("Document and Identity Fraud Monitoring")
tab_over, tab_queue, tab_rules, tab_mon, tab_doc = st.tabs(
    ["Overview", "Alert Queue and Case", "Rule Performance", "Monitoring", "Document Model"])

with tab_over:
    k = q("""SELECT COUNT(*) n, SUM(l.is_fraud) f, SUM(l.loss_amount) loss, SUM(s.alert::INT) alerts,
                    SUM((s.alert AND l.is_fraud = 1)::INT) tp
             FROM fraud_labels l JOIN application_scores s USING (application_id)""").iloc[0]
    c = st.columns(5)
    c[0].metric("Applications", f"{int(k.n):,}")
    c[1].metric("Fraud rate", f"{100 * k.f / k.n:.2f}%")
    c[2].metric("Fraud loss", f"{k.loss / 1e6:,.1f}M TRY")
    c[3].metric("Alerts", f"{int(k.alerts):,}")
    c[4].metric("Alert precision", f"{100 * k.tp / max(1, k.alerts):.1f}%")
    trend = q("""SELECT CAST(date_trunc('month', a.app_ts) AS DATE) AS app_month, l.scenario, COUNT(*) AS n
                 FROM applications a JOIN fraud_labels l USING (application_id) WHERE l.is_fraud = 1 GROUP BY 1, 2""")
    st.subheader("Monthly fraud applications by scenario")
    st.bar_chart(trend.pivot_table(index="app_month", columns="scenario", values="n", fill_value=0))
    st.subheader("Segment risk")
    st.dataframe(table("sql_01_segmentation.csv"), width="stretch", hide_index=True)

with tab_queue:
    queue = table("sql_10_alert_queue.csv")
    st.caption("Alerts of the last 30 days, highest hybrid score first")
    st.dataframe(queue, width="stretch", hide_index=True, height=280)
    ids = queue.application_id.tolist() if len(queue) else []
    app_id = st.selectbox("Select a case", ids) if ids else st.text_input("Application ID", "APP0000001")
    if app_id:
        a = q("SELECT * FROM applications WHERE application_id = ?", (app_id,))
        if a.empty:
            st.warning("Application not found")
        else:
            a = a.iloc[0]
            f = q("SELECT * FROM application_features WHERE application_id = ?", (app_id,)).iloc[0]
            s = q("SELECT * FROM application_scores WHERE application_id = ?", (app_id,)).iloc[0]
            d = q("SELECT * FROM documents WHERE application_id = ?", (app_id,)).iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybrid score", f"{s.hybrid_score:.3f}")
            c2.metric("Rule score", f"{s.rule_score:.0f}", s.decision)
            c3.metric("Model score", f"{s.ml_score:.3f}")
            left, right = st.columns(2)
            with left:
                st.markdown("**Fired rules**")
                fired = [r for r in s.rules_fired.split(",") if r]
                st.dataframe(pd.DataFrame([{"rule": r, "description": RULES[r]["description"], "weight": RULES[r]["weight"],
                                            "owner": RULES[r]["owner"]} for r in fired if r in RULES]),
                             hide_index=True, width="stretch")
                st.markdown("**Application**")
                st.json({k: str(a[k]) for k in ["app_ts", "channel", "branch_id", "agent_id", "product_type", "plan_type",
                                                "financing_amount", "down_payment", "declared_income", "occupation",
                                                "phone", "iban", "device_id"]})
                st.markdown("**Link and velocity features**")
                st.json({k: (float(f[k]) if pd.notna(f[k]) else None) for k in
                         ["same_tckn_other_branch_30d", "phone_other_tckn_90d", "iban_other_tckn_90d",
                          "address_other_tckn_30d", "device_other_tckn_7d", "guarantor_other_apps_180d",
                          "agent_early_cancel_rate_180d", "income_to_occupation_median", "down_payment_months_income"]})
            with right:
                st.markdown(f"**ID document** · forgery score {d.doc_score:.2f} · most suspicious field: `{d.doc_top_field}`")
                img_p = ROOT / d.doc_image_path
                if img_p.exists():
                    img = Image.open(img_p).convert("RGB")
                    fs_p = path("processed") / "field_scores.parquet"
                    man = pd.read_parquet(path("processed") / "manifest.parquet", columns=["image_path", "base_doc"])
                    base = man.loc[man.image_path == d.doc_image_path, "base_doc"]
                    if fs_p.exists() and len(base):
                        fs = pd.read_parquet(fs_p, filters=[("image_path", "==", d.doc_image_path)])
                        boxes = pd.read_parquet(path("processed") / "fields.parquet",
                                                filters=[("doc_id", "==", base.iloc[0])]).set_index("field")
                        draw = ImageDraw.Draw(img)
                        for r in fs.itertuples():
                            if r.field in boxes.index:
                                b = boxes.loc[r.field]
                                col = (220, 38, 38) if r.field_score >= 0.5 else (234, 179, 8) if r.field_score >= 0.2 else (34, 197, 94)
                                draw.rectangle([b.x, b.y, b.x + b.w, b.y + b.h], outline=col, width=5)
                    st.image(img, caption="Red: suspicious field, yellow: elevated, green: clean", width="stretch")

with tab_rules:
    eff = table("rule_effectiveness.csv")
    st.subheader("Rule effectiveness (whole period)")
    st.dataframe(eff, width="stretch", hide_index=True)
    st.subheader("Greedy rule order: added recall per rule")
    st.dataframe(table("rule_greedy_order.csv"), width="stretch", hide_index=True)
    monthly = table("sql_08_rule_performance_monthly.csv")
    if len(monthly):
        sel = st.multiselect("Monthly precision", sorted(monthly.rule_id.unique()), default=list(monthly.rule_id.unique()[:3]))
        st.line_chart(monthly[monthly.rule_id.isin(sel)].pivot_table(index="app_month", columns="rule_id", values="precision_pct"))
    st.subheader("Methods at the same alert capacity (out-of-time)")
    st.dataframe(table("scoring_method_comparison.csv"), width="stretch", hide_index=True)
    fig = path("figures") / "capacity_curves.png"
    if fig.exists():
        st.image(str(fig))

with tab_mon:
    kri = table("monitoring_kri.csv")
    if len(kri):
        piv = kri.pivot(index="month", columns="kri", values="status")
        colors = {"RED": "background-color:#fecaca", "AMBER": "background-color:#fde68a", "GREEN": "background-color:#bbf7d0"}
        st.subheader("Key risk indicators")
        st.dataframe(piv.style.map(lambda v: colors.get(v, "")), width="stretch")
    fig = path("figures") / "monitoring_recall_psi.png"
    if fig.exists():
        st.image(str(fig))
    st.subheader("Candidate-rule back-test")
    st.dataframe(table("monitoring_candidate_rule_backtest.csv"), width="stretch", hide_index=True)
    st.subheader("Shared-attribute clusters")
    st.dataframe(table("sql_04_shared_attribute_clusters.csv"), width="stretch", hide_index=True)
    st.subheader("Agent anomalies")
    st.dataframe(table("sql_05_agent_anomaly.csv").head(20), width="stretch", hide_index=True)

with tab_doc:
    st.dataframe(table("doc_model_metrics.csv"), hide_index=True)
    c1, c2 = st.columns(2)
    c1.dataframe(table("doc_model_by_ctype.csv"), hide_index=True)
    c2.dataframe(table("doc_model_by_doc_type.csv"), hide_index=True)
    st.dataframe(table("doc_model_by_tampered_field.csv"), hide_index=True)
    fig = path("figures") / "doc_model_curves.png"
    if fig.exists():
        st.image(str(fig))
    st.subheader("OCR checks")
    st.dataframe(table("ocr_check_performance.csv"), hide_index=True)
