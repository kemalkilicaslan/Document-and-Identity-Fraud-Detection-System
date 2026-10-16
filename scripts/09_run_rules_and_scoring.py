"""Step 09: rule engine, LightGBM, hybrid score and capacity-based alert threshold.

LightGBM trains on applications before `scoring.train_until` minus the maturity
window, using only fraud labels confirmed by the cut-off.
"""
import json

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from _bootstrap import ROOT, done, step

from docfraud.config import load_config, path, seed_everything
from docfraud.rules.engine import load_rules, score
from docfraud.scoring.metrics import capacity_curve, greedy_rule_order, rule_effectiveness

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FEATURES = ["age", "tckn_invalid", "guarantor_tckn_invalid", "income_to_occupation_median", "installment_to_income",
            "down_payment_ratio", "down_payment_months_income", "allocation_month", "is_night", "is_online",
            "email_many_digits", "month_end", "same_tckn_apps_30d", "same_tckn_other_branch_30d",
            "phone_other_tckn_90d", "iban_other_tckn_90d", "address_other_tckn_30d", "device_other_tckn_7d",
            "ipnet_other_tckn_7d", "guarantor_other_apps_180d", "agent_prior_apps_180d", "agent_apps_30d",
            "agent_early_cancel_rate_180d", "doc_score", "mrz_checkdigit_fail", "mrz_viz_mismatch",
            "date_logic_fail", "financing_amount", "channel", "product_type", "plan_type"]
CATEGORICAL = ["channel", "product_type", "plan_type"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

if __name__ == "__main__":
    step("09 · Rules and scoring")
    cfg = load_config()
    sc = cfg["scoring"]
    seed = seed_everything()
    con = duckdb.connect(str(path("db")))
    rules_cfg = load_rules(ROOT / "configs" / "rules.yaml")

    rule_scores, hits = score(con, rules_cfg)
    con.execute("CREATE OR REPLACE TABLE rule_hits AS SELECT * FROM hits")
    print(f"{len(hits):,} rule hits on {rule_scores.n_rules.gt(0).sum():,} applications")

    df = con.execute("""
        SELECT f.*, l.is_fraud, l.scenario, l.loss_amount, l.confirmed_date
        FROM application_features f JOIN fraud_labels l USING (application_id)
    """).df().merge(rule_scores, on="application_id")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    for c in FEATURES:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    cut = pd.Timestamp(sc["train_until"])
    train = df[df.app_ts < cut - pd.Timedelta(days=sc["maturity_days"])].copy()
    test = df[df.app_ts >= cut].copy()
    train["y_known"] = ((train.is_fraud == 1) & (pd.to_datetime(train.confirmed_date) < cut)).astype(int)
    print(f"train: {len(train):,} apps (until {cut.date()} minus {sc['maturity_days']} days maturity), "
          f"{train.is_fraud.sum()} frauds of which {train.y_known.sum()} confirmed by "
          f"{cut.date()} | test: {len(test):,} apps, {test.is_fraud.sum()} frauds")

    t_split = train.app_ts.quantile(0.8)
    fit, cal = train[train.app_ts < t_split], train[train.app_ts >= t_split]
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  scale_pos_weight=5.0, verbose=-1, seed=seed)
    dfit = lgb.Dataset(fit[FEATURES], fit.y_known, categorical_feature=CATEGORICAL)
    dcal = lgb.Dataset(cal[FEATURES], cal.y_known, categorical_feature=CATEGORICAL, reference=dfit)
    model = lgb.train(params, dfit, num_boost_round=2000, valid_sets=[dcal],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    platt = LogisticRegression().fit(logit(model.predict(cal[FEATURES]))[:, None], cal.y_known)
    df["ml_score"] = platt.predict_proba(logit(model.predict(df[FEATURES]))[:, None])[:, 1]
    df["hybrid_score"] = 0.5 * df.rule_score / 100 + 0.5 * df.ml_score
    test = df[df.app_ts >= cut]

    model.save_model(str(path("models") / "app_lgbm.txt"))
    (path("models") / "app_lgbm_meta.json").write_text(json.dumps(
        {"features": FEATURES, "categorical": CATEGORICAL, "platt_a": float(platt.coef_[0][0]),
         "platt_b": float(platt.intercept_[0]), "categories": {c: list(df[c].cat.categories) for c in CATEGORICAL}}))
    imp = pd.DataFrame({"feature": FEATURES, "gain": model.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(path("tables") / "scoring_feature_importance.csv", index=False)

    cap = sc["review_capacity_pct"]
    summary, curves = [], {}
    for name, col in [("Rules", "rule_score"), ("LightGBM", "ml_score"), ("Hybrid", "hybrid_score")]:
        y, s, loss = test.is_fraud.values, test[col].values, test.loss_amount.values
        cur = capacity_curve(y, s + 1e-9 * np.random.default_rng(seed).random(len(s)), loss)
        curves[name] = cur
        at = cur.iloc[(cur.alert_rate - cap).abs().argmin()]
        summary.append(dict(method=name, roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
                            alert_rate=at.alert_rate, threshold=at.threshold, precision=at.precision,
                            recall=at.recall, loss_recall=at.loss_recall, loss_caught=at.loss_caught))
    summary = pd.DataFrame(summary)
    summary.to_csv(path("tables") / "scoring_method_comparison.csv", index=False)
    print(summary.round(3).to_string(index=False))
    pd.concat([c.assign(method=m) for m, c in curves.items()]).to_csv(path("tables") / "scoring_capacity_curves.csv", index=False)

    thr = float(summary.set_index("method").at["Hybrid", "threshold"])
    df["alert"] = df.hybrid_score >= thr
    by_scn = (df[(df.app_ts >= cut) & (df.is_fraud == 1)].assign(
        rule_alert=lambda d: d.rule_score >= summary.set_index("method").at["Rules", "threshold"])
        .groupby("scenario").agg(frauds=("is_fraud", "size"), hybrid_recall=("alert", "mean"),
                                 rules_recall=("rule_alert", "mean")))
    by_scn.to_csv(path("tables") / "scoring_recall_by_scenario.csv")
    print("\nrecall by scenario (test period):\n", by_scn.round(3))

    labels = df[["application_id", "is_fraud", "loss_amount"]]
    eff = rule_effectiveness(hits, labels)
    eff.to_csv(path("tables") / "rule_effectiveness.csv")
    greedy = greedy_rule_order(hits, labels)
    greedy.to_csv(path("tables") / "rule_greedy_order.csv", index=False)
    print("\nrule effectiveness:\n", eff.round(3).to_string())

    out = df[["application_id", "app_ts", "rule_score", "rules_fired", "n_rules", "decision", "ml_score",
              "hybrid_score", "alert"]]
    con.execute("CREATE OR REPLACE TABLE application_scores AS SELECT * FROM out")
    (path("models") / "alert_threshold.json").write_text(json.dumps({"hybrid_threshold": thr, "capacity": cap}))
    con.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, c in curves.items():
        ax[0].plot(c.alert_rate * 100, c.recall * 100, label=m)
        ax[1].plot(c.alert_rate * 100, c.loss_recall * 100, label=m)
    for a, t in zip(ax, ["Fraud applications caught (%)", "Fraud loss prevented (%)"]):
        a.axvline(cap * 100, ls="--", c="grey")
        a.set(xlabel="Alert rate (% of applications)", ylabel=t)
        a.legend()
    fig.suptitle("Alert capacity vs. fraud caught (out-of-time test)")
    fig.tight_layout()
    fig.savefig(path("figures") / "capacity_curves.png", dpi=150)
    done()
