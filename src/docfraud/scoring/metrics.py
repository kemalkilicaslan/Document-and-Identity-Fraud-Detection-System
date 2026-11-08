"""Fraud-team metrics: alert-capacity curves, rule effectiveness and PSI."""
from __future__ import annotations

import numpy as np
import pandas as pd


def capacity_curve(y: np.ndarray, score: np.ndarray, loss: np.ndarray,
                   rates=np.round(np.arange(0.005, 0.205, 0.005), 3)) -> pd.DataFrame:
    """For each alert rate (share of applications sent to review): threshold,
    precision, recall and the share of fraud loss that would be stopped."""
    order = np.argsort(-score, kind="stable")
    y, score, loss = y[order], score[order], loss[order]
    rows = []
    for r in rates:
        k = max(1, int(round(len(y) * r)))
        rows.append(dict(alert_rate=r, threshold=score[k - 1], alerts=k, tp=int(y[:k].sum()),
                         precision=y[:k].mean(), recall=y[:k].sum() / max(1, y.sum()),
                         loss_caught=loss[:k].sum(), loss_recall=loss[:k].sum() / max(1, loss.sum())))
    return pd.DataFrame(rows)


def rule_effectiveness(hits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per rule: volume, precision, recall, loss caught, and unique catches
    (frauds no other rule caught) — the input for pruning and tuning."""
    lab = labels.set_index("application_id")
    h = hits.join(lab[["is_fraud", "loss_amount"]], on="application_id")
    n_fraud, total_loss = lab.is_fraud.sum(), lab.loss_amount.sum()
    fired_per_app = h.groupby("application_id").rule_id.nunique()
    h["only_rule"] = h.application_id.map(fired_per_app) == 1
    g = h.groupby("rule_id")
    out = pd.DataFrame({
        "alerts": g.size(),
        "tp": g.is_fraud.sum(),
        "precision": g.is_fraud.mean(),
        "recall": g.is_fraud.sum() / n_fraud,
        "loss_caught_share": g.apply(lambda d: d.loc[d.is_fraud == 1, "loss_amount"].sum()) / total_loss,
        "unique_tp": g.apply(lambda d: int((d.only_rule & (d.is_fraud == 1)).sum())),
        "fp_per_tp": g.apply(lambda d: (d.is_fraud == 0).sum() / max(1, d.is_fraud.sum())),
    })
    return out.sort_values("tp", ascending=False)


def greedy_rule_order(hits: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add rules one at a time, each time the one with the most *additional*
    fraud caught. Rules at the bottom add almost nothing on top of the rest."""
    fraud = set(labels.loc[labels.is_fraud == 1, "application_id"])
    sets = {r: set(d.application_id) for r, d in hits.groupby("rule_id")}
    caught, alerts, rows = set(), set(), []
    remaining = dict(sets)
    while remaining:
        best = max(remaining, key=lambda r: (len((remaining[r] & fraud) - caught), -len(remaining[r] - alerts)))
        add = (remaining[best] & fraud) - caught
        caught |= remaining[best] & fraud
        alerts |= remaining[best]
        rows.append(dict(rule_id=best, added_tp=len(add), cum_recall=len(caught) / max(1, len(fraud)),
                         cum_alerts=len(alerts), cum_precision=len(caught) / max(1, len(alerts))))
        del remaining[best]
    return pd.DataFrame(rows)


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index with quantile bins from the expected sample."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / max(1, len(actual))
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))
