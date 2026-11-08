"""Config-driven rule engine running on DuckDB.

Rules live in YAML as SQL conditions over `application_features`, so analysts
can add, tune or retire a rule without touching Python code.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import yaml


def load_rules(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rule_hits(con: duckdb.DuckDBPyConnection, rules: list[dict], table: str = "application_features") -> pd.DataFrame:
    """Long table (application_id, rule_id, weight, owner) with one row per fired rule."""
    parts = [f"SELECT application_id, '{r['id']}' AS rule_id, {int(r['weight'])} AS weight, "
             f"'{r['owner']}' AS owner FROM {table} WHERE COALESCE(({r['condition']}), FALSE)" for r in rules]
    return con.execute(" UNION ALL ".join(parts)).df()


def score(con: duckdb.DuckDBPyConnection, cfg: dict, table: str = "application_features") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rule score (0-100), decision and fired rules for every application."""
    hits = rule_hits(con, cfg["rules"], table)
    ids = con.execute(f"SELECT application_id FROM {table}").df()
    agg = hits.groupby("application_id").agg(rule_score=("weight", "sum"),
                                             rules_fired=("rule_id", lambda s: ",".join(sorted(s))),
                                             n_rules=("rule_id", "size")).reset_index()
    out = ids.merge(agg, on="application_id", how="left")
    out["rule_score"] = out.rule_score.fillna(0).clip(upper=100)
    out["n_rules"] = out.n_rules.fillna(0).astype(int)
    out["rules_fired"] = out.rules_fired.fillna("")
    bands = cfg["decision_bands"]
    out["decision"] = "APPROVE"
    out.loc[out.rule_score >= bands["REVIEW"], "decision"] = "REVIEW"
    out.loc[out.rule_score >= bands["DECLINE"], "decision"] = "DECLINE"
    return out, hits
