from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from docfraud.rules.engine import score
from docfraud.scoring.metrics import capacity_curve, psi
from docfraud.validators.tckn import generate_tckn, is_valid_tckn

ROOT = Path(__file__).resolve().parents[1]


def test_sql_tckn_macro_matches_python():
    import random
    rng = random.Random(1)
    vals = [generate_tckn(rng) for _ in range(200)] + ["10000000147", "01234567890", "123", None]
    con = duckdb.connect()
    con.execute((ROOT / "sql" / "00_macros.sql").read_text())
    df = pd.DataFrame({"t": vals})  # noqa: F841 - read by DuckDB through a replacement scan
    out = con.execute("SELECT t, COALESCE(tckn_valid(t), FALSE) v FROM df").df()
    assert (out.v == out.t.map(is_valid_tckn)).all()


def test_identity_velocity_masks_national_id():
    # Two identities that share the same masked form must still get separate pseudonymous ids
    con = duckdb.connect()
    con.execute("""CREATE TABLE applications AS SELECT * FROM (VALUES
        ('A1', '10000000146', TIMESTAMP '2026-01-01', 'S01'),
        ('A2', '10000000146', TIMESTAMP '2026-01-05', 'S02'),
        ('A3', '10099999946', TIMESTAMP '2026-02-01', 'S01'),
        ('A4', '10099999946', TIMESTAMP '2026-02-03', 'S01'),
        ('A5', '20000000254', TIMESTAMP '2026-01-01', 'S01')) t(application_id, tckn, app_ts, branch_id)""")
    con.execute("""CREATE TABLE fraud_labels AS SELECT * FROM (VALUES
        ('A1', 1), ('A2', 0), ('A3', 0), ('A4', 0), ('A5', 0)) t(application_id, is_fraud)""")
    out = con.execute((ROOT / "sql" / "analysis" / "03_identity_velocity.sql").read_text()).df()
    assert out.tckn_masked.tolist() == ["100******46", "100******46"]
    assert out.identity_id.tolist() == ["ID-0001", "ID-0002"]
    assert out.branches.tolist() == [2, 1] and "tckn" not in out.columns


def test_shared_attribute_clusters_mask_personal_values():
    con = duckdb.connect()
    rows = [(f"A{i}", f"1000000000{i}", "05000000001", "TR000000000000000000000001", "DEV-1", "WEB", "20000000001")
            for i in range(3)]
    con.execute("CREATE TABLE applications (application_id VARCHAR, tckn VARCHAR, phone VARCHAR, iban VARCHAR, "
                "device_id VARCHAR, channel VARCHAR, guarantor_tckn VARCHAR, app_ts TIMESTAMP, branch_id VARCHAR, "
                "agent_id VARCHAR)")
    con.executemany("INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?, ?, TIMESTAMP '2026-01-01', 'S01', 'T01')", rows)
    con.execute("CREATE TABLE fraud_labels AS SELECT application_id, 1 AS is_fraud, 10.0 AS loss_amount FROM applications")
    out = con.execute((ROOT / "sql" / "analysis" / "04_shared_attribute_clusters.sql").read_text()).df()
    masked = dict(zip(out.link_type, out.link_value))
    assert masked == {"DEVICE": "DEV-1", "GUARANTOR": "200******01", "IBAN": "TR00******************0001",
                      "PHONE": "0500*****01"}
    assert sorted(out.link_id) == ["LNK-0001", "LNK-0002", "LNK-0003", "LNK-0004"]


def test_rule_engine_scores_and_bands():
    con = duckdb.connect()
    con.execute("""CREATE TABLE application_features AS SELECT * FROM (VALUES
        ('A', 0.95, 0), ('B', 0.10, 3), ('C', 0.10, 0)) t(application_id, doc_score, phone_other_tckn_90d)""")
    cfg = {"decision_bands": {"DECLINE": 70, "REVIEW": 30},
           "rules": [{"id": "DOC", "condition": "doc_score >= 0.8", "weight": 45, "owner": "Operations"},
                     {"id": "PHONE", "condition": "phone_other_tckn_90d >= 2", "weight": 35, "owner": "Internal Control"}]}
    out, hits = score(con, cfg)
    out = out.set_index("application_id")
    assert out.loc["A", "decision"] == "REVIEW" and out.loc["A", "rules_fired"] == "DOC"
    assert out.loc["B", "rule_score"] == 35
    assert out.loc["C", "decision"] == "APPROVE" and out.loc["C", "n_rules"] == 0
    assert len(hits) == 2


def test_capacity_curve_and_psi():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .1])
    cur = capacity_curve(y, s, y * 100.0, rates=np.array([0.3]))
    assert cur.recall.iloc[0] == 1.0 and abs(cur.precision.iloc[0] - 2 / 3) < 1e-9
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(1, 1, size=5000)) > 0.25
