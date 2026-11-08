"""Step 08: load the synthetic tables into DuckDB and build the point-in-time feature table."""
import duckdb
import pandas as pd

from _bootstrap import ROOT, done, step

from docfraud.config import path
from docfraud.data.synthetic import OCCUPATIONS

step("08 · DuckDB")
db_path = path("db")
db_path.unlink(missing_ok=True)
con = duckdb.connect(str(db_path))

for p in sorted(path("synthetic").glob("*.parquet")):
    con.execute(f"CREATE TABLE {p.stem} AS SELECT * FROM read_parquet('{p}')")
    print(f"{p.stem:14s} {con.execute(f'SELECT COUNT(*) FROM {p.stem}').fetchone()[0]:>9,}")

ref = pd.DataFrame({"occupation": list(OCCUPATIONS), "median_income": list(OCCUPATIONS.values())})
con.execute("CREATE TABLE occupation_reference AS SELECT * FROM ref")

for f in ("00_macros.sql", "01_features.sql"):
    con.execute((ROOT / "sql" / f).read_text(encoding="utf-8"))
n = con.execute("SELECT COUNT(*) FROM application_features").fetchone()[0]
print(f"application_features {n:,} rows")
print(con.execute("""
    SELECT l.is_fraud, AVG(tckn_invalid::INT) tckn_invalid, AVG((phone_other_tckn_90d > 0)::INT) shared_phone,
           AVG(doc_score) doc_score, AVG(income_to_occupation_median) income_ratio
    FROM application_features f JOIN fraud_labels l USING (application_id) GROUP BY 1 ORDER BY 1
""").df().round(3).to_string(index=False))
con.close()
done()
