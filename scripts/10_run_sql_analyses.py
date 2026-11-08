"""Step 10: run every query in sql/analysis/ and save the results."""
import duckdb

from _bootstrap import ROOT, done, step

from docfraud.config import path

step("10 · SQL analyses")
con = duckdb.connect(str(path("db")), read_only=True)
for f in sorted((ROOT / "sql" / "analysis").glob("*.sql")):
    df = con.execute(f.read_text(encoding="utf-8")).df()
    df.to_csv(path("tables") / f"sql_{f.stem}.csv", index=False)
    print(f"\n--- {f.name}: {len(df):,} rows")
    print(df.head(6).to_string(index=False, max_colwidth=40))
con.close()
done()
