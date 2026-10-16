"""Step 07: synthetic applications; ID images come from the SIDTD test split only."""
import pandas as pd

from _bootstrap import done, step

from docfraud.config import load_config, path, seed_everything
from docfraud.data.synthetic import generate

step("07 · Synthetic applications")
cfg = load_config()
seed = seed_everything()

docs = pd.read_parquet(path("processed") / "doc_scores.parquet")
pool = docs[docs.split == "test"]
ocr_p = path("processed") / "ocr_checks.parquet"
if ocr_p.exists():
    ocr = pd.read_parquet(ocr_p)[["image_path", "mrz_checkdigit_fail", "mrz_viz_mismatch", "date_logic_fail"]]
    pool = pool.merge(ocr, on="image_path")
else:
    print("[warn] ocr_checks.parquet not found; OCR flags will be empty")
print(f"document pool: {len(pool)} images ({pool.label.sum()} forged)")

tables = generate(cfg, pool, seed)
out = path("synthetic")
for name, df in tables.items():
    df.to_parquet(out / f"{name}.parquet", index=False)
    print(f"{name:14s} {len(df):>8,} rows")

lab = tables["fraud_labels"]
print(f"\nfraud rate: {lab.is_fraud.mean():.2%}")
print(lab[lab.is_fraud == 1].groupby("scenario").agg(apps=("is_fraud", "size"), loss_try=("loss_amount", "sum"))
      .sort_values("apps", ascending=False).round(0))
done()
