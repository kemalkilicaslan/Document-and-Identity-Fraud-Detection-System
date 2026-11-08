"""Step 01: SIDTD image manifest and field layouts."""
from _bootstrap import done, step

from docfraud.config import load_config, path
from docfraud.data.sidtd import build_manifest, load_fields

step("01 · SIDTD manifest")
cfg = load_config()
interim, processed = path("interim"), path("processed")

fields = load_fields(interim, cfg["sidtd"]["non_text_fields"])
fields.to_parquet(processed / "fields.parquet", index=False)
print(f"fields: {len(fields):,} rows, {fields.doc_id.nunique()} documents")

m = build_manifest(interim)
known = set(zip(fields.doc_id, fields.field))
fake = m[m.label == 1]
missing = sum(any((r.base_doc, f) not in known for f in r.tampered_fields) for r in fake.itertuples())
print(f"fake images whose tampered field has no box: {missing}")

m.to_parquet(processed / "manifest.parquet", index=False)
overview = (m.groupby(["doc_type", "source", "label"]).size().unstack(["source", "label"], fill_value=0))
overview.to_csv(path("tables") / "sidtd_overview.csv")
print(overview)
print("\nfake types:\n", m[m.label == 1].ctype.value_counts().to_string())
print(f"\nimages: {len(m):,}, groups: {m.group_id.nunique()}")
done()
