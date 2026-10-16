"""Step 06: MRZ and date-logic OCR checks, measured on genuine and forged test documents."""
from multiprocessing import Pool

import pandas as pd

from _bootstrap import ROOT, done, step

from docfraud.config import load_config, path, seed_everything
from docfraud.ocr.engine import check_document

FLAGS = ["mrz_checkdigit_fail", "mrz_viz_mismatch", "date_logic_fail"]
BOXES: dict = {}


def _init(boxes):
    global BOXES
    BOXES = boxes


def work(r):
    try:
        res = check_document(str(ROOT / r["image_path"]), r["doc_type"], BOXES[r["base_doc"]])
    except Exception as e:
        res = {"error": str(e)}
    return {"image_path": r["image_path"], **res}


if __name__ == "__main__":
    step("06 · OCR checks")
    cfg = load_config()
    seed = seed_everything()
    docs = pd.read_parquet(path("processed") / "doc_scores.parquet")
    man = pd.read_parquet(path("processed") / "manifest.parquet", columns=["image_path", "base_doc"])
    test = docs[docs.split == "test"].merge(man, on="image_path")
    n = cfg["ocr"]["max_images"] // 2
    sample = pd.concat([g.sample(min(len(g), n), random_state=seed) for _, g in test.groupby("label")])

    fields = pd.read_parquet(path("processed") / "fields.parquet")
    need = set(sample.base_doc)
    fields = fields[fields.doc_id.isin(need) & fields.is_text]
    boxes = {d: {r.field: (r.x, r.y, r.w, r.h) for r in g.itertuples()} for d, g in fields.groupby("doc_id")}

    print(f"OCR on {len(sample)} test images ({sample.label.sum()} forged)")
    out = []
    with Pool(cfg["ocr"]["processes"], initializer=_init, initargs=(boxes,)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, sample.to_dict("records"), chunksize=4)):
            out.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(sample)}", flush=True)
    res = sample[["image_path", "label", "doc_type", "ctype", "tampered_fields"]].merge(pd.DataFrame(out), on="image_path")
    for f in FLAGS:
        res[f] = res[f].astype("boolean")
    res.to_parquet(path("processed") / "ocr_checks.parquet", index=False)

    perf = []
    for f in FLAGS:
        ev = res[res[f].notna()]
        if ev.empty:
            continue
        perf.append(dict(check=f, evaluated=len(ev),
                         flag_rate_genuine=ev.loc[ev.label == 0, f].mean(),
                         flag_rate_forged=ev.loc[ev.label == 1, f].mean(),
                         precision_in_sample=ev.loc[ev[f], "label"].mean() if ev[f].any() else float("nan")))
    perf = pd.DataFrame(perf)
    perf.to_csv(path("tables") / "ocr_check_performance.csv", index=False)
    print(perf.round(3).to_string(index=False))
    fk = res[res.label == 1].copy()
    fk["field"] = fk.tampered_fields.str.split(",").str[0]
    by = fk.groupby("field")[FLAGS].mean().round(3)
    by.to_csv(path("tables") / "ocr_check_by_tampered_field.csv")
    done()
