"""Step 03: crop every text field into a fixed-height patch.

Training images keep their tampered fields plus a few untouched ones; validation
and test images keep every text field, as in production.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from _bootstrap import ROOT, done, step

from docfraud.config import load_config, path, rel, seed_everything
from docfraud.data.patches import crop_field, to_height

cfg = load_config()
pc = cfg["patches"]
OUT = path("patches")
BOXES: dict = {}


def _init(boxes):
    global BOXES
    BOXES = boxes


def work(job):
    r, seed = job
    rng = np.random.default_rng(seed)
    boxes = BOXES[r["base_doc"]]
    tampered = set(r["tampered_fields"])
    names = sorted(boxes)
    if r["split"] == "train":
        others = [f for f in names if f not in tampered]
        k = min(len(others), pc["train_other_fields"])
        names = [f for f in names if f in tampered] + list(rng.choice(others, k, replace=False))
    img = Image.open(ROOT / r["image_path"]).convert("RGB")
    stem = os.path.splitext(os.path.basename(r["image_path"]))[0]
    rows = []
    for f in names:
        patch = to_height(crop_field(img, *boxes[f], pc["pad_px"], pc["pad_ratio"]), pc["height"])
        p = OUT / r["split"] / f"{stem}__{f.replace('/', '-')}.png"
        patch.save(p)
        rows.append(dict(patch_path=rel(p), image_path=r["image_path"],
                         split=r["split"], field=f, label=int(f in tampered),
                         doc_label=r["label"], ctype=r["ctype"], doc_type=r["doc_type"],
                         source=r["source"]))
    return rows


if __name__ == "__main__":
    step("03 · Field patches")
    seed = seed_everything()
    m = pd.read_parquet(path("processed") / "manifest.parquet")
    sp = pd.read_parquet(path("splits") / "split.parquet")
    m = m.merge(sp, on="image_path")
    fields = pd.read_parquet(path("processed") / "fields.parquet")
    fields = fields[fields.is_text]
    boxes = {d: {r.field: (r.x, r.y, r.w, r.h) for r in g.itertuples()} for d, g in fields.groupby("doc_id")}
    for s in ("train", "val", "test"):
        (OUT / s).mkdir(parents=True, exist_ok=True)

    jobs = [(r, seed + i) for i, r in enumerate(m.to_dict("records"))]
    rows = []
    with Pool(max(1, (os.cpu_count() or 4) - 2), initializer=_init, initargs=(boxes,)) as pool:
        for i, res in enumerate(pool.imap_unordered(work, jobs, chunksize=16)):
            rows.extend(res)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(jobs):,} images")
    idx = pd.DataFrame(rows)
    idx.to_parquet(path("processed") / "patch_index.parquet", index=False)
    print(idx.groupby(["split", "label"]).size().unstack())
    done()
