"""Step 02: train/val/test split grouped by source document.

Groups join documents mixed by Crop_and_Replace; clip frames are subsampled per
document and per fake variant so near-identical frames do not dominate.
"""
import numpy as np
import pandas as pd

from _bootstrap import done, step

from docfraud.config import load_config, path, seed_everything

step("02 · Splits")
cfg = load_config()
seed = seed_everything()
rng = np.random.default_rng(seed)
s = cfg["splits"]

m = pd.read_parquet(path("processed") / "manifest.parquet")

groups = m.groupby("group_id").agg(doc_type=("doc_type", "first"), has_fake=("label", "max")).reset_index()
groups["split"] = ""
for _, g in groups.groupby(["doc_type", "has_fake"]):
    idx = rng.permutation(g.index.values)
    n_tr = int(round(len(idx) * s["train"]))
    n_va = int(round(len(idx) * s["val"]))
    groups.loc[idx[:n_tr], "split"] = "train"
    groups.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
    groups.loc[idx[n_tr + n_va:], "split"] = "test"
m = m.merge(groups[["group_id", "split"]], on="group_id")

keep = []
for split, part in m.groupby("split"):
    keep.append(part[part.source == "template"])
    clips = part[part.source == "clip"]
    reals = clips[clips.label == 0]
    keep.append(reals.sample(frac=1, random_state=seed).groupby("base_doc").head(s["real_frames_per_doc"][split]))
    fakes = clips[clips.label == 1]
    keep.append(fakes.sample(frac=1, random_state=seed).groupby("variant_id").head(s["fake_frames_per_variant"][split]))
sel = pd.concat(keep, ignore_index=True)

sel[["image_path", "split"]].to_parquet(path("splits") / "split.parquet", index=False)
summary = sel.groupby(["split", "label"]).agg(images=("image_path", "size"), groups=("group_id", "nunique"))
summary.to_csv(path("tables") / "split_summary.csv")
print(summary)
assert sel.groupby("group_id").split.nunique().max() == 1, "a group leaked into two splits"
print("no group appears in more than one split")
done()
