"""Step 13: image-processing figures from the test split (field cropping, window scoring, localisation, OCR)."""
import json

import matplotlib
import numpy as np
import onnxruntime as ort
import pandas as pd
import pytesseract
from matplotlib.patches import Rectangle
from PIL import Image

from _bootstrap import ROOT, done, step

from docfraud.classifier.model import normalize
from docfraud.config import load_config, path
from docfraud.data.patches import crop_field, to_height, windows
from docfraud.ocr.engine import MRZ_CHARS, PASSPORTS, preprocess_line

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CMAP = plt.get_cmap("RdYlGn_r")
# Figures made of ID-image pixels are saved as JPEG: about 4x smaller than PNG at no visible loss
JPEG = {"quality": 90, "optimize": True}


def label_color(score: float, threshold: float) -> str:
    return "#b2182b" if score >= threshold else "#1b7837"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_image(rel_path: str) -> Image.Image:
    return Image.open(ROOT / rel_path).convert("RGB")


def draw_fields(ax, img, boxes: pd.DataFrame, scores: dict, tampered: set, lw=2.0):
    ax.imshow(img)
    for f, b in boxes.iterrows():
        if f not in scores:
            continue
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, lw=lw, ec=CMAP(scores[f])))
        if f in tampered:
            pad = 14
            ax.add_patch(Rectangle((b.x - pad, b.y - pad), b.w + 2 * pad, b.h + 2 * pad,
                                   fill=False, lw=lw, ec="#00b4d8", ls="--"))
    ax.axis("off")


def pick(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed) if len(df) else df


if __name__ == "__main__":
    step("13 · Image-processing figures")
    cfg = load_config()
    seed, pc = cfg["seed"], cfg["patches"]
    cal = json.loads((path("models") / "doc_calibration.json").read_text())
    sess = ort.InferenceSession(str(path("models") / "doc_classifier.onnx"), providers=["CPUExecutionProvider"])

    docs = pd.read_parquet(path("processed") / "doc_scores.parquet")
    docs = docs[docs.split == "test"].merge(
        pd.read_parquet(path("processed") / "manifest.parquet", columns=["image_path", "base_doc"]), on="image_path")
    fs = pd.read_parquet(path("processed") / "field_scores.parquet")
    fs = fs[fs.split == "test"]
    fields = pd.read_parquet(path("processed") / "fields.parquet")
    fields = fields[fields.is_text]
    docs["tampered"] = docs.tampered_fields.str.split(",").map(lambda v: {f for f in v if f})
    docs["localised"] = [t in tf for t, tf in zip(docs.top_field, docs.tampered)]
    flagged = docs[(docs.label == 1) & (docs.doc_score >= cal["threshold"]) & docs.localised]

    def boxes_of(r):
        return fields[fields.doc_id == r.base_doc].set_index("field")

    def scores_of(r):
        return dict(fs.loc[fs.image_path == r.image_path, ["field", "field_score"]].values)

    # Pipeline: dewarped document -> padded field crop -> fixed-height patch -> scored windows
    cand = flagged[(flagged.ctype == "Inpaint_and_Rewrite") & (flagged.tampered.map(len) == 1)].copy()
    cand["field"] = cand.tampered.map(lambda t: next(iter(t)))
    cand = cand.merge(fields[["doc_id", "field", "w", "h"]], left_on=["base_doc", "field"], right_on=["doc_id", "field"])
    cand["patch_w"] = cand.w * pc["height"] / cand.h
    cand = cand[cand.patch_w.between(2.5 * cal["window"], 4.5 * cal["window"])]
    r = cand.sort_values("source").drop_duplicates("source").iloc[0]
    img, boxes, field = load_image(r.image_path), boxes_of(r), r.field
    crop = crop_field(img, *boxes.loc[field, ["x", "y", "w", "h"]].astype(int), pc["pad_px"], pc["pad_ratio"])
    patch = to_height(crop, pc["height"])
    win = windows(np.asarray(patch), cal["window"], cal["stride"])
    win_scores = sigmoid(cal["platt_a"] * sess.run(None, {"input": normalize(win)})[0].ravel() + cal["platt_b"])

    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(4, len(win), height_ratios=[5, 0.9, 0.9, 1.1], hspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(img)
    for f, b in boxes.iterrows():
        ax.add_patch(Rectangle((b.x, b.y), b.w, b.h, fill=False, lw=1.2,
                               ec="#e63946" if f == field else "#adb5bd"))
    ax.set_title(f"1. Dewarped {'video frame' if r.source == 'clip' else 'document'} with the template field boxes · {r.doc_type}, "
                 f"tampered field in red: {field}", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(crop)
    ax.set_title(f"2. Field crop with padding ({crop.width}×{crop.height} px)", fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[2, :])
    ax.imshow(patch)
    ax.set_title(f"3. Resized to {pc['height']} px height, aspect ratio kept ({patch.width}×{patch.height} px)",
                 fontsize=10)
    ax.axis("off")
    for i, (w, sc) in enumerate(zip(win, win_scores)):
        ax = fig.add_subplot(gs[3, i])
        ax.imshow(w)
        ax.set_title(f"4.{i + 1} window · {sc:.3f}", fontsize=9, color=label_color(sc, cal["threshold"]))
        ax.axis("off")
    fig.suptitle(f"From ID image to field score: {len(win)} windows of {cal['window']} px (stride {cal['stride']}); "
                 f"field score = max window = {win_scores.max():.3f}", fontsize=11)
    fig.savefig(path("figures") / "image_pipeline.jpg", dpi=130, bbox_inches="tight", pil_kwargs=JPEG)
    plt.close(fig)

    # Localisation on test documents
    cases = [
        ("Inpaint_and_Rewrite, template", flagged[(flagged.ctype == "Inpaint_and_Rewrite") & (flagged.source == "template")]),
        ("Inpaint_and_Rewrite, video frame", flagged[(flagged.ctype == "Inpaint_and_Rewrite") & (flagged.source == "clip")]),
        ("Crop_and_Replace", flagged[flagged.ctype == "Crop_and_Replace"]),
        ("Genuine", docs[(docs.label == 0) & (docs.source == "clip") & (docs.doc_score < cal["threshold"])]),
    ]
    used = {r.doc_type}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, (title, pool) in zip(axes.ravel(), cases):
        fresh = pool[~pool.doc_type.isin(used)]
        c = pick(fresh if len(fresh) else pool, 1, seed).iloc[0]
        used.add(c.doc_type)
        draw_fields(ax, load_image(c.image_path), boxes_of(c), scores_of(c), c.tampered)
        tamp = ", ".join(sorted(c.tampered)) or "none"
        ax.set_title(f"{title} · {c.doc_type}\ndocument score {c.doc_score:.3f} · top field {c.top_field} · "
                     f"tampered {tamp}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Field forgery score")
    fig.suptitle("Field-level localisation on test documents (dashed cyan = ground-truth tampered field)", fontsize=11)
    fig.savefig(path("figures") / "forgery_localisation.jpg", dpi=130, bbox_inches="tight", pil_kwargs=JPEG)
    plt.close(fig)

    # Genuine vs forged versions of the same field
    pi = pd.read_parquet(path("processed") / "patch_index.parquet", columns=["patch_path", "image_path", "field"])
    fp = fs.merge(pi, on=["image_path", "field"]).merge(docs[["image_path", "doc_type", "ctype", "source"]], on="image_path")
    fp = fp[fp.source == "template"]
    rows = []
    for ctype in ("Inpaint_and_Rewrite", "Crop_and_Replace"):
        forged = fp[(fp.label == 1) & (fp.ctype == ctype)]
        genuine = fp[(fp.label == 0) & (fp.ctype == "Real")]
        pairs = forged.merge(genuine, on=["doc_type", "field"], suffixes=("_f", "_g"))
        pairs = pairs[pairs.patch_path_f.map(lambda p: Image.open(ROOT / p).width) < 900]
        pairs = pairs.drop_duplicates("patch_path_f").drop_duplicates("patch_path_g").drop_duplicates(["doc_type", "field"])
        for _, g in pick(pairs, 3, seed).iterrows():
            rows.append((ctype, g))
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 1.35 * len(rows) + 0.6))
    for (ctype, g), (a_g, a_f) in zip(rows, np.atleast_2d(axes)):
        a_g.imshow(Image.open(ROOT / g.patch_path_g))
        a_g.set_title(f"Genuine · {g.doc_type} / {g.field} · score {g.field_score_g:.3f}", fontsize=9,
                      color=label_color(g.field_score_g, cal["threshold"]))
        a_f.imshow(Image.open(ROOT / g.patch_path_f))
        a_f.set_title(f"{ctype} · score {g.field_score_f:.3f}", fontsize=9, color=label_color(g.field_score_f, cal["threshold"]))
        a_g.axis("off")
        a_f.axis("off")
    fig.suptitle("Field patches the CNN sees: genuine vs. forged", fontsize=11)
    fig.tight_layout()
    fig.savefig(path("figures") / "field_patches.jpg", dpi=130, bbox_inches="tight", pil_kwargs=JPEG)
    plt.close(fig)

    # OCR preprocessing of the MRZ line
    mrz = docs[docs.doc_type.isin(PASSPORTS) & (docs.label == 0)]
    mrz = mrz[mrz.base_doc.isin(fields.loc[fields.field == "mrz_line1", "doc_id"])]
    examples = [("template", mrz[mrz.source == "template"]), ("video frame", mrz[mrz.source == "clip"])]
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3.2))
    for (src, pool), (a_raw, a_pre) in zip(examples, axes):
        c = pick(pool, 1, seed).iloc[0]
        img = load_image(c.image_path)
        box = tuple(int(v) for v in fields.loc[(fields.doc_id == c.base_doc) & (fields.field == "mrz_line1"),
                                               ["x", "y", "w", "h"]].iloc[0])
        pre = preprocess_line(img, box)
        d = pytesseract.image_to_data(pre, lang="eng", config=f"--psm 7 -c tessedit_char_whitelist={MRZ_CHARS}",
                                      output_type=pytesseract.Output.DICT)
        words = [(t, float(cf)) for t, cf in zip(d["text"], d["conf"]) if t.strip() and float(cf) >= 0]
        text = " ".join(t for t, _ in words) or "(nothing read)"
        a_raw.imshow(crop_field(img, *box, pad_px=4, pad_ratio=0.1))
        a_raw.set_title(f"Raw MRZ crop · {c.doc_type} {src}", fontsize=9)
        a_pre.imshow(pre, cmap="gray")
        a_pre.set_title(f"Grayscale, 64 px, auto-contrast → Tesseract reads:\n{text}", fontsize=8,
                        family="monospace")
        a_raw.axis("off")
        a_pre.axis("off")
    fig.suptitle("OCR preprocessing of the MRZ line", fontsize=11)
    fig.tight_layout()
    fig.savefig(path("figures") / "ocr_preprocessing.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("figures: image_pipeline, forgery_localisation, field_patches, ocr_preprocessing")
    done()
