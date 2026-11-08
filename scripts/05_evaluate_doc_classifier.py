"""Step 05: document scores, Platt calibration, test metrics and ONNX export.

Document score = max field logit; the operating point sends 5% of genuine
validation documents to review.
"""
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from _bootstrap import ROOT, done, step

from docfraud.classifier.model import build_model, device, score_arrays
from docfraud.config import load_config, path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def recall_at_fpr(y, s, fpr_target):
    thr = np.quantile(s[y == 0], 1 - fpr_target)
    return (s[y == 1] > thr).mean(), thr


if __name__ == "__main__":
    step("05 · Evaluate document classifier")
    cfg = load_config()
    c = cfg["classifier"]
    dev = device()
    ck = torch.load(path("models") / "doc_classifier.pt", map_location="cpu")
    model = build_model(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    idx = pd.read_parquet(path("processed") / "patch_index.parquet")
    idx = idx[idx.split.isin(["val", "test"])].reset_index(drop=True)
    print(f"scoring {len(idx):,} field patches from {idx.image_path.nunique():,} images")
    logits = np.empty(len(idx), dtype=np.float32)
    CH = 4000
    for s in range(0, len(idx), CH):
        arrs = [np.asarray(Image.open(ROOT / p).convert("RGB")) for p in idx.patch_path[s:s + CH]]
        logits[s:s + CH] = score_arrays(model, arrs, c["window"], c["stride"], dev)
        print(f"  {min(s + CH, len(idx)):,}/{len(idx):,}", flush=True)
    idx["field_logit"] = logits

    top = idx.loc[idx.groupby("image_path").field_logit.idxmax()]
    docs = top[["image_path", "split", "doc_label", "doc_type", "ctype", "source", "field", "field_logit"]].rename(
        columns={"doc_label": "label", "field": "top_field", "field_logit": "doc_logit"})
    tamp = idx[idx.label == 1].groupby("image_path").field.apply(lambda f: ",".join(sorted(f)))
    docs["tampered_fields"] = docs.image_path.map(tamp).fillna("")

    val = docs[docs.split == "val"]
    platt = LogisticRegression().fit(val[["doc_logit"]], val.label)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    docs["doc_score"] = 1 / (1 + np.exp(-(a * docs.doc_logit + b)))
    idx["field_score"] = 1 / (1 + np.exp(-(a * idx.field_logit + b)))
    _, thr_logit = recall_at_fpr(val.label.values, val.doc_logit.values, 0.05)
    thr = float(1 / (1 + np.exp(-(a * thr_logit + b))))

    test = docs[docs.split == "test"]
    y, sc = test.label.values, test.doc_score.values
    rows = [dict(metric="doc_roc_auc", value=roc_auc_score(y, sc)),
            dict(metric="doc_pr_auc", value=average_precision_score(y, sc))]
    for f in (0.01, 0.05, 0.10):
        rows.append(dict(metric=f"doc_recall_at_{int(f * 100)}pct_fpr", value=recall_at_fpr(y, sc, f)[0]))
    tp = idx[idx.split == "test"]
    rows += [dict(metric="field_roc_auc", value=roc_auc_score(tp.label, tp.field_logit)),
             dict(metric="field_pr_auc", value=average_precision_score(tp.label, tp.field_logit))]
    flagged = test[(test.label == 1) & (test.doc_score >= thr)]
    loc = np.mean([t in f.split(",") for t, f in zip(flagged.top_field, flagged.tampered_fields)]) if len(flagged) else np.nan
    rows += [dict(metric="localisation_acc_top_field", value=loc),
             dict(metric="operating_threshold", value=thr),
             dict(metric="test_fpr_at_threshold", value=(test[test.label == 0].doc_score >= thr).mean()),
             dict(metric="test_recall_at_threshold", value=(test[test.label == 1].doc_score >= thr).mean())]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(path("tables") / "doc_model_metrics.csv", index=False)
    print(metrics.to_string(index=False))

    test = test.assign(flag=test.doc_score >= thr)
    for col in ("ctype", "doc_type"):
        t = test.groupby(col).agg(images=("label", "size"), fakes=("label", "sum"), flagged=("flag", "sum"),
                                  recall=("flag", lambda f: f[test.loc[f.index, "label"] == 1].mean()),
                                  fpr=("flag", lambda f: f[test.loc[f.index, "label"] == 0].mean()))
        t.to_csv(path("tables") / f"doc_model_by_{col}.csv")
        print(t.round(3))
    tf = tp.assign(flag=tp.field_score >= thr)
    by_field = tf[tf.label == 1].groupby("field").agg(tampered=("flag", "size"), caught=("flag", "mean"))
    by_field = by_field[by_field.tampered >= 10].sort_values("caught")
    by_field.to_csv(path("tables") / "doc_model_by_tampered_field.csv")
    print("\nhardest tampered fields:\n", by_field.head(8).round(3))

    docs.to_parquet(path("processed") / "doc_scores.parquet", index=False)
    idx[["image_path", "split", "field", "label", "field_score"]].to_parquet(
        path("processed") / "field_scores.parquet", index=False)
    (path("models") / "doc_calibration.json").write_text(json.dumps(
        {"platt_a": a, "platt_b": b, "threshold": thr, "window": c["window"], "stride": c["stride"],
         "height": cfg["patches"]["height"]}, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y, sc)
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[0].set(title=f"ROC (AUC {roc_auc_score(y, sc):.3f})", xlabel="FPR", ylabel="TPR")
    p, r, _ = precision_recall_curve(y, sc)
    ax[1].plot(r, p)
    ax[1].set(title=f"PR (AP {average_precision_score(y, sc):.3f})", xlabel="Recall", ylabel="Precision")
    fig.tight_layout()
    fig.savefig(path("figures") / "doc_model_curves.png", dpi=150)

    dummy = torch.randn(1, 3, cfg["patches"]["height"], c["window"])
    torch.onnx.export(model.cpu(), dummy, path("models") / "doc_classifier.onnx", input_names=["input"],
                      output_names=["logit"], dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
                      opset_version=18, external_data=False)
    print("exported models/doc_classifier.onnx")
    done()
