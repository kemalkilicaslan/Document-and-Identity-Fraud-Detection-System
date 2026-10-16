"""Step 04: train the field-level forgery CNN and keep the best validation PR-AUC checkpoint."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from _bootstrap import ROOT, done, step

from docfraud.classifier.model import PatchDataset, build_model, device
from docfraud.config import load_config, path, seed_everything


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(dev)).squeeze(1).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


if __name__ == "__main__":
    step("04 · Train document classifier")
    cfg = load_config()
    c = cfg["classifier"]
    seed = seed_everything()
    torch.manual_seed(seed)
    dev = device()
    print("device:", dev)

    idx = pd.read_parquet(path("processed") / "patch_index.parquet")
    tr, va = idx[idx.split == "train"], idx[idx.split == "val"]
    va = pd.concat([va[va.label == 1], va[va.label == 0].sample(min(len(va), 6 * va.label.sum()), random_state=seed)])
    print(f"train patches: {len(tr):,} ({tr.label.mean():.1%} tampered) | val: {len(va):,}")

    ds_tr = PatchDataset([ROOT / p for p in tr.patch_path], tr.label, c["window"], train=True, seed=seed)
    ds_va = PatchDataset([ROOT / p for p in va.patch_path], va.label, c["window"], train=False)
    kw = dict(batch_size=c["batch_size"], num_workers=c["num_workers"], persistent_workers=c["num_workers"] > 0)
    dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, shuffle=False, **kw)

    model = build_model(c["backbone"], c["pretrained"]).to(dev)
    pos_weight = torch.tensor((1 - tr.label.mean()) / tr.label.mean(), device=dev, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["epochs"] * len(dl_tr))

    best, hist = -1.0, []
    ckpt = path("models") / "doc_classifier.pt"
    for ep in range(1, c["epochs"] + 1):
        model.train()
        total = 0.0
        for i, (x, y) in enumerate(dl_tr):
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  epoch {ep} step {i + 1}/{len(dl_tr)} loss {total / (i + 1):.4f}", flush=True)
        lg, lb = predict(model, dl_va, dev)
        auc, ap = roc_auc_score(lb, lg), average_precision_score(lb, lg)
        hist.append(dict(epoch=ep, train_loss=total / len(dl_tr), val_roc_auc=auc, val_pr_auc=ap))
        print(f"epoch {ep}: train loss {total / len(dl_tr):.4f} | val ROC-AUC {auc:.4f} PR-AUC {ap:.4f}")
        if ap > best:
            best = ap
            torch.save({"state_dict": model.state_dict(), "backbone": c["backbone"], "epoch": ep,
                        "val_pr_auc": ap}, ckpt)
    pd.DataFrame(hist).to_csv(path("tables") / "doc_training_history.csv", index=False)
    print(f"best val PR-AUC {best:.4f} -> {ckpt}")
    done()
