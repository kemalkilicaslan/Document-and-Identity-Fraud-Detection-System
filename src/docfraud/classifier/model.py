"""Field-level forgery CNN: each text field is scored separately and the document takes the maximum."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from docfraud.data.patches import windows

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(backbone: str, pretrained: bool) -> torch.nn.Module:
    import timm
    try:
        return timm.create_model(backbone, pretrained=pretrained, num_classes=1)
    except Exception as e:
        print(f"[warn] pretrained weights unavailable ({e}); training from scratch")
        return timm.create_model(backbone, pretrained=False, num_classes=1)


def normalize(batch_hwc_uint8: np.ndarray) -> np.ndarray:
    x = batch_hwc_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class PatchDataset(Dataset):
    """Training dataset: one random window per patch with light photometric jitter."""

    def __init__(self, paths, labels, window: int, train: bool, seed: int = 0):
        self.paths, self.labels = list(paths), np.asarray(labels, dtype=np.float32)
        self.window, self.train = window, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        arr = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        w = arr.shape[1]
        if w > self.window:
            s = self.rng.integers(0, w - self.window + 1) if self.train else (w - self.window) // 2
            arr = arr[:, s:s + self.window]
        else:
            arr = windows(arr, self.window, self.window)[0]
        if self.train:
            a = self.rng.uniform(0.8, 1.2)
            b = self.rng.uniform(-20, 20)
            arr = np.clip(arr.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        x = normalize(arr[None])[0]
        return torch.from_numpy(x), torch.tensor(self.labels[i])


@torch.no_grad()
def score_arrays(model, arrays: list[np.ndarray], window: int, stride: int,
                 dev: torch.device, batch_size: int = 256) -> np.ndarray:
    """Max-over-windows logit for each patch (HxWx3 uint8 arrays)."""
    model.eval()
    wins, owner = [], []
    for i, a in enumerate(arrays):
        w = windows(a, window, stride)
        wins.append(w)
        owner.extend([i] * len(w))
    wins = np.concatenate(wins)
    owner = np.asarray(owner)
    logits = np.empty(len(wins), dtype=np.float32)
    for s in range(0, len(wins), batch_size):
        x = torch.from_numpy(normalize(wins[s:s + batch_size])).to(dev)
        logits[s:s + batch_size] = model(x).squeeze(1).float().cpu().numpy()
    out = np.full(len(arrays), -np.inf, dtype=np.float32)
    np.maximum.at(out, owner, logits)
    return out
