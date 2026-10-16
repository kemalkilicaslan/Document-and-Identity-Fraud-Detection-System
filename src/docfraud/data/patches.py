"""Cropping field patches out of a dewarped document image."""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop_field(img: Image.Image, x: int, y: int, w: int, h: int,
               pad_px: int = 6, pad_ratio: float = 0.12) -> Image.Image:
    px = pad_px + int(w * pad_ratio / 2)
    py = pad_px + int(h * pad_ratio)
    box = (max(0, x - px), max(0, y - py), min(img.width, x + w + px), min(img.height, y + h + py))
    return img.crop(box)


def to_height(patch: Image.Image, height: int) -> Image.Image:
    """Resize to a fixed height, keeping the aspect ratio."""
    w = max(1, round(patch.width * height / max(1, patch.height)))
    return patch.resize((w, height), Image.Resampling.BILINEAR)


def windows(arr: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Split an HxWx3 patch into fixed-width windows; short patches are padded with grey.

    Returns an array of shape (n_windows, H, window, 3).
    """
    h, w, _ = arr.shape
    if w <= window:
        out = np.full((h, window, 3), 127, dtype=arr.dtype)
        off = (window - w) // 2
        out[:, off:off + w] = arr
        return out[None]
    starts = list(range(0, w - window + 1, stride))
    if starts[-1] != w - window:
        starts.append(w - window)
    return np.stack([arr[:, s:s + window] for s in starts])


def field_layout(fields, doc_type: str) -> dict[str, tuple[int, int, int, int]]:
    """Median box of every text field of a document type, widened 15% to absorb layout drift."""
    sub = fields[(fields.doc_type == doc_type) & fields.is_text]
    med = sub.groupby("field")[["x", "y", "w", "h"]].median().round().astype(int)
    med["w"] = (med["w"] * 1.15).astype(int)
    return {f: tuple(r) for f, r in med.iterrows()}
