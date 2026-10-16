"""Reading the SIDTD dataset: image manifest, field layouts and leakage-safe groups.

Layout after unzipping into data/interim:
  templates/Images/{reals,fakes}/*.jpg       templates/Annotations/reals/<doc_type>.json (VIA)
  templates/Annotations/fakes/<name>.json
  clips_cropped/Images/{reals,fakes}/*.jpg    clips_cropped/annotations/fakes/<name>_info.json

Cropped clip frames are dewarped to the template size, so the field boxes of the
source template can be used on them directly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DOC_RE = re.compile(r"^([a-z]{3}_[a-z]+)_(\d+)")


def doc_id_of(name: str) -> str:
    m = DOC_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected SIDTD file name: {name}")
    return f"{m.group(1)}_{m.group(2)}"


def _clean(v):
    return None if v in (None, "None", "") else v


def load_fields(interim: Path, non_text: list[str]) -> pd.DataFrame:
    """One row per (document, field) with the box and the ground-truth text."""
    rows = []
    for p in sorted((interim / "templates" / "Annotations" / "reals").glob("*.json")):
        doc_type = p.stem
        meta = json.loads(p.read_text(encoding="utf-8"))["_via_img_metadata"]
        for item in meta.values():
            doc_id = f"{doc_type}_{Path(item['filename']).stem}"
            for r in item["regions"]:
                name = r["region_attributes"].get("field_name", "")
                s = r["shape_attributes"]
                if s.get("name") == "rect":
                    x, y, w, h = s["x"], s["y"], s["width"], s["height"]
                else:
                    xs, ys = s["all_points_x"], s["all_points_y"]
                    x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                rows.append(dict(doc_id=doc_id, doc_type=doc_type, field=name,
                                 value=r["region_attributes"].get("value", ""),
                                 x=int(x), y=int(y), w=int(w), h=int(h),
                                 is_text=name not in non_text))
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["doc_id", "field"]).reset_index(drop=True)


def build_manifest(interim: Path) -> pd.DataFrame:
    rows = []
    t = interim / "templates"
    for img in sorted((t / "Images" / "reals").glob("*.jpg")):
        rows.append(dict(image_path=img, source="template", label=0, base_doc=img.stem,
                         variant_id=img.stem))
    for img in sorted((t / "Images" / "fakes").glob("*.jpg")):
        a = json.loads((t / "Annotations" / "fakes" / f"{img.stem}.json").read_text())
        rows.append(dict(image_path=img, source="template", label=1, base_doc=doc_id_of(img.stem),
                         variant_id=img.stem, ctype=a["ctype"], field=_clean(a["field"]),
                         second_field=_clean(a["second_field"]),
                         src=Path(a["src"]).stem, second_src=_clean(a["second_src"])))

    c = interim / "clips_cropped"
    for img in sorted((c / "Images" / "reals").glob("*.jpg")):
        rows.append(dict(image_path=img, source="clip", label=0, base_doc=doc_id_of(img.stem),
                         variant_id=doc_id_of(img.stem)))
    for img in sorted((c / "Images" / "fakes").glob("*.jpg")):
        ann = c / "annotations" / "fakes" / f"{img.stem.replace('_dewarped', '')}_info.json"
        a = json.loads(ann.read_text())
        rows.append(dict(image_path=img, source="clip", label=1, base_doc=doc_id_of(img.stem),
                         variant_id=img.stem.split("_frame")[0], ctype=a["ctype"],
                         field=_clean(a["field"]), second_field=_clean(a["second_field"]),
                         src=Path(a["src"]).stem, second_src=_clean(a["second_src"])))

    df = pd.DataFrame(rows)
    df["second_src"] = df["second_src"].map(lambda v: Path(v).stem if isinstance(v, str) else None)
    df["image_path"] = df["image_path"].map(lambda p: str(Path(p).relative_to(interim.parent.parent)))
    df["doc_type"] = df["base_doc"].str.rsplit("_", n=1).str[0]
    df["country"] = df["doc_type"].str[:3]
    df["doc_kind"] = df["doc_type"].str[4:]
    df["ctype"] = df["ctype"].fillna("Real")
    df["tampered_fields"] = df.apply(
        lambda r: sorted({f for f in (r.get("field"), r.get("second_field")) if isinstance(f, str)}),
        axis=1)
    df["group_id"] = _groups(df)
    return df


def _groups(df: pd.DataFrame) -> pd.Series:
    """Union-find over documents that share content (Crop_and_Replace mixes two documents)."""
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, r in df.iterrows():
        docs = [d for d in (r["base_doc"], r.get("src"), r.get("second_src")) if isinstance(d, str)]
        for d in docs[1:]:
            parent[find(d)] = find(docs[0])
        find(docs[0])
    return df["base_doc"].map(find)
