"""Tesseract OCR on field crops and the MRZ / date consistency checks built on it."""
from __future__ import annotations

import re
from datetime import date

import pytesseract
from PIL import Image, ImageOps

from docfraud.data.patches import crop_field
from docfraud.validators.dates import date_logic_flags, parse_date
from docfraud.validators.mrz import check_td3_line2, mrz_date, parse_td3_line2

MRZ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
PASSPORTS = {"aze_passport", "grc_passport", "lva_passport", "srb_passport"}
DATE_FIELDS = ("birth_date", "issue_date", "expiry_date")


def preprocess_line(img: Image.Image, box, height: int = 64) -> Image.Image:
    """Crop a text line, convert to grayscale, upscale to a fixed height and stretch the contrast."""
    patch = ImageOps.grayscale(crop_field(img, *box, pad_px=4, pad_ratio=0.1))
    scale = height / max(1, patch.height)
    patch = patch.resize((max(1, int(patch.width * scale)), height), Image.Resampling.BICUBIC)
    return ImageOps.autocontrast(patch)


def ocr_line(img: Image.Image, box, whitelist: str | None = None, height: int = 64) -> tuple[str, float]:
    """OCR one text line; returns (text, mean word confidence 0-100)."""
    patch = preprocess_line(img, box, height)
    cfg = "--psm 7"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(patch, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    return " ".join(t for t, _ in words), sum(c for _, c in words) / len(words)


DIGIT_POS = {9, 19, 27, 42, 43, *range(13, 19), *range(21, 27)}
TO_DIGIT = str.maketrans("OQDILZSBGT", "0001125867")


def best_td3_line2(raw: str) -> str:
    """Pick the 44-character window with the most valid check digits and fix
    letter/digit confusions in numeric positions (OCR often adds a stray char)."""
    s = "".join(c for c in raw.upper().replace(" ", "") if c.isalnum() or c == "<")
    cands = [s[i:i + 44] for i in range(len(s) - 43)] if len(s) > 44 else [s.ljust(44, "<")]
    fixed = ["".join(c.translate(TO_DIGIT) if i in DIGIT_POS else c for i, c in enumerate(w)) for w in cands]
    return max(fixed, key=lambda w: sum(check_td3_line2(w).values()))


TD3_LINE2 = re.compile(r"[A-Z0-9<]{9}[0-9][A-Z<]{3}[0-9]{6}[0-9][MF<][0-9]{6}[0-9][A-Z0-9<]{14}[0-9<][0-9]")
MIN_CONF = 75.0
MIN_CONF_MRZ = 60.0


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _alnum(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())


def check_document(image_path: str, doc_type: str, boxes: dict[str, tuple]) -> dict:
    """Run OCR checks on one dewarped document image.

    Flags are True when the check failed, False when it passed and None when the
    fields needed for it could not be read.
    """
    img = Image.open(image_path).convert("RGB")
    out: dict = {}
    dates: dict[str, date | None] = {}
    for f in DATE_FIELDS:
        if f in boxes:
            wl = None if doc_type == "grc_passport" else "0123456789.-/ "
            txt, conf = ocr_line(img, boxes[f], whitelist=wl)
            out[f"ocr_{f}"] = txt
            dates[f] = parse_date(txt) if conf >= MIN_CONF else None
    logic = date_logic_flags(dates.get("birth_date"), dates.get("issue_date"), dates.get("expiry_date"))
    out["date_logic_fail"] = any(logic.values()) if logic else None

    out["mrz_checkdigit_fail"] = None
    out["mrz_viz_mismatch"] = None
    if doc_type in PASSPORTS and "mrz_line1" in boxes:
        raw, conf = ocr_line(img, boxes["mrz_line1"], whitelist=MRZ_CHARS)
        out["ocr_mrz_line1"], out["ocr_mrz_conf"] = raw, conf
        line2 = best_td3_line2(raw) if len(raw) >= 40 else ""
        if conf >= MIN_CONF_MRZ and TD3_LINE2.fullmatch(line2):
            out["mrz_checkdigit_fail"] = not all(check_td3_line2(line2).values())
            p = parse_td3_line2(line2)
            mism = []
            if "number" in boxes:
                viz_txt, viz_conf = ocr_line(img, boxes["number"])
                viz_no = _alnum(viz_txt)
                if viz_no and viz_conf >= MIN_CONF:
                    mism.append(edit_distance(viz_no, p["document_number"].strip("<")) > 1)
            mrz_birth = mrz_date(p["birth_date"])
            if dates.get("birth_date") and mrz_birth:
                mism.append(dates["birth_date"] != mrz_birth)
            mrz_exp = mrz_date(p["expiry_date"], future=True)
            if dates.get("expiry_date") and mrz_exp:
                mism.append(dates["expiry_date"] != mrz_exp)
            out["mrz_viz_mismatch"] = any(mism) if mism else None
    return out
