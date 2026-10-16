"""Scoring API: uvicorn docfraud.api:app --reload (after run_all.sh)."""
from __future__ import annotations

import io
import json
from functools import lru_cache

import duckdb
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from docfraud.classifier.model import normalize
from docfraud.config import ROOT, path
from docfraud.data.patches import crop_field, field_layout, to_height, windows
from docfraud.rules.engine import load_rules, score

app = FastAPI(title="Document & Identity Fraud Detection API", version="1.0")


@lru_cache(maxsize=1)
def doc_model():
    cal = json.loads((path("models") / "doc_calibration.json").read_text())
    sess = ort.InferenceSession(str(path("models") / "doc_classifier.onnx"), providers=["CPUExecutionProvider"])
    fields = pd.read_parquet(path("processed") / "fields.parquet")
    return sess, cal, fields


@lru_cache(maxsize=1)
def app_model():
    meta = json.loads((path("models") / "app_lgbm_meta.json").read_text())
    booster = lgb.Booster(model_file=str(path("models") / "app_lgbm.txt"))
    thr = json.loads((path("models") / "alert_threshold.json").read_text())
    return booster, meta, thr, load_rules(ROOT / "configs" / "rules.yaml")


def template_size(doc_type: str) -> tuple[int, int]:
    img = path("interim") / "templates" / "Images" / "reals" / f"{doc_type}_00.jpg"
    with Image.open(img) as im:
        return im.size


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/score")
async def score_document(file: UploadFile = File(...), doc_type: str = Form(...)):
    sess, cal, fields = doc_model()
    if doc_type not in set(fields.doc_type):
        raise HTTPException(400, f"unknown doc_type; one of {sorted(set(fields.doc_type))}")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize(template_size(doc_type))
    out = []
    for f, box in field_layout(fields, doc_type).items():
        arr = np.asarray(to_height(crop_field(img, *box), cal["height"]))
        w = windows(arr, cal["window"], cal["stride"])
        logit = float(sess.run(None, {"input": normalize(w)})[0].max())
        out.append({"field": f, "score": float(1 / (1 + np.exp(-(cal["platt_a"] * logit + cal["platt_b"]))))})
    out.sort(key=lambda d: -d["score"])
    doc_score = out[0]["score"] if out else 0.0
    return {"doc_type": doc_type, "doc_score": round(doc_score, 4), "suspicious": bool(doc_score >= cal["threshold"]),
            "threshold": round(cal["threshold"], 4), "top_field": out[0]["field"] if out else None,
            "fields": [{**d, "score": round(d["score"], 4)} for d in out]}


@app.get("/applications/{app_id}")
def get_application(app_id: str):
    con = duckdb.connect(str(path("db")), read_only=True)
    try:
        s = con.execute("SELECT * FROM application_scores WHERE application_id = ?", [app_id]).df()
        if s.empty:
            raise HTTPException(404, "application not found")
        f = con.execute("SELECT * FROM application_features WHERE application_id = ?", [app_id]).df()
        hits = con.execute("SELECT rule_id, weight, owner FROM rule_hits WHERE application_id = ?", [app_id]).df()
    finally:
        con.close()
    return {"scores": json.loads(s.to_json(orient="records", date_format="iso"))[0],
            "rules": json.loads(hits.to_json(orient="records")),
            "features": json.loads(f.to_json(orient="records", date_format="iso"))[0]}


class ApplicationFeatures(BaseModel):
    """Point-in-time features of a new application, as built by sql/01_features.sql."""
    features: dict


@app.post("/applications/score")
def score_application(req: ApplicationFeatures):
    booster, meta, thr, rules = app_model()
    row = pd.DataFrame([req.features])
    missing = [c for c in meta["features"] if c not in row]
    if missing:
        raise HTTPException(400, f"missing features: {missing}")
    con = duckdb.connect()
    con.register("one", row)
    con.execute("CREATE TABLE application_features AS SELECT 'NEW' AS application_id, * FROM one")
    rs, hits = score(con, rules)
    con.close()
    X = row[meta["features"]].copy()
    for c in meta["categorical"]:
        X[c] = pd.Categorical(X[c], categories=meta["categories"][c])
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    p = float(np.clip(booster.predict(X)[0], 1e-6, 1 - 1e-6))
    ml = 1 / (1 + np.exp(-(meta["platt_a"] * np.log(p / (1 - p)) + meta["platt_b"])))
    rule_score = float(rs.rule_score.iloc[0])
    hybrid = 0.5 * rule_score / 100 + 0.5 * ml
    return {"rule_score": rule_score, "rule_decision": str(rs.decision.iloc[0]), "ml_score": round(float(ml), 4),
            "hybrid_score": round(float(hybrid), 4), "alert": bool(hybrid >= thr["hybrid_threshold"]),
            "reasons": json.loads(hits[["rule_id", "weight", "owner"]].to_json(orient="records"))}
