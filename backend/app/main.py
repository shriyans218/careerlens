"""
CareerLens API — serves career predictions from either:
  1. an uploaded resume file (pdf/docx/txt), auto-scored into features, or
  2. manually provided feature scores (for the "learn/tune it yourself" UI)

Run with: uvicorn app.main:app --reload --port 8000
"""
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .feature_extraction import extract_features, features_to_vector, FEATURE_ORDER
from .resume_reader import read_resume
from .resume_parser import parse_resume_entities
from .tech_model import predict_tech_roles

# backend/app/main.py -> project root is two levels up (careerlens/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model_out" / "careerlens_final_model.joblib"
MODEL_PATH = os.environ.get("CAREERLENS_MODEL_PATH", str(DEFAULT_MODEL_PATH))

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model bundle not found at {MODEL_PATH!r}. "
        "Place careerlens_final_model.joblib under <project_root>/model_out/, "
        "or set CAREERLENS_MODEL_PATH to its location."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]

# Second model: trained directly on resume text -> tech job category.
# Fixes what the trait model structurally can't see (specific skills like
# "Python", "Kubernetes", "Hadoop"). Optional -- if missing, resume
# predictions still work with just the trait model.
TECH_MODEL_PATH = PROJECT_ROOT / "model_out" / "careerlens_tech_model.joblib"
tech_bundle = joblib.load(TECH_MODEL_PATH) if TECH_MODEL_PATH.is_file() else None

app = FastAPI(title="CareerLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend origin in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_top_k(feature_vector: list, k: int = 5):
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:k]
    return [
        {"career": label_encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
        for i in top_idx
    ]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "n_classes": len(label_encoder.classes_),
        "tech_model_loaded": tech_bundle is not None,
        "tech_n_classes": len(tech_bundle["classes"]) if tech_bundle else 0,
    }


@app.post("/api/predict-resume")
async def predict_resume(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = read_resume(file.filename, file_bytes)
    if not text.strip():
        raise HTTPException(400, "Could not extract any text from the uploaded file.")

    scores = extract_features(text)
    vector = features_to_vector(scores)
    predictions = predict_top_k(vector, k=5)
    tech_predictions = predict_tech_roles(tech_bundle, text, k=5)
    parsed_entities = parse_resume_entities(text)

    return {
        "top_prediction": predictions[0]["career"],
        "top_5": predictions,
        "tech_top_5": tech_predictions,  # null if tech model isn't bundled
        "resume_text": text,             # additive: powers the parsing/highlight view
        "parsed_entities": parsed_entities,  # additive: [{start,end,label,text}]
    }


class ManualScores(BaseModel):
    Linguistic: float
    Musical: float
    Bodily: float
    Logical_Mathematical: float
    Spatial_Visualization: float
    Interpersonal: float
    Intrapersonal: float
    Naturalist: float


@app.post("/api/predict-scores")
def predict_scores(scores: ManualScores):
    ordered = [
        scores.Linguistic, scores.Musical, scores.Bodily,
        scores.Logical_Mathematical, scores.Spatial_Visualization,
        scores.Interpersonal, scores.Intrapersonal, scores.Naturalist,
    ]
    predictions = predict_top_k(ordered, k=5)
    return {"top_prediction": predictions[0]["career"], "top_5": predictions}
