"""
CareerLens API — serves career predictions from either:
  1. an uploaded resume file (pdf/docx/txt), auto-scored into features, or
  2. manually provided feature scores (for the "learn/tune it yourself" UI)

Run with: uvicorn app.main:app --reload --port 8000
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .feature_extraction import extract_features, features_to_vector, FEATURE_ORDER
from .resume_reader import read_resume
from .resume_parser import parse_resume_entities
from .tech_model import predict_tech_roles
from .gap_analysis import analyze_gap

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

# Static dashboard data: real logged model-comparison macro-F1 scores
# and a real t-SNE projection of the training set, produced offline by
# training/09_export_dashboard_data.py. Optional -- dashboard endpoints
# 404 gracefully if it hasn't been generated yet.
DASHBOARD_DATA_PATH = PROJECT_ROOT / "model_out" / "dashboard_data.json"
DASHBOARD_PROJECTION_PATH = PROJECT_ROOT / "model_out" / "dashboard_projection.joblib"
dashboard_data = None
dashboard_projection = None
if DASHBOARD_DATA_PATH.is_file():
    with open(DASHBOARD_DATA_PATH) as f:
        dashboard_data = json.load(f)
if DASHBOARD_PROJECTION_PATH.is_file():
    dashboard_projection = joblib.load(DASHBOARD_PROJECTION_PATH)

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


_MODEL_DISPLAY_NAMES = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "logreg": "Logistic Regression",
}


def predict_per_model(feature_vector: list):
    """Dashboard-only: ask each sub-model inside the deployed ensemble
    for its own independent top prediction + confidence on THIS resume.
    Does not affect predict_top_k / the ensemble's actual result -- the
    ensemble's soft-voted output is still what's returned as
    top_prediction / top_5 everywhere else."""
    if not hasattr(model, "named_estimators_"):
        return None  # deployed model isn't a VotingClassifier bundle
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    breakdown = []
    for key, est in model.named_estimators_.items():
        proba = est.predict_proba(X_scaled)[0]
        top_i = int(np.argmax(proba))
        breakdown.append({
            "model": _MODEL_DISPLAY_NAMES.get(key, key),
            "career": label_encoder.classes_[top_i],
            "confidence": round(float(proba[top_i]), 4),
        })
    return breakdown


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "n_classes": len(label_encoder.classes_),
        "tech_model_loaded": tech_bundle is not None,
        "tech_n_classes": len(tech_bundle["classes"]) if tech_bundle else 0,
    }


@app.get("/api/dashboard")
def get_dashboard_data():
    """Real, precomputed analytics: actual logged model-comparison
    macro-F1 scores and a real t-SNE projection of the training set.
    Nothing here is invented per-request."""
    if dashboard_data is None:
        raise HTTPException(
            404,
            "Dashboard data not generated yet. Run "
            "training/09_export_dashboard_data.py to produce "
            "model_out/dashboard_data.json.",
        )
    return dashboard_data


def project_resume_point(feature_vector: list):
    """Projects this resume's real 8-dim trait vector into the same 2D
    space as the training-set t-SNE plot, using the PCA(2) fit on the
    same scaled features (t-SNE itself has no transform() for new
    points). Returns None if the projection model wasn't exported."""
    if dashboard_projection is None:
        return None
    proj_scaler = dashboard_projection["scaler"]
    pca = dashboard_projection["pca"]
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = proj_scaler.transform(X)
    coords = pca.transform(X_scaled)[0]
    return {"x": round(float(coords[0]), 3), "y": round(float(coords[1]), 3)}



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
    resume_point = project_resume_point(vector)
    model_breakdown = predict_per_model(vector)  # dashboard-only, doesn't touch predictions above

    return {
        "top_prediction": predictions[0]["career"],
        "top_5": predictions,
        "tech_top_5": tech_predictions,  # null if tech model isn't bundled
        "resume_text": text,             # additive: powers the parsing/highlight view
        "parsed_entities": parsed_entities,  # additive: [{start,end,label,text}]
        "resume_point": resume_point,    # additive: {x,y} on the dashboard's t-SNE map, or null
        "model_breakdown": model_breakdown,  # additive, dashboard-only: [{model, career, confidence}]
        "trait_scores": scores,          # additive: {trait: score}, powers the skill-gap report
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
    point = project_resume_point(ordered)
    return {"top_prediction": predictions[0]["career"], "top_5": predictions, "resume_point": point}


class GapReportRequest(BaseModel):
    """Accepts trait scores using the SAME key format extract_features()
    produces (e.g. "Logical - Mathematical", "Spatial-Visualization"),
    via field aliases -- this is what the frontend sends back from
    result.trait_scores after a resume prediction. Also accepts the
    underscore-safe field names directly (e.g. from the assessment
    sliders), since populate_by_name is enabled below.
    `career` defaults to whichever career the scores best match if not given."""
    Linguistic: float
    Musical: float
    Bodily: float
    Logical_Mathematical: float = Field(alias="Logical - Mathematical")
    Spatial_Visualization: float = Field(alias="Spatial-Visualization")
    Interpersonal: float
    Intrapersonal: float
    Naturalist: float
    career: str | None = None

    model_config = {"populate_by_name": True}


@app.post("/api/gap-report")
def gap_report(req: GapReportRequest):
    scores = {
        "Linguistic": req.Linguistic,
        "Musical": req.Musical,
        "Bodily": req.Bodily,
        "Logical - Mathematical": req.Logical_Mathematical,
        "Spatial-Visualization": req.Spatial_Visualization,
        "Interpersonal": req.Interpersonal,
        "Intrapersonal": req.Intrapersonal,
        "Naturalist": req.Naturalist,
    }

    career = req.career
    if not career:
        ordered = features_to_vector(scores)
        career = predict_top_k(ordered, k=1)[0]["career"]

    report = analyze_gap(scores, career)
    if report is None:
        raise HTTPException(
            404,
            f"No target profile available for career {career!r}. "
            "This can happen if the career name doesn't match either "
            "the 72-career trait dataset or the tech-role resume dataset.",
        )
    return report
