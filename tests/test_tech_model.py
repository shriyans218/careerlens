import joblib
from pathlib import Path

from backend.app.tech_model import predict_tech_roles

TECH_MODEL_PATH = Path(__file__).resolve().parent.parent / "model_out" / "careerlens_tech_model.joblib"


def test_predict_tech_roles(sample_resume_text):
    if not TECH_MODEL_PATH.is_file():
        return
    bundle = joblib.load(TECH_MODEL_PATH)
    preds = predict_tech_roles(bundle, sample_resume_text, k=5)
    assert len(preds) == 5
