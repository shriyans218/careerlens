"""Skill-based tech-role predictions, using a TF-IDF + LinearSVC model
trained directly on resume text (see training/08_train_tech_model.py).
Runs alongside the trait model as a second, tech-specific opinion --
does not replace it.
"""
import re


def clean_resume_text(text: str) -> str:
    # Must match the cleaning in training/08_train_tech_model.py exactly,
    # or the vectorizer sees out-of-distribution tokens at inference time.
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+.# ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def predict_tech_roles(tech_bundle: dict, resume_text: str, k: int = 5):
    if tech_bundle is None:
        return None
    vectorizer = tech_bundle["vectorizer"]
    model = tech_bundle["model"]
    label_encoder = tech_bundle["label_encoder"]

    cleaned = clean_resume_text(resume_text)
    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]

    top_idx = proba.argsort()[::-1][:k]
    return [
        {"career": label_encoder.classes_[i], "confidence": round(float(proba[i]), 4)}
        for i in top_idx
    ]
