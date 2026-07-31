"""
STEP 8: Second model, separate from the trait-based CPDS classifier.

The trait model only ever sees 8 broad psychometric scores -- it has no
concept of "Python" or "Kubernetes", so it can't distinguish a software
engineer from an accountant beyond "both score high on Logical-
Mathematical". This model fixes that specific gap for TECH roles by
training directly on resume text -> job category, using a real labeled
dataset (962 resumes, 25 categories, incl. Data Science, DevOps Engineer,
Python Developer, Java Developer, Network Security Engineer, ETL
Developer, Hadoop, Blockchain, SAP Developer, DotNet Developer, etc.)
Source: "UpdatedResumeDataSet" (public resume-classification dataset,
originally on Kaggle, mirrored here from
github.com/DhyanilMehta/Resume-Screening-ML-Project).

This does NOT replace the trait model -- it runs alongside it as a
second, tech-specific opinion. See backend/app/main.py for how both are
combined into one API response.
"""
import re
import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "resume_categories.csv")
print("Shape:", df.shape, "| classes:", df["Category"].nunique())


def clean(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9+.# ]", " ", text)  # keep tokens like "C++", "C#", ".NET"
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


df["clean_resume"] = df["Resume"].apply(clean)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Category"])

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_resume"], y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=6000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    stop_words="english",
)
X_train_v = vectorizer.fit_transform(X_train)
X_test_v = vectorizer.transform(X_test)

# LinearSVC is a strong baseline for sparse TF-IDF text classification.
# Wrapped in CalibratedClassifierCV so we get probability-like scores
# (LinearSVC has no predict_proba natively) for the "% match" UI.
t0 = time.time()
base = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
model.fit(X_train_v, y_train)
print(f"Fit time: {time.time()-t0:.1f}s")

preds = model.predict(X_test_v)
macro_f1 = f1_score(y_test, preds, average="macro")
acc = accuracy_score(y_test, preds)
print(f"Test macro-F1: {macro_f1:.4f}")
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, preds, target_names=label_encoder.classes_, zero_division=0))

# Retrain on 100% of data for deployment
X_all_v = vectorizer.fit_transform(df["clean_resume"])
final_model = CalibratedClassifierCV(
    LinearSVC(C=1.0, class_weight="balanced", random_state=42), method="sigmoid", cv=5
)
final_model.fit(X_all_v, y)

joblib.dump(
    {
        "model": final_model,
        "vectorizer": vectorizer,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "test_macro_f1": round(macro_f1, 4),
        "test_accuracy": round(acc, 4),
        "n_train_rows": len(df),
        "source": "UpdatedResumeDataSet (public resume-classification dataset, 25 categories)",
    },
    ROOT / "model_out" / "careerlens_tech_model.joblib",
)
print("Saved tech-role skill-based model.")
