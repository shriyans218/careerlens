"""
STEP 6: Improve on the single-Random-Forest baseline (97.5% macro-F1,
22MB) with a calibrated soft-voting ensemble of RF + XGBoost + Logistic
Regression. Soft voting averages class probabilities across 3 different
model families instead of trusting one algorithm's (often overconfident)
probability output -- this tends to both raise held-out macro-F1 and
produce better-calibrated, less spiky confidence percentages.

Run from careerlens/training/, with careerlens/data/cpds_clean.csv present.
"""
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["Linguistic", "Musical", "Bodily", "Logical - Mathematical",
            "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]

df = pd.read_csv(ROOT / "data" / "cpds_clean.csv")
X = df[FEATURES]
y_raw = df["Job profession"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


def eval_model(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    macro_f1 = f1_score(y_te, preds, average="macro")
    acc = accuracy_score(y_te, preds)
    print(f"{name:28s} macro-F1={macro_f1:.4f}  acc={acc:.4f}  "
          f"fit={time.time()-t0:.1f}s")
    return macro_f1, acc


# --- Baseline single model, for comparison ---------------------------------
rf_solo = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=1,
)
baseline_f1, baseline_acc = eval_model(
    "RF solo (current deployed)", rf_solo, X_train_s, y_train, X_test_s, y_test)

# --- Soft-voting ensemble ---------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=150, max_depth=14, min_samples_leaf=1,
    class_weight="balanced", random_state=42, n_jobs=1,
)
xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=1,
)
logreg = LogisticRegression(
    max_iter=3000, C=1.0, class_weight="balanced",
)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("logreg", logreg)],
    voting="soft",
    weights=[2, 2, 1],  # trust the tree models slightly more than linear
    n_jobs=1,
)
ensemble_f1, ensemble_acc = eval_model(
    "Soft-voting ensemble", ensemble, X_train_s, y_train, X_test_s, y_test)

# --- Calibrate the ensemble's probabilities ---------------------------------
# VotingClassifier(soft) already averages probabilities, but individual
# members (esp. RF) are notoriously overconfident. Wrap in isotonic
# calibration via cross-val on the training fold so the exported
# probabilities are closer to true confidence, not just relative ranking.
calibrated = CalibratedClassifierCV(ensemble, method="isotonic", cv=3)
calib_f1, calib_acc = eval_model(
    "Calibrated soft-voting ensemble", calibrated, X_train_s, y_train, X_test_s, y_test)

print("\n--- Summary ---")
print(f"RF solo (deployed):            macro-F1={baseline_f1:.4f}")
print(f"Soft-voting ensemble:          macro-F1={ensemble_f1:.4f}")
print(f"Calibrated soft-voting:        macro-F1={calib_f1:.4f}")

candidates = {
    "rf_solo": (baseline_f1, rf_solo),
    "ensemble": (ensemble_f1, ensemble),
    "calibrated": (calib_f1, calibrated),
}
winner_name = max(candidates, key=lambda k: candidates[k][0])
print(f"\n>>> WINNER: {winner_name} (macro-F1={candidates[winner_name][0]:.4f}) <<<")

joblib.dump(
    {"winner": winner_name, "scores": {k: v[0] for k, v in candidates.items()}},
    ROOT / "model_out" / "ensemble_eval.joblib",
)
