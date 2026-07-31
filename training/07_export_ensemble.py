"""
STEP 7: Retrain the winning config from 06_train_ensemble.py -- a soft-
voting ensemble of RF + XGBoost + LogisticRegression, tuned smaller and
shallower than the initial guess (which turned out to both overfit AND
bloat file size) -- on 100% of the data. 98.75% held-out macro-F1 at
~11MB, vs the old single Random Forest's 97.47% at 22MB. Exports the
same bundle shape the backend already expects, so no backend changes
are needed.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["Linguistic", "Musical", "Bodily", "Logical - Mathematical",
            "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]

df = pd.read_csv(ROOT / "data" / "cpds_clean.csv")
X = df[FEATURES]
y_raw = df["Job profession"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(
    n_estimators=60, max_depth=8, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=1,
)
xgb = XGBClassifier(
    n_estimators=80, max_depth=4, learning_rate=0.1,
    tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=1,
)
logreg = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")

final_model = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("logreg", logreg)],
    voting="soft", weights=[2, 2, 1], n_jobs=1,
)
final_model.fit(X_scaled, y)

train_acc = final_model.score(X_scaled, y)
print("Full-data training-fit accuracy:", round(train_acc, 4))

joblib.dump(
    {
        "model": final_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "features": FEATURES,
        "feature_min": X.min().to_dict(),
        "feature_max": X.max().to_dict(),
        "classes": label_encoder.classes_.tolist(),
        "train_acc": train_acc,
        "test_macro_f1": 0.9875,  # held-out eval of this tuned config
    },
    ROOT / "model_out" / "careerlens_final_model.joblib",
)
print("Saved calibrated ensemble as deployed model.")
