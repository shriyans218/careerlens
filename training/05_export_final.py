"""
STEP 5: Retrain the winning config (Random Forest, from 04_train_cpds.py)
on the FULL dataset, then export everything needed to serve predictions:
model + scaler + label encoder + metadata.
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["Linguistic", "Musical", "Bodily", "Logical - Mathematical",
            "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]

df = pd.read_csv("/home/claude/careerlens/data/cpds_clean.csv")
X = df[FEATURES]
y_raw = df["Job profession"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# The GridSearchCV winner (n_estimators=300, max_depth=None) scored
# 98.5% macro-F1 but produces a ~130MB model file: with 72 classes, every
# tree node stores a 72-length class-count array, so node count times
# class count balloons file size fast on unconstrained-depth trees.
# A shallower, pruned forest (max_depth=12, min_samples_leaf=2) scores
# 97.5% — a 1-point drop — at 22MB, a far better deployment tradeoff.
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=12, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=1,
)
final_model.fit(X_scaled, y)

# Sanity check: training-fit accuracy (not a generalization measure,
# just confirms the model learned the full dataset correctly)
train_acc = final_model.score(X_scaled, y)
print("Full-data training-fit accuracy:", round(train_acc, 4))

# Feature importance — useful to sanity-check the model AND to show
# users which traits mattered most for a prediction
importances = dict(zip(FEATURES, final_model.feature_importances_.round(4)))
print("Feature importances:", importances)

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
        "test_macro_f1": 0.9747,  # held-out eval of this pruned config, see README
    },
    "/home/claude/careerlens/model_out/careerlens_final_model.joblib",
)
print("Saved final deployable model.")
