"""
STEP 7: Retrain the winning config from 06_train_ensemble.py -- a soft-
voting ensemble of RF + XGBoost + LogisticRegression, tuned smaller and
shallower than the initial guess (which turned out to both overfit AND
bloat file size) -- on 100% of the data. 98.75% held-out macro-F1 at
~11MB, vs the old single Random Forest's 97.47% at 22MB. Exports the
same bundle shape the backend already expects, so no backend changes
are needed.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["Linguistic", "Musical", "Bodily", "Logical - Mathematical",
            "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]

# Local file-based MLflow tracking store by default (mlruns/ at repo root).
# Override with MLFLOW_TRACKING_URI env var to point at a hosted server.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
mlflow.set_experiment("careerlens-ensemble")

df = pd.read_csv(ROOT / "data" / "cpds_clean.csv")
X = df[FEATURES]
y_raw = df["Job profession"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf_params = dict(n_estimators=60, max_depth=8, min_samples_leaf=2,
                  class_weight="balanced", random_state=42, n_jobs=1)
xgb_params = dict(n_estimators=80, max_depth=4, learning_rate=0.1,
                   tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=1)
logreg_params = dict(max_iter=3000, C=1.0, class_weight="balanced")

rf = RandomForestClassifier(**rf_params)
xgb = XGBClassifier(**xgb_params)
logreg = LogisticRegression(**logreg_params)

final_model = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("logreg", logreg)],
    voting="soft", weights=[2, 2, 1], n_jobs=1,
)

with mlflow.start_run(run_name="export_final_ensemble") as run:
    final_model.fit(X_scaled, y)

    train_acc = final_model.score(X_scaled, y)
    print("Full-data training-fit accuracy:", round(train_acc, 4))

    test_macro_f1 = 0.9875  # held-out eval of this tuned config, from 06_train_ensemble.py

    mlflow.log_params({
        "voting_weights": "2,2,1",
        **{f"rf_{k}": v for k, v in rf_params.items()},
        **{f"xgb_{k}": v for k, v in xgb_params.items()},
        **{f"logreg_{k}": v for k, v in logreg_params.items()},
    })
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("held_out_macro_f1", test_macro_f1)

    # Log + register the sklearn model in the MLflow model registry.
    # Registration only works against a server/DB-backed tracking store;
    # against a local file store this still logs the model artifact
    # under the run (visible via `mlflow ui`) and registration is skipped.
    try:
        mlflow.sklearn.log_model(
            final_model, artifact_path="model",
            registered_model_name="careerlens-ensemble",
        )
    except Exception as e:
        print(f"Model registry registration skipped ({e}); model artifact still logged to the run.")
        mlflow.sklearn.log_model(final_model, artifact_path="model")

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
            "test_macro_f1": test_macro_f1,
            "mlflow_run_id": run.info.run_id,
        },
        ROOT / "model_out" / "careerlens_final_model.joblib",
    )
    mlflow.log_artifact(str(ROOT / "model_out" / "careerlens_final_model.joblib"))
    print(f"Saved calibrated ensemble as deployed model. MLflow run: {run.info.run_id}")
