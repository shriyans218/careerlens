"""
STEP 4: Same pipeline as CPP, applied to CPDS (multiple-intelligence
dataset). Explore -> split -> preprocess -> tune 3 models -> compare.
"""
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("/home/claude/careerlens/data/cpds_clean.csv")
FEATURES = ["Linguistic", "Musical", "Bodily", "Logical - Mathematical",
            "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist"]

print("Shape:", df.shape, "| classes:", df["Job profession"].nunique())

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

results = {}

def evaluate(name, grid):
    t0 = time.time()
    grid.fit(X_train_s, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test_s)
    macro_f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print("Best params:", grid.best_params_)
    print(f"CV best macro-F1: {grid.best_score_:.4f}")
    print(f"Test macro-F1:    {macro_f1:.4f}")
    print(f"Test accuracy:    {acc:.4f}")
    print(f"Fit time: {time.time()-t0:.1f}s")
    results[name] = {"model": best, "macro_f1": macro_f1, "accuracy": acc,
                      "best_params": grid.best_params_}

evaluate("Logistic Regression", GridSearchCV(
    LogisticRegression(max_iter=3000, class_weight="balanced"),
    {"C": [0.1, 1.0, 5.0]}, scoring="f1_macro", cv=5, n_jobs=1))

evaluate("Random Forest", GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
    {"n_estimators": [200, 300], "max_depth": [None, 15]},
    scoring="f1_macro", cv=5, n_jobs=1))

evaluate("XGBoost", GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="mlogloss", tree_method="hist", n_jobs=1),
    {"n_estimators": [200, 300], "max_depth": [4, 6]},
    scoring="f1_macro", cv=5, n_jobs=1))

winner = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\n\n>>> WINNER on CPDS dataset: {winner} (test macro-F1 = {results[winner]['macro_f1']:.4f}) <<<")

joblib.dump(
    {"results": results, "winner": winner, "scaler": scaler,
     "label_encoder": label_encoder, "features": FEATURES},
    "/home/claude/careerlens/model_out/cpds_results.joblib",
)
