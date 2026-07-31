"""
STEP 3: Train + tune three model families, compare on the SAME held-out
test set, pick a winner.

Why GridSearchCV: manually guessing hyperparameters (e.g. "let's try
max_depth=5") is how people accidentally overfit or underfit. GridSearchCV
tries every combination in the grid, evaluates each with k-fold CV on the
training data (never touching the test set), and reports the best.

Why macro F1 as the scoring metric: with 90 classes of similar size here
it doesn't matter much, but macro F1 is the right default for
classification problems where you care about every class, not just the
frequent ones. It's the arithmetic mean of each class's own F1 score.
"""
import time
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from xgboost import XGBClassifier

bundle = joblib.load("/home/claude/careerlens/model_out/cpp_split.joblib")
X_train, X_test = bundle["X_train"], bundle["X_test"]
y_train, y_test = bundle["y_train"], bundle["y_test"]
label_encoder = bundle["label_encoder"]

results = {}

def evaluate(name, grid_search, X_test, y_test):
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    fit_time = time.time() - t0
    best = grid_search.best_estimator_
    preds = best.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {name} ===")
    print("Best params:", grid_search.best_params_)
    print(f"CV best macro-F1 (train folds): {grid_search.best_score_:.4f}")
    print(f"Held-out test macro-F1:         {macro_f1:.4f}")
    print(f"Held-out test accuracy:         {acc:.4f}")
    print(f"Fit time: {fit_time:.1f}s")
    results[name] = {
        "model": best, "macro_f1": macro_f1, "accuracy": acc,
        "best_params": grid_search.best_params_,
    }

# ---- 1. Logistic Regression (baseline) ----
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=3000, class_weight="balanced"),
    param_grid={"C": [0.1, 1.0, 5.0, 10.0]},
    scoring="f1_macro", cv=5, n_jobs=-1,
)
evaluate("Logistic Regression", lr_grid, X_test, y_test)

# ---- 2. Random Forest ----
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=1),
    param_grid={
        "n_estimators": [200, 300],
        "max_depth": [None, 20],
    },
    scoring="f1_macro", cv=3, n_jobs=1,
)
evaluate("Random Forest", rf_grid, X_test, y_test)

# ---- 3. XGBoost ----
xgb_grid = GridSearchCV(
    XGBClassifier(
        random_state=42, eval_metric="mlogloss",
        tree_method="hist", n_jobs=1,
    ),
    param_grid={
        "n_estimators": [200, 300],
        "max_depth": [4, 6],
    },
    scoring="f1_macro", cv=3, n_jobs=1,
)
evaluate("XGBoost", xgb_grid, X_test, y_test)

# ---- Pick winner ----
winner_name = max(results, key=lambda k: results[k]["macro_f1"])
print(f"\n\n>>> WINNER on CPP dataset: {winner_name} "
      f"(test macro-F1 = {results[winner_name]['macro_f1']:.4f}) <<<")

joblib.dump(
    {"results": results, "winner": winner_name,
     "preprocessor": bundle["preprocessor"], "label_encoder": label_encoder},
    "/home/claude/careerlens/model_out/cpp_results.joblib",
)
