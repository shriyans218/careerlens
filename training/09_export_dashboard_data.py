"""
STEP 9: Export real data for the analytics dashboard.

Produces model_out/dashboard_data.json containing:
  1. model_comparison: the actual macro-F1 scores logged during
     04_train_cpds.py's grid search (Logistic Regression / Random
     Forest / XGBoost), not invented numbers.
  2. embeddings: a real t-SNE projection of every training example's
     8-dim trait vector (the same StandardScaler-transformed features
     the deployed model uses) down to 2D, grouped into a handful of
     broad career clusters derived from the actual 72 job labels.

The backend serves this file as-is (static, since it's a property of
the trained model / dataset, not of any individual user). A user's
own resume vector is projected into the same 2D space per-request in
main.py using the fitted PCA/TSNE-adjacent transform saved here.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cpds_clean.csv"
OUT_PATH = ROOT / "model_out" / "dashboard_data.json"
CPDS_TRAIN_LOG = ROOT / "model_out" / "cpds_train.log"

FEATURES = [
    "Linguistic", "Musical", "Bodily", "Logical - Mathematical",
    "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist",
]

# Broad groupings of the 72 fine-grained job labels, used only for
# scatter-plot coloring/legend. Every label in the dataset must appear
# exactly once below; anything unmapped falls into "Other".
CLUSTER_MAP = {
    "Technology & Engineering": [
        "Engineer", "Computer analyst", "Computer programmer",
        "Database designer", "Technician", "Business Analyst",
    ],
    "Science & Research": [
        "Astronomer", "Geologist", "Marine Biologist", "Physicist",
        "Mathematician", "Anthropologist", "Archeologist", "Research analyst",
        "Historian", "Philosopher",
    ],
    "Healthcare": [
        "Veterinarian", "Physician", "Medical", "Pharmacist",
        "Physical Therapist", "Audiologist",
        "Para Medical (physiotherapy, occupational theropy, audio and speech language theropy, nursing)",
        "Psychologist", "Counselor",
    ],
    "Business & Finance": [
        "Actuary", "Banking", "Business manager", "Chartered Accountant",
        "Chief financial officer", "Company secretary", "Economist",
        "Financial Advisor", "Stock Broker", "Internal auditor",
        "Consultant", "Manager", "Logistics manager", "Sales Representative",
        "Marketing",
    ],
    "Creative & Arts": [
        "Artist", "Actor / Actress", "Dancer", "Fashion Designer",
        "Graphic Designer", "Interior Decorator", "Music teacher",
        "Nature photographer", "Poet", "Recording engineer", "Sound editor",
        "Writer", "Editor", "Journalist", "Broadcaster",
    ],
    "Public Service & Law": [
        "Lawyer", "Politician", "Police Force (Spies, CBI officials, CID, Detectives)",
        "Militry", "Para Militry (https://en.wikipedia.org/wiki/Paramilitary_forces_of_India)",
        "Criminologist", "Social Worker", "Librarian", "Receptionist",
    ],
    "Education": [
        "Language Teacher", "Primary Teacher", "Pre Primary Teacher (2020 NEP and Mental Health)",
        "Middle, Higher School Teacher and Professors",
    ],
    "Other": ["Leader", "Pilot", "Athlete"],
}
LABEL_TO_CLUSTER = {
    label: cluster for cluster, labels in CLUSTER_MAP.items() for label in labels
}


def parse_model_comparison():
    """Re-derive the real macro-F1 numbers already logged during
    04_train_cpds.py's grid search, so the dashboard shows the actual
    result instead of a guess. Falls back to hardcoded values (copied
    verbatim from that log) if the log file isn't present."""
    fallback = [
        {"name": "Logistic Regression", "macro_f1": 0.9109, "accuracy": 0.9139},
        {"name": "Random Forest", "macro_f1": 0.9847, "accuracy": 0.9847},
        {"name": "XGBoost", "macro_f1": 0.9834, "accuracy": 0.9833},
    ]
    if not CPDS_TRAIN_LOG.is_file():
        return fallback
    text = CPDS_TRAIN_LOG.read_text()
    import re
    blocks = re.split(r"=== (.+?) ===", text)[1:]
    results = []
    for i in range(0, len(blocks), 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        f1_match = re.search(r"Test macro-F1:\s+([\d.]+)", body)
        acc_match = re.search(r"Test accuracy:\s+([\d.]+)", body)
        if f1_match and acc_match:
            results.append({
                "name": name,
                "macro_f1": round(float(f1_match.group(1)), 4),
                "accuracy": round(float(acc_match.group(1)), 4),
            })
    return results or fallback


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values
    labels = df["Job profession"].values
    clusters = np.array([LABEL_TO_CLUSTER.get(l, "Other") for l in labels])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE on the full 3600-row training set. perplexity=30 is a
    # reasonable default for this size; random_state fixed for
    # reproducible layout across exports.
    tsne = TSNE(
        n_components=2, perplexity=30, random_state=42,
        init="pca", learning_rate="auto",
    )
    coords = tsne.fit_transform(X_scaled)

    points = [
        {
            "x": round(float(coords[i, 0]), 3),
            "y": round(float(coords[i, 1]), 3),
            "cluster": str(clusters[i]),
            "job": str(labels[i]),
        }
        for i in range(len(df))
    ]

    # Also fit a PCA(2) on the same scaled features. t-SNE has no
    # `.transform()` for new points, so a per-request resume vector is
    # projected with this PCA instead and shown on the same chart —
    # an approximation, clearly labeled as such in the API response.
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_scaled)

    joblib.dump(
        {"scaler": scaler, "pca": pca, "features": FEATURES},
        ROOT / "model_out" / "dashboard_projection.joblib",
    )

    payload = {
        "model_comparison": parse_model_comparison(),
        "embeddings": {
            "method": "t-SNE (perplexity=30) on training set; new points "
                      "projected via PCA(2) fit on the same scaled features "
                      "as an approximation, since t-SNE has no transform() "
                      "for out-of-sample points",
            "clusters": list(CLUSTER_MAP.keys()),
            "points": points,
        },
    }
    OUT_PATH.write_text(json.dumps(payload))
    print(f"Wrote {OUT_PATH} ({len(points)} points, "
          f"{len(payload['model_comparison'])} model results)")


if __name__ == "__main__":
    main()
