# Model Card: careerlens_tech_model.joblib

- **Type**: TF-IDF + LinearSVC classifier
- **Training data**: resume_categories.csv (see dataset-cards/resume-categories.md)
- **Input**: raw resume text (resume-upload path only)
- **Output**: probability-ranked list over 25 job categories
- **Held-out macro-F1**: 99.45% — treat with skepticism (near-duplicate training resumes inflate this).
- **Trained via**: `training/08_train_tech_model.py`
- **Known limitations**: only 25 categories; optional (returns `null` if bundle absent).
