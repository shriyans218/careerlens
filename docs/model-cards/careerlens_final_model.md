# Model Card: careerlens_final_model.joblib

- **Type**: calibrated soft-voting ensemble (Random Forest + XGBoost + Logistic Regression)
- **Training data**: CPDS (see dataset-cards/cpds.md)
- **Input**: 8 trait scores (0–20 each)
- **Output**: probability distribution over 72 careers
- **Held-out macro-F1**: 98.75% (vs. 97.5% single-RF baseline)
- **Bundle size**: ~11MB
- **Trained via**: `training/06_train_ensemble.py`, exported by `training/07_export_ensemble.py`
- **Known limitations**: only sees 8 broad psychometric traits; resume-derived scores come from a keyword heuristic.
