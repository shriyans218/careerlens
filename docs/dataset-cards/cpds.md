# Dataset Card: CPDS (Career Path Dataset — Multiple Intelligences)

- **File(s)**: `data/cpds.xlsx`, `data/cpds_clean.csv`
- **Size**: 3,600 rows, 72 balanced career classes (50 rows/class)
- **Features**: 8 multiple-intelligence trait scores (Linguistic,
  Musical, Bodily, Logical-Mathematical, Spatial-Visualization,
  Interpersonal, Intrapersonal, Naturalist)
- **Label**: career title (72 classes)
- **Use**: primary training data for the deployed soft-voting ensemble
  (`careerlens_final_model.joblib`)
- **Known limitations**: trait scores are self-reported/synthetic
  multiple-intelligence measures, not validated psychometric
  instruments; resume-derived trait scores (via
  `feature_extraction.py`) are a keyword heuristic, not a real test.
- **Held-out macro-F1**: LR 91.1%, RF 98.5%, XGBoost 98.3%; deployed
  ensemble 98.75%.
