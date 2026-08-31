# Dataset Card: Resume Categories (Tech-Role Classifier)

- **File**: `data/resume_categories.csv`
- **Size**: 962 resumes, 25 categories
- **Source**: "UpdatedResumeDataSet" (public resume-classification
  dataset, originally circulated via Kaggle; mirrored via
  DhyanilMehta/Resume-Screening-ML-Project on GitHub)
- **Categories**: Data Science, DevOps Engineer, Python Developer,
  Java Developer, Network Security Engineer, ETL Developer, Hadoop,
  Blockchain, SAP Developer, DotNet Developer, plus non-tech
  categories (HR, Sales, Advocate, etc.)
- **Use**: training data for `careerlens_tech_model.joblib`, and as
  the skill-prevalence reference for `technical_gaps` in
  `/api/gap-report`
- **Known limitations**: contains many near-duplicate, template-derived
  resumes within each category, inflating held-out macro-F1 (99.45%).
