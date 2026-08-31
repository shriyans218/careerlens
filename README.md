# CareerLens

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ensemble-F7931E?logo=scikit-learn&logoColor=white)

AI-powered career-fit predictor. Given a resume or a quick self-assessment,
CareerLens predicts the closest-matching career out of **72 options** using
a calibrated soft-voting ensemble trained on multiple-intelligence trait
scores — plus a second model that reads resume text directly for
tech-role matching.

## Contents

- [What's in here](#whats-in-here)
- [Data decisions](#data-decisions--why-only-one-dataset-made-it-in)
- [Deployed model — soft-voting ensemble](#deployed-model--soft-voting-ensemble-current)
- [Second model — tech-role classifier](#second-model--tech-role-skill-classifier-added-later)
- [Analytics dashboard](#analytics-dashboard)
- [Skill-gap analysis (trait + technical)](#skill-gap-analysis-trait--technical)
- [Rerunning the training pipeline](#rerunning-the-training-pipeline-yourself-manual-step-by-step)
- [Running the backend](#running-the-backend)
- [Running the frontend](#running-the-frontend)
- [Deploying](#deploying)
- [Known limitation](#known-limitation--be-upfront-about-this)

 ## Docs

- [API Reference](docs/api-reference.md)
- [Dataset Cards](docs/dataset-cards/)
- [Model Cards](docs/model-cards/)

## What's in here

```
careerlens/
├── data/                     raw + cleaned training datasets
├── training/                 9 manual training scripts, run in order
├── model_out/                 trained model artifacts (.joblib) + logs
├── backend/                  FastAPI service that serves predictions
│   └── app/
│       ├── main.py               API routes
│       ├── feature_extraction.py   resume -> trait score heuristic
│       ├── gap_analysis.py         trait + technical skill-gap report logic
│       ├── resume_parser.py        skill/role entity extraction (also feeds gap_analysis.py)
│       └── resume_reader.py        PDF/DOCX/TXT text extraction
└── frontend/                  React (Vite) UI
    └── src/
        ├── App.jsx                main screen (upload / self-assessment / result)
        ├── Dashboard.jsx           analytics dashboard (model comparison + t-SNE map)
        ├── ApertureMark.jsx       the lens-aperture logo/loading mark
        └── App.css
```

## Data decisions — why only one dataset made it in

Four datasets were evaluated:

| Dataset | Verdict |
|---|---|
| CPP (career_path_in_all_field.csv) | ❌ All three models (Logistic Regression, Random Forest, XGBoost) scored at random-chance level (~1%). The labels don't correlate with the features — likely randomly generated. |
| CPDS (Dataset Project 404.xlsx) | ✅ **Used.** 3,600 rows, 72 balanced careers, 8 multiple-intelligence features. |
| CPD (Data_final.csv) | ❌ 105 rows, 104 unique career labels — ~1 sample per class, mathematically impossible to generalize from. |
| AICPT (personality.csv) | ❌ 2,527 rows, every single row has a unique career title — zero repeated classes. |

**Model comparison on CPDS** (held-out test macro-F1, from the actual `04_train_cpds.py` grid search):

| Model | Macro F1 |
|---|---|
| Logistic Regression | 91.1% |
| Random Forest | 98.5% |
| XGBoost | 98.3% |

Random Forest won this initial comparison and became the baseline model
(`05_export_final.py`). The deployed model was later upgraded to a
soft-voting ensemble of all three — see below.

## Deployed model — soft-voting ensemble (current)

`careerlens_final_model.joblib` (the bundle the backend actually loads)
is a **calibrated soft-voting ensemble** of Random Forest + XGBoost +
Logistic Regression, trained in `06_train_ensemble.py` /
`07_export_ensemble.py` — not the single Random Forest from the initial
comparison above. Same input features (8 trait scores), same output
shape (72 careers), so no backend changes were needed when it was
swapped in.

| | Macro F1 | Bundle size |
|---|---|---|
| Single Random Forest (baseline) | 97.5% | ~22 MB |
| Soft-voting ensemble (deployed) | **98.75%** | ~11 MB |

## Second model — tech-role skill classifier (added later)

The trait model above only ever sees 8 broad psychometric scores. It has
no concept of "Python" or "Kubernetes" — it can only tell that someone
scores high on "Logical-Mathematical", which is equally true of an
accountant, a physicist, and a software engineer. That's a structural
ceiling, not something more tuning fixes.

`careerlens_tech_model.joblib` is a second, separate model that runs
**only on the resume-upload path** (there's no text for the 8-question
self-assessment to feed it). It's a TF-IDF + LinearSVC classifier
trained directly on resume text → job category, using a real labeled
dataset:

- **Source**: "UpdatedResumeDataSet" (public resume-classification
  dataset, originally circulated via Kaggle; mirrored here from
  [DhyanilMehta/Resume-Screening-ML-Project](https://github.com/DhyanilMehta/Resume-Screening-ML-Project)).
- **962 resumes, 25 categories**, including Data Science, DevOps
  Engineer, Python Developer, Java Developer, Network Security Engineer,
  ETL Developer, Hadoop, Blockchain, SAP Developer, DotNet Developer,
  plus non-tech categories (HR, Sales, Advocate, etc.) carried along
  since they were in the source data.
- **Held-out macro-F1: 99.45%.** Be skeptical of this number, same as
  the note above about the trait model — this specific public dataset is
  known to contain many near-duplicate, template-derived resumes within
  each category, which inflates held-out scores. Treat it as "very
  strong on resumes similar in style to the training set," not as a
  guarantee on arbitrary real-world resumes.
- Retrain with `python3 training/08_train_tech_model.py`.

The API returns both: `top_5` (trait model, all 72 careers) and
`tech_top_5` (tech model, 25 categories, `null` if that bundle isn't
present). They are NOT merged into one score — shown as two separate,
clearly-labeled opinions, since they're different models trained on
different data with different meanings behind their confidence numbers.

## Skill-gap analysis (trait + technical)

`POST /api/gap-report` compares a resume/assessment against a target
career on two separate axes:

- **Trait gap** (`gaps`) — the original 8 broad multiple-intelligence
  scores (Linguistic, Logical-Mathematical, etc.) vs. the target
  career's average trait profile from CPDS. Useful as a general aptitude
  signal, but too coarse to say what to actually go learn.
- **Technical skill gap** (`technical_gaps`) — a second, more concrete
  layer added on top: it checks which specific technical skills
  (Python, SQL, Docker, React, AWS, Kubernetes, ...) — the same
  vocabulary `resume_parser.py` already highlights on the resume view —
  are common among real resumes in the target category
  (`resume_categories.csv`, the tech model's dataset) versus which of
  those skills were actually detected in the user's resume.

  Requires `resume_text` in the request body (the raw resume text
  returned as part of `/api/predict-resume`'s response) — without it,
  or for a career with no matching tech-resume category, this field is
  `null` and the report falls back to trait-only gaps.

  Each entry: `{skill, target_prevalence, resume_has_it, severity}`.
  Missing and already-have skills are ranked and capped independently
  (top 12 of each, most-common-in-target first) before being combined —
  an earlier version capped the merged list instead, which silently
  dropped a resume's matched skills off the end whenever it had more
  than ~12 combined entries, making a strong resume look like it had
  zero matching skills. Fixed in `gap_analysis.py`.

The frontend renders this as a two-column "Your Skills / Missing for
[Career]" checklist card with a skill-alignment bar (`% skills present
÷ total tracked skills for that career`), alongside the existing
trait-gap bars.

## Analytics dashboard

`09_export_dashboard_data.py` exports `model_out/dashboard_data.json`
(the real logged macro-F1 for LR/RF/XGBoost from `04`'s grid search)
and `model_out/dashboard_projection.joblib` (a fitted scaler + t-SNE-
derived 2D projection of every training example's trait vector,
grouped into broad career clusters). The backend serves this as:

- `GET /api/dashboard` — static model-comparison scores + the 2D
  training-set embedding, for charting.
- On `POST /api/predict-resume`, the response additionally includes
  `resume_point` (the current resume's own vector projected into that
  same 2D space) and `model_breakdown` (each ensemble member's
  individual prediction + confidence, for a per-model comparison view).

Both dashboard fields are additive — regenerating them is optional; the
core `/api/predict-*` prediction fields work with or without
`dashboard_data.json` / `dashboard_projection.joblib` present.

## Rerunning the training pipeline yourself (manual, step by step)

```bash
cd training
python3 01_explore_cpp.py        # inspect shape, dtypes, missing values, class balance
python3 02_preprocess_cpp.py     # stratified split + scaling (CPP — kept for reference; unusable data)
python3 03_train_cpp.py          # tune & compare LR/RF/XGBoost on CPP (confirms no signal)
python3 04_train_cpds.py         # tune & compare LR/RF/XGBoost on CPDS (the real training run)
python3 05_export_final.py       # retrain the winning single-RF config on 100% of CPDS (baseline, superseded by 07)
python3 06_train_ensemble.py     # compare a calibrated soft-voting RF+XGBoost+LR ensemble against the RF baseline
python3 07_export_ensemble.py    # retrain the winning ensemble config on 100% of CPDS — this is the deployed model
python3 08_train_tech_model.py   # TF-IDF + LinearSVC resume-text classifier (second, separate model)
python3 09_export_dashboard_data.py  # export real model-comparison scores + t-SNE projection for the dashboard
```

Key ideas to remember for next time:
- **Explore before modeling.** Check class balance first — it tells you upfront whether a real train/test split is even possible.
- **Split before you fit any transformer.** Fitting a scaler/encoder on the full dataset then splitting leaks test-set information into training.
- **Stratify your split** on the target when class sizes are small, so no class gets starved in the test set.
- **Score with macro F1, not accuracy**, when you care about every class equally rather than just the frequent ones.
- **GridSearchCV** tunes hyperparameters via cross-validation on the training set only — never touches the test set until final evaluation.
- **Retrain the winner on 100% of the data** for the deployed model once you've picked a winner via the held-out test set — the test split was only for comparison, not for the shipped model.
- **A too-good score (98%+) deserves suspicion, not celebration** — check whether the dataset is realistically noisy or suspiciously clean/synthetic before trusting it blindly.
- **Soft-voting ensembles beat a single model** here: averaging class probabilities across RF + XGBoost + Logistic Regression raised held-out macro-F1 from 97.5% to 98.75%, while also shrinking the exported bundle from ~22MB to ~11MB by tuning the ensemble smaller/shallower than the first guess.

## Running the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | sanity check |
| `GET` | `/api/dashboard` | model-comparison scores + 2D training-set embedding |
| `POST` | `/api/predict-resume` | multipart file upload (pdf/docx/txt) → top prediction + top 5, plus `resume_point` and `model_breakdown` |
| `POST` | `/api/predict-scores` | JSON body with 8 trait scores (0–20 each) → top prediction + top 5 |
| `POST` | `/api/gap-report` | JSON body: 8 trait scores + `career` + optional `resume_text` → trait-gap report (`gaps`) plus, when `resume_text` is given, a technical skill-gap report (`technical_gaps`) |

## Running the frontend

```bash
cd frontend
npm install
npm run dev          # local dev server, http://localhost:5173
npm run build         # production build -> dist/
```

Set `VITE_API_BASE` (e.g. in a `.env` file) to point at your deployed
backend URL before building for production. Defaults to
`http://localhost:8000` for local dev.

## Deploying

- **Backend**: any Python host that runs FastAPI/uvicorn — Render, Railway,
  Fly.io, or a plain VM. Make sure `model_out/careerlens_final_model.joblib`
  ships alongside the backend code (update `MODEL_PATH` in `app/main.py` if
  you move it).
- **Frontend**: `npm run build` produces a static `dist/` folder — deploy
  to Vercel, Netlify, or any static host. Set `VITE_API_BASE` to your
  live backend URL first.

## Known limitation — be upfront about this

The resume → trait-score mapping in `feature_extraction.py` is a
transparent keyword heuristic, not a real psychometric test — there's no
way to derive true multiple-intelligence scores from resume text alone.
It's calibrated to produce realistic, in-distribution score vectors (not
maxed-out single dimensions), but predictions from resumes will always be
softer signal than the quick self-assessment path, which lets the person
score themselves directly on the same 8 traits the model was trained on.
