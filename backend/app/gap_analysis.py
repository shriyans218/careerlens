"""
Skill gap analysis -- compares a resume's extracted trait scores against
a real target profile for the predicted career, then produces per-trait
gaps and actionable, keyword-driven suggestions.

Two source datasets back the target profiles, so every career surfaced
anywhere in the app (both the 72-career trait model and the 25-category
tech model) can get an authentic, data-derived report -- nothing here is
guessed or hand-typed per request:

1. data/cpds_clean.csv -- the trait-survey dataset the main 72-career
   model was trained on. Column means per career give the target profile
   directly.
2. data/resume_categories.csv -- real resumes labeled by tech category
   (the same data training/08_train_tech_model.py trains on). For each
   category we run the SAME extract_features() keyword-scoring used on
   the user's own resume, then average across that category's resumes.
   This keeps the two sides of the comparison apples-to-apples: both are
   derived by the identical scoring function, just on different resumes.

If a predicted career exists in neither source (typos, or a label that's
since changed), analyze_gap returns None and the caller should respond
with a clear "not available" rather than a fabricated report.
"""
from pathlib import Path

import pandas as pd

from .feature_extraction import FEATURE_ORDER, KEYWORDS, extract_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CPDS_PATH = PROJECT_ROOT / "data" / "cpds_clean.csv"
RESUME_CATEGORIES_PATH = PROJECT_ROOT / "data" / "resume_categories.csv"

MIN_CATEGORY_SAMPLES = 5  # below this, a tech category's average is too noisy to trust

# Lazily computed, cached at module scope so repeated requests are fast.
_survey_profiles = None       # {career: {trait: mean_score}}  -- from cpds_clean.csv
_tech_profiles = None         # {category: {trait: mean_score}} -- from resume_categories.csv
_tech_sample_counts = None    # {category: n_resumes_used}


def _load_survey_profiles() -> dict:
    global _survey_profiles
    if _survey_profiles is not None:
        return _survey_profiles
    df = pd.read_csv(CPDS_PATH)
    grouped = df.groupby("Job profession")[FEATURE_ORDER].mean()
    _survey_profiles = {
        career: {trait: round(float(row[trait]), 2) for trait in FEATURE_ORDER}
        for career, row in grouped.iterrows()
    }
    return _survey_profiles


def _load_tech_profiles() -> dict:
    """Builds target trait profiles for tech resume categories by scoring
    every real resume in that category with the same extract_features()
    logic used on the user's resume, then averaging. Computed once and
    cached -- this touches ~960 resumes so it's worth not repeating per
    request, but it's cheap enough (~1-2s) to not need persisting to disk."""
    global _tech_profiles, _tech_sample_counts
    if _tech_profiles is not None:
        return _tech_profiles

    df = pd.read_csv(RESUME_CATEGORIES_PATH)
    profiles = {}
    counts = {}
    for category, group in df.groupby("Category"):
        resumes = group["Resume"].dropna().astype(str).tolist()
        if len(resumes) < MIN_CATEGORY_SAMPLES:
            continue  # too few samples to trust an average
        per_resume_scores = [extract_features(text) for text in resumes]
        profile = {
            trait: round(sum(s[trait] for s in per_resume_scores) / len(per_resume_scores), 2)
            for trait in FEATURE_ORDER
        }
        profiles[category] = profile
        counts[category] = len(resumes)

    _tech_profiles = profiles
    _tech_sample_counts = counts
    return _tech_profiles


# A few representative keywords per trait, reused from feature_extraction's
# keyword banks, to turn "you're behind on Logical-Mathematical" into a
# concrete suggestion instead of just a number.
_SUGGESTION_HINTS = {trait: keywords[:4] for trait, keywords in KEYWORDS.items()}

_TRAIT_LABELS = {
    "Linguistic": "written communication and language skills",
    "Musical": "musical / auditory skills",
    "Bodily": "hands-on / physical-practical skills",
    "Logical - Mathematical": "analytical, quantitative and technical skills",
    "Spatial-Visualization": "design and visual/spatial skills",
    "Interpersonal": "collaboration and people-facing skills",
    "Intrapersonal": "self-direction and independent ownership",
    "Naturalist": "environmental/domain-observation skills",
}


def get_career_profile(career: str):
    """Returns (profile_dict, source_label) for a career/category, checking
    the trait-survey data first, then the tech-resume data. Returns
    (None, None) if the career isn't found in either -- callers should
    treat that as "no report available", not silently fall back."""
    survey = _load_survey_profiles()
    if career in survey:
        return survey[career], "trait_survey"

    tech = _load_tech_profiles()
    if career in tech:
        return tech[career], "tech_resumes"

    return None, None


def analyze_gap(resume_scores: dict, career: str):
    """
    resume_scores: {trait: score} as produced by feature_extraction.extract_features
    career: target career/category name (usually predictions[0]["career"]
            from either the trait model or the tech model)

    Returns None if no target profile exists for this career in either
    data source. Otherwise returns:
      {
        "career": ...,
        "source": "trait_survey" | "tech_resumes",
        "overall_readiness": 0-100 float,
        "gaps": [ {trait, resume_score, target_score, gap, severity,
                    suggestion}, ... ]  # sorted, biggest gap first
      }
    """
    profile, source = get_career_profile(career)
    if profile is None:
        return None

    gaps = []
    for trait in FEATURE_ORDER:
        resume_val = float(resume_scores.get(trait, 0.0))
        target_val = float(profile[trait])
        gap = round(target_val - resume_val, 2)

        if gap >= 4:
            severity = "high"
        elif gap >= 1.5:
            severity = "medium"
        elif gap > -1.5:
            severity = "on_track"
        else:
            severity = "strength"

        suggestion = None
        if severity in ("high", "medium"):
            hints = _SUGGESTION_HINTS.get(trait, [])
            hint_text = ", ".join(hints) if hints else trait.lower()
            suggestion = (
                f"Strengthen {_TRAIT_LABELS.get(trait, trait)}. "
                f"Consider highlighting or gaining experience in areas like: {hint_text}."
            )

        gaps.append({
            "trait": trait,
            "resume_score": round(resume_val, 2),
            "target_score": target_val,
            "gap": gap,
            "severity": severity,
            "suggestion": suggestion,
        })

    gaps.sort(key=lambda g: g["gap"], reverse=True)

    # Simple readiness score: how close overall, capped 0-100. Only
    # positive gaps (areas you're behind) count against readiness --
    # being ahead on a trait doesn't further inflate the score.
    total_gap = sum(max(g["gap"], 0) for g in gaps)
    max_possible_gap = 20 * len(FEATURE_ORDER)
    readiness = round(100 * (1 - total_gap / max_possible_gap), 1)
    readiness = max(0.0, min(100.0, readiness))

    return {
        "career": career,
        "source": source,
        "overall_readiness": readiness,
        "gaps": gaps,
    }
