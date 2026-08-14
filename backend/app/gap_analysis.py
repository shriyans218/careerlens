"""
Skill gap analysis — compares a resume's extracted trait scores against
the average trait profile of people in the predicted (or any requested)
career, using the same training data (data/cpds_clean.csv) the model
was trained on. Produces per-trait gaps plus actionable, keyword-driven
suggestions for closing the biggest ones.

Nothing here is invented per-request: career profiles are real column
means computed once from the training CSV at import time.
"""
from pathlib import Path

import pandas as pd

from .feature_extraction import FEATURE_ORDER, KEYWORDS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CPDS_PATH = PROJECT_ROOT / "data" / "cpds_clean.csv"

# {career: {trait: mean_score}}, computed once at import time.
_career_profiles = None


def _load_career_profiles() -> dict:
    global _career_profiles
    if _career_profiles is not None:
        return _career_profiles
    df = pd.read_csv(CPDS_PATH)
    grouped = df.groupby("Job profession")[FEATURE_ORDER].mean()
    _career_profiles = {
        career: {trait: round(float(row[trait]), 2) for trait in FEATURE_ORDER}
        for career, row in grouped.iterrows()
    }
    return _career_profiles


# A few representative keywords per trait, reused from feature_extraction's
# keyword banks, to turn "you're behind on Logical-Mathematical" into a
# concrete suggestion instead of just a number.
_SUGGESTION_HINTS = {
    trait: keywords[:4] for trait, keywords in KEYWORDS.items()
}

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


def get_career_profile(career: str) -> dict:
    """Returns the {trait: mean_score} profile for a career, or None if
    the career isn't in the training data (e.g. came only from the tech
    model, which uses a different label space)."""
    profiles = _load_career_profiles()
    return profiles.get(career)


def analyze_gap(resume_scores: dict, career: str) -> dict:
    """
    resume_scores: {trait: score} as produced by feature_extraction.extract_features
    career: target career name (usually predictions[0]["career"])

    Returns None if we have no training profile for this career (e.g.
    it only came from the tech-role model). Otherwise returns:
      {
        "career": ...,
        "overall_readiness": 0-100 float,
        "gaps": [ {trait, resume_score, target_score, gap, severity,
                    suggestion} , ... ]  # sorted, biggest gap first
      }
    """
    profile = get_career_profile(career)
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

    # Simple readiness score: how close overall, capped 0-100.
    total_gap = sum(max(g["gap"], 0) for g in gaps)
    max_possible_gap = 20 * len(FEATURE_ORDER)
    readiness = round(100 * (1 - total_gap / max_possible_gap), 1)
    readiness = max(0.0, min(100.0, readiness))

    return {
        "career": career,
        "overall_readiness": readiness,
        "gaps": gaps,
    }
