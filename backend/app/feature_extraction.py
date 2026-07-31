"""
Maps free-text resume content to the 8 multiple-intelligence scores the
CareerLens model was trained on. There is no ground-truth way to derive
real psychometric scores from a resume (that requires an actual test) —
this is a transparent, keyword-signal heuristic: how strongly does the
resume's language associate with each intelligence domain.

Score scale matches training data: roughly 0-20, mean ~9-15 per domain.
"""
import re

# Keyword banks per intelligence domain. Kept broad but domain-specific
# to reduce cross-contamination between categories.
KEYWORDS = {
    "Linguistic": [
        "writing", "written", "editor", "editing", "journalist", "journalism",
        "content", "communication", "blog", "author", "translat", "speech",
        "storytelling", "copywriting", "copywriter", "grammar", "publishing",
        "technical writing", "documentation", "narrative", "proofread",
    ],
    "Musical": [
        "music", "composer", "composition", "singer", "singing", "instrument",
        "audio engineer", "sound design", "orchestra", "dj ", "songwriter",
        "band", "rhythm", "melody", "musician", "conductor", "choir",
    ],
    "Bodily": [
        "sports", "athlete", "athletic", "coach", "physical therapy",
        "physiotherapy", "dance", "dancer", "fitness", "trainer", "yoga",
        "martial arts", "surgeon", "surgery", "mechanic", "craftsman",
        "manual dexterity", "hands-on", "carpentry", "physical education",
    ],
    "Logical - Mathematical": [
        "data analysis", "statistics", "statistical", "algorithm",
        "programming", "software engineer", "engineering", "finance",
        "accounting", "mathematics", "mathematical", "research", "science",
        "quantitative", "coding", "python", "sql", "machine learning",
        "logic", "analytics", "modeling", "optimization",
    ],
    "Spatial-Visualization": [
        "design", "architecture", "architectural", "cad", "graphic design",
        "ux", "ui", "user experience", "3d modeling", "photography",
        "art direction", "visualization", "mapping", "gis", "illustrator",
        "layout", "rendering", "interior design", "animation",
    ],
    "Interpersonal": [
        "sales", "teamwork", "leadership", "management", "customer service",
        "human resources", "hr ", "teaching", "counseling", "negotiation",
        "public relations", "mentoring", "collaboration", "stakeholder",
        "client relations", "team lead", "supervis",
    ],
    "Intrapersonal": [
        "self-motivated", "independent researcher", "entrepreneur",
        "freelancer", "self-directed", "personal development", "therapist",
        "introspect", "philosophy", "reflective practice", "solo",
        "autonomous", "self-taught", "founder",
    ],
    "Naturalist": [
        "environmental science", "biology", "biological", "agriculture",
        "ecology", "ecological", "wildlife", "sustainability", "geology",
        "marine biology", "botany", "outdoor", "conservation",
        "environmental", "climate", "forestry", "horticulture",
    ],
}

FEATURE_ORDER = [
    "Linguistic", "Musical", "Bodily", "Logical - Mathematical",
    "Spatial-Visualization", "Interpersonal", "Intrapersonal", "Naturalist",
]

# Training-data mean/std per domain (from cpds_clean.csv). Real profiles
# are elevated across SEVERAL dimensions at once, not just one maxed-out
# axis — so instead of mapping keyword hits directly onto a 0-20 scale
# (which produces unrealistic, out-of-distribution vectors: one feature
# at ceiling, everything else at floor), we treat keyword hits as
# RELATIVE signal strength across domains and place each domain's score
# around its own training mean, shifted up/down by how strong that
# domain's signal is relative to the resume's other domains.
TRAIN_MEAN = {
    "Linguistic": 13.06, "Musical": 9.54, "Bodily": 12.08,
    "Logical - Mathematical": 15.51, "Spatial-Visualization": 9.77,
    "Interpersonal": 15.55, "Intrapersonal": 14.76, "Naturalist": 11.04,
}
TRAIN_STD = {
    "Linguistic": 3.71, "Musical": 4.27, "Bodily": 4.30,
    "Logical - Mathematical": 3.84, "Spatial-Visualization": 3.72,
    "Interpersonal": 3.52, "Intrapersonal": 3.59, "Naturalist": 4.32,
}
FEATURE_MIN, FEATURE_MAX = 0.0, 20.0


def extract_features(resume_text: str) -> dict:
    """Returns dict of {feature_name: score 0-20} from resume text,
    centered on realistic training-data distributions."""
    text = resume_text.lower()
    raw_hits = {}
    for domain, keywords in KEYWORDS.items():
        count = 0
        for kw in keywords:
            kw = kw.strip()
            # \b boundaries prevent substring false-positives, e.g. bare
            # "ui" matching inside "building", or "design" matching inside
            # "designed a system" (engineering sense, not graphic design).
            pattern = r"\b" + re.escape(kw).replace(r"\ ", r"\s+") + r"\b"
            count += len(re.findall(pattern, text))
        raw_hits[domain] = count

    hit_values = list(raw_hits.values())
    avg_hits = sum(hit_values) / len(hit_values)
    variance = sum((h - avg_hits) ** 2 for h in hit_values) / len(hit_values)
    std_hits = variance ** 0.5 or 1.0  # avoid div-by-zero when no keywords hit at all

    scores = {}
    for domain, count in raw_hits.items():
        z = (count - avg_hits) / std_hits  # relative strength within THIS resume
        # Cap z so one extreme keyword-stuffed domain can't blow past
        # realistic bounds; +/-2 std already reaches near training min/max.
        z = max(min(z, 2.2), -1.5)
        score = TRAIN_MEAN[domain] + z * TRAIN_STD[domain]
        score = max(FEATURE_MIN, min(FEATURE_MAX, score))
        scores[domain] = round(score, 2)
    return scores


def features_to_vector(scores: dict) -> list:
    return [scores[f] for f in FEATURE_ORDER]
