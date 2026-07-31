"""
Resume parsing / highlighting.

This is additive to the existing prediction pipeline: it does NOT change
feature_extraction.py, tech_model.py, or any prediction logic. It only
locates SKILL and ROLE mentions in the raw resume text so the frontend can
render a "Resume Parsing" style highlighted view.

Pure-Python regex phrase matching (same technique feature_extraction.py
already uses) -- deliberately has NO dependency on spaCy/torch, since
spaCy pulls in thinc -> torch, which can crash on some Windows/Anaconda
setups with broken CUDA/DLL installs unrelated to this project.
"""
import re

SKILL_TERMS = [
    # languages / core tech
    "Python", "Java", "JavaScript", "TypeScript", "C\\+\\+", "C#", "SQL", "R",
    "Go", "Scala", "Kotlin", "Swift", "PHP", "Ruby", "HTML", "CSS", "C(?!\\+)",
    # data / ml
    "Machine Learning", "Deep Learning", "Data Science", "Data Analysis",
    "Data Visualization", "Data Engineering", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy", "Keras",
    "Statistics", "Data Mining", "Big Data", "Hadoop", "Spark",
    # web / infra
    "React", "Node\\.js", "Django", "Flask", "FastAPI", "REST API",
    "Docker", "Kubernetes", "AWS", "Azure", "GCP", "CI/CD", "Git",
    "Linux", "DevOps", "Microservices", "Windows", "Mac OS",
    # databases
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Database Design",
    # tools
    "Photoshop", "Dreamweaver", "UltraEdit", "Rational Rose",
    # other technical
    "Blockchain", "Automation Testing", "Network Security", "ETL",
    "Business Analysis", "Project Management", "Agile", "Scrum",
    # soft skills
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Time Management", "Collaboration",
]

# Job-title / role terms: union of both trained label sets, plus a few
# common resume phrasings not present verbatim in the training labels.
ROLE_TERMS = [
    "Data Scientist Intern", "Data Scientist", "Data Analyst",
    "Business Analyst", "Software Engineer", "Software Developer",
    "Data Engineer", "Machine Learning Engineer", "DevOps Engineer",
    "Python Developer", "Java Developer", "DotNet Developer",
    "Web Designing", "Web Developer", "ETL Developer", "SAP Developer",
    "Network Security Engineer", "Database Administrator", "Database",
    "Civil Engineer", "Mechanical Engineer", "Electrical Engineering",
    "Automation Testing", "Testing", "Operations Manager", "PMO",
    "HR", "Sales", "Arts", "Advocate", "Blockchain", "Hadoop",
    "Engineer", "Manager", "Consultant", "Technician", "Researcher",
]


def _compile(terms):
    # Longest term first, so e.g. "Data Scientist Intern" is tried before
    # the shorter "Data Scientist" at the same position.
    ordered = sorted(terms, key=len, reverse=True)
    # Use lookaround boundaries instead of \b: \b fails to anchor right
    # after symbol chars like '+' or '#' (e.g. "C++" followed by a space
    # would silently never match with \b...\b), so anchor on "not
    # preceded/followed by a word character" directly instead.
    pattern = (
        r"(?<![A-Za-z0-9_])("
        + "|".join(t.replace(" ", r"\s+") for t in ordered)
        + r")(?![A-Za-z0-9_])"
    )
    return re.compile(pattern, flags=re.IGNORECASE)


_SKILL_RE = _compile(SKILL_TERMS)
_ROLE_RE = _compile(ROLE_TERMS)


def parse_resume_entities(text: str) -> list:
    """Returns a list of {start, end, label, text} spans (character
    offsets into `text`), sorted and de-overlapped so the frontend can
    render non-conflicting highlights. Longer matches win ties."""
    if not text or not text.strip():
        return []

    raw = []
    for regex, label in ((_SKILL_RE, "SKILL"), (_ROLE_RE, "ROLE")):
        for m in regex.finditer(text):
            raw.append((m.start(), m.end(), label, m.group(0)))

    # Prefer longer spans when they overlap.
    kept = []
    for start, end, label, span_text in sorted(raw, key=lambda s: -(s[1] - s[0])):
        if any(not (end <= k[0] or start >= k[1]) for k in kept):
            continue
        kept.append((start, end, label, span_text))
    kept.sort(key=lambda s: s[0])

    return [
        {"start": s, "end": e, "label": label, "text": span_text}
        for s, e, label, span_text in kept
    ]
