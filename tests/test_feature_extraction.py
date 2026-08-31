from backend.app.feature_extraction import extract_features, features_to_vector, FEATURE_ORDER


def test_extract_features_returns_all_traits(sample_resume_text):
    scores = extract_features(sample_resume_text)
    assert len(scores) == len(FEATURE_ORDER)


def test_extract_features_scores_in_range(sample_resume_text):
    scores = extract_features(sample_resume_text)
    for v in scores.values():
        assert 0 <= v <= 20


def test_features_to_vector_length(sample_resume_text):
    scores = extract_features(sample_resume_text)
    vector = features_to_vector(scores)
    assert len(vector) == len(FEATURE_ORDER)
