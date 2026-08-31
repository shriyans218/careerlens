from backend.app.resume_parser import parse_resume_entities


def test_parse_resume_entities_finds_known_skill(sample_resume_text):
    entities = parse_resume_entities(sample_resume_text)
    assert isinstance(entities, list)
    texts = [e.get("text", "").lower() for e in entities]
    assert any("python" in t or "docker" in t for t in texts)
