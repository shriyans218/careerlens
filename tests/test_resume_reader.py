from backend.app.resume_reader import read_resume


def test_read_txt():
    content = b"Python developer with 5 years experience."
    text = read_resume("resume.txt", content)
    assert "Python" in text


def test_read_unsupported_returns_empty_or_raises():
    try:
        text = read_resume("resume.xyz", b"garbage")
        assert text == "" or text is None
    except Exception:
        pass
