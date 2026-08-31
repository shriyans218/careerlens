from backend.app.gap_analysis import analyze_gap, list_known_careers


def test_technical_gaps_independently_capped(sample_resume_text):
    careers = list_known_careers()
    assert careers
    scores = {
        "Linguistic": 10, "Musical": 5, "Bodily": 5,
        "Logical - Mathematical": 15, "Spatial-Visualization": 10,
        "Interpersonal": 8, "Intrapersonal": 8, "Naturalist": 3,
    }
    report = analyze_gap(scores, careers[0], resume_text=sample_resume_text)
    if report and report.get("technical_gaps"):
        tg = report["technical_gaps"]
        have = [s for s in tg if s.get("resume_has_it")]
        missing = [s for s in tg if not s.get("resume_has_it")]
        assert len(have) <= 12
        assert len(missing) <= 12


def test_gap_report_has_trait_gaps():
    careers = list_known_careers()
    scores = {
        "Linguistic": 10, "Musical": 5, "Bodily": 5,
        "Logical - Mathematical": 15, "Spatial-Visualization": 10,
        "Interpersonal": 8, "Intrapersonal": 8, "Naturalist": 3,
    }
    report = analyze_gap(scores, careers[0], resume_text=None)
    assert "gaps" in report
