def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_careers(client):
    r = client.get("/api/careers")
    assert r.status_code == 200
    assert isinstance(r.json()["careers"], list)


def test_predict_scores(client):
    payload = {
        "Linguistic": 10, "Musical": 5, "Bodily": 5,
        "Logical_Mathematical": 15, "Spatial_Visualization": 10,
        "Interpersonal": 8, "Intrapersonal": 8, "Naturalist": 3,
    }
    r = client.post("/api/predict-scores", json=payload)
    assert r.status_code == 200
    assert len(r.json()["top_5"]) == 5


def test_predict_resume(client):
    files = {"file": ("resume.txt", b"Python developer skilled in Django and SQL.", "text/plain")}
    r = client.post("/api/predict-resume", files=files)
    assert r.status_code == 200
    body = r.json()
    assert "top_5" in body
    assert "trait_scores" in body


def test_predict_resume_rejects_empty_file(client):
    files = {"file": ("empty.txt", b"", "text/plain")}
    r = client.post("/api/predict-resume", files=files)
    assert r.status_code == 400


def test_gap_report_unresolvable_career(client):
    payload = {
        "Linguistic": 10, "Musical": 5, "Bodily": 5,
        "Logical - Mathematical": 15, "Spatial-Visualization": 10,
        "Interpersonal": 8, "Intrapersonal": 8, "Naturalist": 3,
        "career": "definitely-not-a-real-career-xyz",
    }
    r = client.post("/api/gap-report", json=payload)
    assert r.status_code == 404
