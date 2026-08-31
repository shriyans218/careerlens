import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_resume_text():
    return (
        "Experienced Python developer skilled in Django, SQL, Docker, "
        "AWS, and React. Built and deployed CI/CD pipelines using "
        "Kubernetes and Jenkins. Strong background in machine learning "
        "with scikit-learn and TensorFlow."
    )
