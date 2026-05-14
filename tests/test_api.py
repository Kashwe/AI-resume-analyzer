from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add backend folder to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_upload_resume_invalid_file():
    response = client.post(
        "/upload-resume",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files are accepted" in response.json()["detail"]


def test_match_job_empty_description():
    response = client.post(
        "/match-job",
        files={"file": ("resume.pdf", b"%PDF-1.4", "application/pdf")},
        data={"job_description": ""},
    )

    assert response.status_code in [400, 422]