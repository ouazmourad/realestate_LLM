import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# ensure root directory on path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from main import app

client = TestClient(app)


def setup_function() -> None:
    shutil.rmtree(Path("data"), ignore_errors=True)


def test_ingest_and_ask() -> None:
    payload = {
        "client_id": "tenant1",
        "docs": [
            {"id": "policy", "text": "Pets are not allowed in the building."},
            {"id": "fees", "text": "The HOA fee is $100."},
        ],
    }
    r = client.post("/v1/ingest-json", json=payload)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.post("/v1/ask", json={"client_id": "tenant1", "question": "Are pets allowed?"})
    data = r.json()
    assert "answer" in data
    assert "Pets are not allowed" in data["answer"]
    assert data["citations"]
    assert any("policy" in c["id"] for c in data["citations"])
