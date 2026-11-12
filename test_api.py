from fastapi.testclient import TestClient
from api import app

def test_health():
    client = TestClient(app)
    assert client.get("/health").status_code == 200

def test_invalid_request():
    client = TestClient(app)
    assert client.post("/predict",json={}).status_code == 422
