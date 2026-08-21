from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_is_available_without_model_loading() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_info_does_not_claim_a_trained_model() -> None:
    response = client.get("/api/v1/model/info")
    payload = response.json()

    assert response.status_code == 200
    assert payload["readiness"] == "architecture_only"
    assert payload["raw_input_storage_default"] is False
    assert "not a medical diagnosis" in payload["disclaimer"].lower()

