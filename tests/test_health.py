from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_returns_expected_payload() -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["message"] == "System is operational."
    assert payload["data"]["status"] == "healthy"
    assert "version" in payload["data"]
    assert "llm_provider" in payload["data"]
    assert "vector_store_ready" in payload["data"]
    assert "ingest_api_enabled" in payload["data"]
