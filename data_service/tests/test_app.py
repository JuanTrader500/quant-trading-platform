import pytest
from fastapi.testclient import TestClient
from data_service.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_latest_features_not_found():
    # Test with a completely invalid pair_code
    response = client.get("/features/latest?pair_code=NON_EXISTENT_PAIR")
    assert response.status_code == 404
    assert "pair_code desconocido" in response.json()["detail"]

def test_pipeline_run_failure():
    # Mock PipelineManager to simulate a failure
    with patch("data_service.app.main.PipelineManager") as mock_pm:
        instance = mock_pm.return_value
        instance.execute.return_value = False
        
        response = client.post("/pipeline/run")
        assert response.status_code == 500
        assert "El pipeline falló" in response.json()["detail"]

# Import patch here because it's used inside test_pipeline_run_failure
from unittest.mock import patch
