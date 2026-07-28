from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_dummy_model():
    class DummyModel:
        def predict(self, X):
            return [0.012 for _ in X]

    return DummyModel()


def test_health_reports_no_model_when_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    with patch("registry.model_registry.load_production_model", return_value=(None, None)), \
         patch("registry.model_registry.load_latest_local_backup", return_value=(None, None)), \
         patch("clients.data_service_client.health", return_value=False):
        from app.main import app
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["model_loaded"] is False


def test_predict_tomorrow_returns_prediction_when_model_and_data_service_are_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    dummy_model = _make_dummy_model()
    latest_features = {
        "date": "2026-07-24",
        "main_log_return": 0.001, "main_log_range": 0.01, "main_body_log": 0.002,
        "main_upper_wick_log": 0.001, "main_lower_wick_log": 0.001,
        "main_vol_5d": 0.01, "main_vol_10d": 0.012,
        "vol_idx_log_close": 3.0, "vol_idx_log_range": 0.05, "vol_idx_log_return": 0.01,
        "day_of_week": 4,
    }

    with patch("registry.model_registry.load_production_model", return_value=(dummy_model, "1")), \
         patch("clients.data_service_client.get_latest_features", return_value=latest_features):
        from app.main import app
        with TestClient(app) as client:
            response = client.post("/predict/tomorrow", params={"pair_code": "SP500_VIX"})
            assert response.status_code == 200
            body = response.json()
            assert body["predicted_range"] == 0.012
            assert body["target_date"] == "2026-07-25"
            assert body["mode"] == "automatic"


def test_predict_testing_rejects_invalid_ohlc(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(tmp_path / "predictions.db"))
    monkeypatch.setenv("SCHEDULER_ENABLED", "false")

    dummy_model = _make_dummy_model()
    with patch("registry.model_registry.load_production_model", return_value=(dummy_model, "1")):
        from app.main import app
        with TestClient(app) as client:
            response = client.post(
                "/predict/testing",
                json={"open": 100, "high": 90, "low": 80, "close": 85},
            )
            assert response.status_code == 422
