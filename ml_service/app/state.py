"""
app/state.py
-------------
RF12: el modelo vigente se carga en memoria al iniciar el servicio y
se reemplaza ahí mismo tras cada reentrenamiento (RF44), sin
requerir reentrenar en cada solicitud de predicción.

Un solo objeto en memoria, protegido por un lock, para que un
reentrenamiento en curso no deje al proceso sirviendo un estado a
medio actualizar mientras llegan predicciones concurrentes.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ModelState:
    model: object | None = None
    model_name: str | None = None
    model_version: str | None = None
    mlflow_run_id: str | None = None
    loaded_at: str | None = None
    latest_metrics: dict = field(default_factory=dict)
    latest_comparison_vs_garch: dict = field(default_factory=dict)

    def is_ready(self) -> bool:
        return self.model is not None


_state = ModelState()
_lock = threading.Lock()


def get_state() -> ModelState:
    with _lock:
        return _state


def set_model(model, model_name: str, model_version: str | None, mlflow_run_id: str | None) -> None:
    with _lock:
        _state.model = model
        _state.model_name = model_name
        _state.model_version = model_version
        _state.mlflow_run_id = mlflow_run_id
        _state.loaded_at = datetime.now(timezone.utc).isoformat()


def set_latest_metrics(metrics_by_model: dict, best_model_name: str) -> None:
    with _lock:
        _state.latest_metrics = {name: m.to_dict() for name, m in metrics_by_model.items()}
        garch = metrics_by_model.get("garch_baseline")
        best = metrics_by_model.get(best_model_name)
        if garch and best:
            _state.latest_comparison_vs_garch = {
                "ml_model": best_model_name,
                "ml_rmse": best.rmse,
                "ml_mae": best.mae,
                "garch_rmse": garch.rmse,
                "garch_mae": garch.mae,
                "ml_beats_garch_rmse": best.rmse < garch.rmse,
                "ml_beats_garch_mae": best.mae < garch.mae,
            }
