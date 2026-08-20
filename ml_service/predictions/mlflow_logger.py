"""
predictions/mlflow_logger.py
------------------------------
Además de guardarse en el SQLite propio (`predictions/db.py`, fuente
de verdad y API de consulta), cada predicción día a día se registra
también como métrica dentro de un run de MLflow de larga duración
("production_monitoring"), para que se pueda ver y comparar en la UI
de MLflow junto con las métricas de los reentrenamientos, sin
necesidad de correlacionar dos sistemas distintos.

MLflow indexa las métricas por `step`: se usa el número de días
transcurridos desde `HISTORICAL_BACKTEST_START` como step, para que
queden ordenadas cronológicamente en los gráficos de MLflow.
"""

from datetime import date

import mlflow

from core.logging_config import get_logger
from core.settings import HISTORICAL_BACKTEST_START, MLFLOW_EXPERIMENT_NAME, MLFLOW_MONITORING_RUN_NAME, MLFLOW_TRACKING_URI

logger = get_logger(__name__)

_EPOCH = date.fromisoformat(HISTORICAL_BACKTEST_START)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _get_or_create_experiment_id() -> str:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)


def _find_or_create_monitoring_run() -> str:
    """Reutiliza siempre el mismo run "production_monitoring" de larga
    duración (uno solo, no uno por día), para que todas las métricas
    diarias queden como una sola serie de tiempo navegable en MLflow."""
    client = mlflow.MlflowClient()
    experiment_id = _get_or_create_experiment_id()

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{MLFLOW_MONITORING_RUN_NAME}'",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id

    with mlflow.start_run(run_name=MLFLOW_MONITORING_RUN_NAME, experiment_id=experiment_id) as run:
        return run.info.run_id


def log_daily_prediction(target_date: str, predicted_range: float, actual_range: float | None = None) -> None:
    """Registra la predicción del día como métrica de MLflow. Si más
    adelante se conoce `actual_range`, se puede volver a llamar para
    loguearlo también (permite graficar predicho vs. real en MLflow)."""
    try:
        run_id = _find_or_create_monitoring_run()
        step = (date.fromisoformat(target_date) - _EPOCH).days
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("daily_predicted_range", predicted_range, step=step)
            if actual_range is not None:
                mlflow.log_metric("daily_actual_range", actual_range, step=step)
    except Exception as exc:
        # No debe tumbar la predicción si MLflow está caído: el SQLite
        # ya es la fuente de verdad para esto.
        logger.error(f"No se pudo loguear la predicción diaria en MLflow (no crítico): {exc}")
