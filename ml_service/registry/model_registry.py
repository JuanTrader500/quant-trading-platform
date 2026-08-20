"""
registry/model_registry.py
----------------------------
RF43: Model Registry sobre MLflow, donde el ML Service publica cada
modelo versionado con sus métricas tras reentrenar.

RF11: versiona cada modelo entrenado (timestamp, hiperparámetros,
métricas de validación) para trazabilidad y rollback.

RF12/RF44: el modelo vigente se identifica con el alias
`production` en el Model Registry de MLflow; el ML Service lo carga
en memoria al iniciar y lo reemplaza ahí mismo tras cada
reentrenamiento exitoso, sin depender de eventos externos.

RNF04: si un reentrenamiento no supera al modelo vigente, no se mueve
el alias `production` y el servicio sigue sirviendo la versión
anterior (ver `training/retrain_manager.py`).
"""

import pickle
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from core.logging_config import get_logger
from core.settings import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_LOCAL_DIR,
)

logger = get_logger(__name__)

PRODUCTION_ALIAS = "production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def log_and_register_run(
    model,
    algorithm_name: str,
    metrics_by_model: dict,
    best_model_name: str,
    training_rows: int,
    feature_columns: list[str],
    promote_to_production: bool,
) -> str:
    """Registra un run de reentrenamiento en MLflow: hiperparámetros,
    métricas de validación de TODOS los candidatos (para trazabilidad
    de la comparación RF19), y el modelo ganador como artifact
    versionado en el Model Registry.

    Devuelve la `run_id` de MLflow. Si `promote_to_production` es
    True, además mueve el alias `production` a esta nueva versión
    (RF44); si es False, la versión queda registrada mas no vigente
    (RNF04: sigue sirviendo la anterior).
    """
    with mlflow.start_run(run_name=f"retrain_{algorithm_name}") as run:
        mlflow.log_param("selected_algorithm", algorithm_name)
        mlflow.log_param("training_rows", training_rows)
        mlflow.log_param("feature_columns", ",".join(feature_columns))

        for model_name, metrics in metrics_by_model.items():
            prefix = model_name
            for metric_name, value in metrics.to_dict().items():
                mlflow.log_metric(f"{prefix}_{metric_name}", value)

        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=MLFLOW_REGISTERED_MODEL_NAME)
        run_id = run.info.run_id

    # También se guarda un respaldo .pkl local (RNF04, RF07), por si
    # MLflow no está disponible al arrancar el proceso.
    _save_local_backup(model, run_id)

    if promote_to_production:
        _promote_latest_version_to_production(run_id)
        logger.info(f"Modelo del run {run_id} promovido a alias '{PRODUCTION_ALIAS}'.")
    else:
        logger.info(
            f"Modelo del run {run_id} registrado pero NO promovido "
            f"(RNF04: no superó al modelo vigente)."
        )

    return run_id


def _save_local_backup(model, run_id: str) -> Path:
    backup_path = MODEL_LOCAL_DIR / f"model_{run_id}.pkl"
    with open(backup_path, "wb") as f:
        pickle.dump(model, f)
    return backup_path


def _promote_latest_version_to_production(run_id: str) -> None:
    versions = _client.search_model_versions(f"name='{MLFLOW_REGISTERED_MODEL_NAME}' and run_id='{run_id}'")
    if not versions:
        raise RuntimeError(f"No se encontró la versión de modelo recién registrada para run_id={run_id}")
    version = versions[0].version
    _client.set_registered_model_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)


def get_production_version() -> str | None:
    """Devuelve el número de versión de MLflow que tiene actualmente el
    alias `production`, o None si todavía no existe ninguna."""
    try:
        version = _client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        return version.version
    except MlflowException:
        return None


def load_production_model():
    """RF12: descarga el modelo vigente (alias `production`) del Model
    Registry al arrancar el proceso FastAPI. Si MLflow no tiene
    ninguna versión con ese alias todavía (primera vez), devuelve
    None — el servicio arranca sin modelo hasta el primer
    reentrenamiento."""
    try:
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)
        version = _client.get_model_version_by_alias(MLFLOW_REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
        logger.info(f"Modelo vigente cargado desde MLflow: versión {version.version}.")
        return model, version.version
    except MlflowException as exc:
        logger.error(f"No hay modelo vigente en el Model Registry todavía: {exc}")
        return None, None
    except Exception as exc:
        logger.error(f"Error inesperado cargando el modelo vigente desde MLflow: {exc}")
        return None, None


def load_latest_local_backup():
    """RNF04: respaldo .pkl local, usado si MLflow no responde al
    arrancar el proceso (ej. contenedor de MLflow todavía no está
    listo)."""
    backups = sorted(MODEL_LOCAL_DIR.glob("model_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        return None, None
    with open(backups[0], "rb") as f:
        model = pickle.load(f)
    return model, backups[0].stem.replace("model_", "")
