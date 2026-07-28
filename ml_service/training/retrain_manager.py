"""
training/retrain_manager.py
------------------------------
RF08: script de reentrenamiento automático usando Walk-Forward
Validation, con periodicidad configurable (mensual por defecto,
disparado por `scheduler/scheduler.py` el día 1 de cada mes).

Flujo de una corrida (`run_retraining`):
  1. Pide el histórico completo al Data Service por HTTP
     (`clients/data_service_client.get_training_dataset`) — RF39.
  2. Corre Walk-Forward Validation con el modelo confirmado (Gradient
     Boosting, hiperparámetros fijos en `models/algorithms.py`) y el
     baseline GARCH (RF10), alineado a la cadencia mensual de
     producción — sirve para medir RMSE/MAE/sesgo (RF17/RF18) y la
     comparación contra GARCH (RF19), no para elegir entre algoritmos.
  3. Entrena el mejor modelo (menor RMSE pooled) sobre TODO el
     histórico disponible (fit final, para servir predicciones).
  4. Compara sus métricas contra el modelo actualmente en producción
     (RNF04): si no mejora, se registra en MLflow pero NO se promueve
     — el servicio sigue sirviendo la versión anterior sin
     interrupción.
  5. Si mejora (o es la primera corrida), se promueve el alias
     `production` en MLflow (RF44) y se reemplaza el modelo en memoria
     del propio proceso (RF12/RF13), sin depender de Django ni de
     eventos externos.
"""

from dataclasses import dataclass

import pandas as pd

from app import state
from clients.data_service_client import get_training_dataset
from core.logging_config import get_logger
from core.settings import DATA_SERVICE_PAIR_CODE, WF_MIN_TRAIN_DAYS, WF_STEP_DAYS
from features.feature_engineering import FEATURE_COLUMNS
from models.algorithms import build_candidate_models
from registry import model_registry
from training.walk_forward import run_walk_forward

logger = get_logger(__name__)

TARGET_COLUMN = "target_range_next_day"


@dataclass
class RetrainOutcome:
    promoted: bool
    best_model_name: str
    mlflow_run_id: str
    training_rows: int


def _dataset_to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise ValueError("El Data Service no devolvió filas de entrenamiento.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(
            f"El Data Service devolvió columnas distintas a las esperadas. Faltan: {sorted(missing)}. "
            "Verifica que FEATURE_COLUMNS siga alineado con feature_schema.py del Data Service (RNF12)."
        )
    return df


def run_retraining(pair_code: str = DATA_SERVICE_PAIR_CODE) -> RetrainOutcome:
    logger.info(f"Iniciando reentrenamiento mensual para pair_code={pair_code} …")

    rows = get_training_dataset(pair_code)
    df = _dataset_to_frame(rows)
    logger.info(f"Dataset de entrenamiento recibido: {len(df)} filas ({df.index.min()} a {df.index.max()}).")

    wf_result = run_walk_forward(
        df=df,
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        min_train_days=WF_MIN_TRAIN_DAYS,
        step_days=WF_STEP_DAYS,
    )
    best_name = wf_result.best_model_name
    if not best_name:
        raise RuntimeError("Walk-Forward Validation no produjo ningún modelo candidato válido.")

    new_metrics = wf_result.metrics_by_model[best_name]
    logger.info(f"Mejor algoritmo por RMSE pooled de walk-forward: {best_name} (RMSE={new_metrics.rmse:.6f}).")

    # Fit final sobre TODO el histórico disponible, para servir predicciones.
    final_model = build_candidate_models()[best_name]
    X_full = df[FEATURE_COLUMNS].to_numpy()
    y_full = df[TARGET_COLUMN].to_numpy()
    final_model.fit(X_full, y_full)

    current_state = state.get_state()
    current_rmse = current_state.latest_metrics.get(current_state.model_name or "", {}).get("rmse")
    promote = current_rmse is None or new_metrics.rmse < current_rmse

    run_id = model_registry.log_and_register_run(
        model=final_model,
        algorithm_name=best_name,
        metrics_by_model=wf_result.metrics_by_model,
        best_model_name=best_name,
        training_rows=len(df),
        feature_columns=FEATURE_COLUMNS,
        promote_to_production=promote,
    )

    if promote:
        version = model_registry.get_production_version()
        state.set_model(final_model, best_name, version, run_id)

    state.set_latest_metrics(wf_result.metrics_by_model, best_name)

    logger.info(
        f"Reentrenamiento completado. run_id={run_id} promovido={promote} "
        f"(RMSE nuevo={new_metrics.rmse:.6f}, RMSE anterior={current_rmse})."
    )
    return RetrainOutcome(promoted=promote, best_model_name=best_name, mlflow_run_id=run_id, training_rows=len(df))
