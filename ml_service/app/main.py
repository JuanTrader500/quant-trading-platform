"""
app/main.py
-------------
API HTTP del ML Service (FastAPI). Stateless respecto a datos de
negocio (RNF17): no tiene base de datos de series temporales propia,
solo un SQLite local para el historial de predicciones día a día.

Endpoints:
  GET  /health                     healthcheck (propio + Data Service).
  POST /predict/testing            modo Testing, OHLC manual (RF14, RF20).
  POST /predict/tomorrow           modo automático (RF15).
  GET  /model/metrics               RMSE/MAE/sesgo + comparación GARCH (RF17-RF19).
  GET  /predictions/history         historial para graficar/comparar (RF24).
  GET  /predictions/export/historical  descarga del CSV date,predict 2022->hoy.
  POST /admin/reload-model          recarga manual del modelo vigente (RF44).
  POST /admin/retrain               dispara un reentrenamiento fuera de cadencia.

RNF20: este servicio no debe exponerse directamente a internet, solo
alcanzable dentro de la red interna de Docker; solo lo consume el Web
Service.
"""

from contextlib import asynccontextmanager
from datetime import date, timedelta

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app import state
from app.schemas import (
    MetricsResponse,
    PredictionResponse,
    ReloadResponse,
    TestingPredictionRequest,
)
from app.security import verify_service_key
from clients import data_service_client
from clients.data_service_client import DataServiceNotFoundError, DataServiceUnavailableError
from core.logging_config import get_logger, set_trace_id
from core.settings import DATA_SERVICE_PAIR_CODE, HISTORICAL_BACKTEST_OUTPUT_DIR, SCHEDULER_ENABLED
from features.feature_engineering import FEATURE_COLUMNS, build_testing_features
from features.ohlc_validation import InvalidOHLCError
from features.scale_conversion import convert_log_range_to_points
from predictions import db as predictions_db
from predictions.mlflow_logger import log_daily_prediction
from registry import model_registry
from scheduler.scheduler import shutdown_scheduler, start_scheduler
from training.retrain_manager import run_retraining

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictions_db.init_db()

    model, version = model_registry.load_production_model()
    if model is None:
        model, version = model_registry.load_latest_local_backup()  # RNF04
        if model is not None:
            logger.info(f"Modelo cargado desde respaldo local .pkl (versión MLflow no disponible): {version}")
    if model is not None:
        state.set_model(model, model_name="unknown_at_startup", model_version=version, mlflow_run_id=None)
    else:
        logger.error("No hay ningún modelo disponible todavía (ni en MLflow ni en respaldo local).")

    if SCHEDULER_ENABLED:
        start_scheduler()

    yield

    if SCHEDULER_ENABLED:
        shutdown_scheduler()


app = FastAPI(title="ML Service", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """RNF19: propaga (o genera) el trace_id de correlación para toda
    la petición, incluyendo las llamadas salientes al Data Service."""
    incoming_trace_id = request.headers.get("X-Trace-Id")
    trace_id = set_trace_id(incoming_trace_id)
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state.get_state().is_ready() else "degraded_no_model",
        "data_service_reachable": data_service_client.health(),
        "model_loaded": state.get_state().is_ready(),
    }


@app.post("/predict/testing", response_model=PredictionResponse)
def predict_testing(payload: TestingPredictionRequest) -> PredictionResponse:
    """RF14/RF20/RF21: modo Testing con OHLC manual."""
    current = state.get_state()
    if not current.is_ready():
        raise HTTPException(status_code=503, detail="El modelo todavía no está disponible. Intenta más tarde.")

    pair_code = payload.pair_code or DATA_SERVICE_PAIR_CODE
    try:
        features = build_testing_features(payload.open, payload.high, payload.low, payload.close, payload.as_of, pair_code)
    except InvalidOHLCError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (DataServiceUnavailableError, DataServiceNotFoundError) as exc:
        raise HTTPException(status_code=502, detail="No se pudo obtener contexto del Data Service.") from exc

    x = [[features[col] for col in FEATURE_COLUMNS]]
    predicted_log_range = float(current.model.predict(x)[0])
    target_date = payload.as_of + timedelta(days=1)

    # Ancla = el Close ingresado manualmente para `as_of` (RF14 ya lo
    # trae, no requiere ninguna llamada extra al Data Service).
    converted = convert_log_range_to_points(predicted_log_range, anchor_close=payload.close)

    predictions_db.insert_prediction(
        target_date=target_date.isoformat(), pair_code=pair_code, mode="testing",
        predicted_log_range=converted.log_range, predicted_range_pct=converted.range_pct,
        anchor_close=converted.anchor_close, predicted_range_points=converted.range_points,
        model_name=current.model_name, model_version=current.model_version,
        mlflow_run_id=current.mlflow_run_id, trace_id=None,
    )

    return PredictionResponse(
        **converted.to_dict(), target_date=target_date,
        model_used=current.model_name, model_version=current.model_version, mode="testing",
    )


@app.post("/predict/tomorrow", response_model=PredictionResponse)
def predict_tomorrow(pair_code: str = Query(default=DATA_SERVICE_PAIR_CODE)) -> PredictionResponse:
    """RF15: modo automático — consulta la última fila de features
    disponible en el Data Service y predice el rango del siguiente día
    hábil."""
    current = state.get_state()
    if not current.is_ready():
        raise HTTPException(status_code=503, detail="El modelo todavía no está disponible. Intenta más tarde.")

    try:
        latest = data_service_client.get_latest_features(pair_code)
    except DataServiceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DataServiceUnavailableError as exc:
        raise HTTPException(status_code=502, detail="El Data Service no está disponible en este momento.") from exc

    missing = [c for c in FEATURE_COLUMNS if latest.get(c) is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan features en el Data Service: {missing}")

    try:
        raw = data_service_client.get_latest_raw_close(pair_code)
    except DataServiceNotFoundError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "El Data Service no expone /raw/latest todavía (necesario para anclar la "
                "conversión a puntos). Agrega ese endpoint en data_service — ver "
                "data_service_CAMBIOS_raw_latest.md."
            ),
        ) from exc
    except DataServiceUnavailableError as exc:
        raise HTTPException(status_code=502, detail="El Data Service no está disponible en este momento.") from exc

    x = [[latest[col] for col in FEATURE_COLUMNS]]
    predicted_log_range = float(current.model.predict(x)[0])
    last_known_date = date.fromisoformat(str(latest["date"])[:10])
    target_date = last_known_date + timedelta(days=1)

    converted = convert_log_range_to_points(predicted_log_range, anchor_close=raw["close"])

    predictions_db.insert_prediction(
        target_date=target_date.isoformat(), pair_code=pair_code, mode="automatic",
        predicted_log_range=converted.log_range, predicted_range_pct=converted.range_pct,
        anchor_close=converted.anchor_close, predicted_range_points=converted.range_points,
        model_name=current.model_name, model_version=current.model_version,
        mlflow_run_id=current.mlflow_run_id, trace_id=None,
    )
    log_daily_prediction(target_date.isoformat(), converted.range_points)

    return PredictionResponse(
        **converted.to_dict(), target_date=target_date,
        model_used=current.model_name, model_version=current.model_version, mode="automatic",
    )


@app.get("/model/metrics", response_model=MetricsResponse)
def model_metrics() -> MetricsResponse:
    """RF17/RF18/RF19: RMSE, MAE, sesgo direccional del modelo vigente,
    y su comparación contra el baseline GARCH."""
    current = state.get_state()
    if not current.latest_metrics:
        raise HTTPException(status_code=404, detail="Todavía no hay métricas: no ha corrido ningún reentrenamiento.")
    return MetricsResponse(
        model_name=current.model_name, model_version=current.model_version,
        metrics_by_model=current.latest_metrics, comparison_vs_garch=current.latest_comparison_vs_garch,
    )


@app.get("/predictions/history")
def predictions_history(
    pair_code: str | None = Query(default=None),
    mode: str | None = Query(default=None, description="testing | automatic | backtest"),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=500, le=5000),
) -> list[dict]:
    """RF24 + uso externo: historial de predicciones consultable para
    graficar o comparar contra el valor real observado."""
    return predictions_db.fetch_history(pair_code, mode, date_from, date_to, limit)


@app.get("/predictions/export/historical")
def export_historical_predictions(filename: str = Query(..., description="Nombre exacto generado por historical_backtest.py")):
    """Descarga del CSV `date,predict` 2022->actualidad generado por
    `training/historical_backtest.py` (corrida batch, no en vivo)."""
    path = HISTORICAL_BACKTEST_OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado. Corre historical_backtest.py primero.")
    return FileResponse(path, media_type="text/csv", filename=filename)


@app.post("/admin/reload-model", response_model=ReloadResponse, dependencies=[Depends(verify_service_key)])
def reload_model() -> ReloadResponse:
    """RF44: fuerza una recarga manual del modelo vigente desde el
    Model Registry, protegido por la clave de servicio interna."""
    model, version = model_registry.load_production_model()
    if model is None:
        return ReloadResponse(reloaded=False, model_name=None, model_version=None)
    state.set_model(model, model_name="reloaded_from_registry", model_version=version, mlflow_run_id=None)
    return ReloadResponse(reloaded=True, model_name="reloaded_from_registry", model_version=version)


@app.post("/admin/retrain", dependencies=[Depends(verify_service_key)])
def trigger_retrain() -> dict:
    """Dispara un reentrenamiento fuera de la cadencia mensual del
    scheduler (útil para pruebas manuales durante tu revisión)."""
    try:
        outcome = run_retraining()
    except Exception as exc:
        logger.error(f"Reentrenamiento manual falló: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="El reentrenamiento falló. Revisa los logs del contenedor.") from exc
    return {
        "promoted": outcome.promoted,
        "best_model_name": outcome.best_model_name,
        "mlflow_run_id": outcome.mlflow_run_id,
        "training_rows": outcome.training_rows,
    }