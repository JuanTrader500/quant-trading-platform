"""
app/main.py
-----------
API HTTP del Data Service (FastAPI). No contiene lógica de negocio:
delega todo en el paquete `pipeline`, para mantener la extracción y el
feature engineering desacoplados de la capa web (RNF10).

Endpoints:
  GET  /health              healthcheck del contenedor/orquestador.
  POST /pipeline/run        corre extracción + preparación (RF01-RF06).
  GET  /features/latest     última fila de features de un par (RF15).
  GET  /features/history    histórico con target conocido, para
                             entrenar modelos de ML (caso de uso 3).

Este servicio no debe exponerse directamente a internet: solo lo
consume el Web Service / ML Service dentro de la red interna.
"""

from datetime import date
from fastapi import FastAPI, HTTPException, Query
from pipeline import db
from pipeline.pipeline_manager import PipelineManager
from pipeline.registry import PAIRS

app = FastAPI(title="Data Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """Healthcheck simple, usado por Docker/Kubernetes para saber si
    el contenedor está listo para recibir tráfico."""
    return {"status": "ok"}


@app.post("/pipeline/run")
def run_pipeline() -> dict:
    """Corre extracción + preparación de forma síncrona.

    Pensado para ser invocado por un scheduler externo (cron, CronJob
    de Kubernetes) o manualmente durante desarrollo/pruebas.
    """
    ok = PipelineManager().execute()
    if not ok:
        raise HTTPException(
            status_code=500,
            detail="El pipeline falló. Ver logs/data_service/data_pipeline.log y la tabla ingestion_log.",
        )
    return {"status": "completed"}


@app.get("/features/latest")
def latest_features(pair_code: str = Query(..., description="Ej. SP500_VIX, NASDAQ_VXN")) -> dict:
    """Última fila de features calculada para un par. Usado por el
    modo "Predicción de Mañana" del ML Service (RF15)."""
    if pair_code not in PAIRS:
        raise HTTPException(status_code=404, detail=f"pair_code desconocido: {pair_code}")

    row = db.fetch_latest_features(pair_code)
    if row is None:
        raise HTTPException(status_code=404, detail="Sin features calculadas todavía para este par.")
    return row


@app.get("/features/history")
def features_history(
    pair_code: str = Query(..., description="Ej. SP500_VIX, NASDAQ_VXN"),
    date_from: date | None = Query(None, description="Filtro opcional, inclusive."),
    date_to: date | None = Query(None, description="Filtro opcional, inclusive."),
) -> list[dict]:
    """Histórico de features con target ya conocido, listo para
    entrenar (Walk-Forward Validation). Caso de uso 3 del Data Service."""
    if pair_code not in PAIRS:
        raise HTTPException(status_code=404, detail=f"pair_code desconocido: {pair_code}")

    return db.fetch_training_dataset(pair_code, date_from, date_to)
