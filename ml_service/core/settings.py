"""
core/settings.py
-----------------
Fuente única de verdad para configuración del ML Service. Todo valor
sensible o dependiente del entorno se lee de variables de entorno
(RNF06) — nunca se hardcodea. Copia `.env.example` a `.env`.

El ML Service es stateless respecto a datos de negocio (RNF17): no
tiene base de datos de series temporales propia. Sí usa un SQLite
local (en un volumen Docker) exclusivamente para el historial de
predicciones día a día que la persona pidió poder exportar/consultar
después (no es la fuente de verdad de features ni de entrenamiento,
esa vive en el Data Service).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
SERVICE_DIR = PACKAGE_DIR.parent

# --- Comunicación con Data Service (RF39, RF41, RF42) -----------------
DATA_SERVICE_BASE_URL = os.getenv("DATA_SERVICE_BASE_URL", "http://data_service:8000")
DATA_SERVICE_PAIR_CODE = os.getenv("DATA_SERVICE_PAIR_CODE", "SP500_VIX")
INTERNAL_SERVICE_KEY = os.getenv("INTERNAL_SERVICE_KEY", "")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "5"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
HTTP_RETRY_BACKOFF_SECONDS = float(os.getenv("HTTP_RETRY_BACKOFF_SECONDS", "1"))

# --- API propia del ML Service (RF41: clave de servicio interna) ------
# Las llamadas administrativas (reload, retrain manual) y las llamadas
# entrantes desde el Web Service deben incluir este header.
ML_SERVICE_API_KEY = os.getenv("ML_SERVICE_API_KEY", "")
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"

# --- MLflow: Model Registry + tracking de reentrenamientos (RF43) -----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sp500_volatility")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "sp500_range_predictor")
MLFLOW_MONITORING_RUN_NAME = os.getenv("MLFLOW_MONITORING_RUN_NAME", "production_monitoring")

# --- SQLite de predicciones (volumen Docker) ---------------------------
PREDICTIONS_DB_PATH = Path(os.getenv("PREDICTIONS_DB_PATH", "/app/data/predictions.db"))
PREDICTIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Export histórico (backtest walk-forward 2022 -> actualidad) ------
HISTORICAL_BACKTEST_START = os.getenv("HISTORICAL_BACKTEST_START", "2022-01-01")
HISTORICAL_BACKTEST_OUTPUT_DIR = Path(os.getenv("HISTORICAL_BACKTEST_OUTPUT_DIR", "/app/data/exports"))
HISTORICAL_BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Modelo local (respaldo .pkl, RNF04) --------------------------------
MODEL_LOCAL_DIR = Path(os.getenv("MODEL_LOCAL_DIR", "/app/data/models"))
MODEL_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# --- Scheduler interno (RF08: reentrenamiento mensual) ------------------
RETRAIN_SCHEDULE_DAY = int(os.getenv("RETRAIN_SCHEDULE_DAY", "1"))  # día 1 de cada mes
RETRAIN_SCHEDULE_HOUR = int(os.getenv("RETRAIN_SCHEDULE_HOUR", "3"))  # 03:00
RETRAIN_SCHEDULE_MINUTE = int(os.getenv("RETRAIN_SCHEDULE_MINUTE", "0"))
RETRAIN_SCHEDULE_TIMEZONE = os.getenv("RETRAIN_SCHEDULE_TIMEZONE", "America/Bogota")
SCHEDULER_ENABLED = os.getenv("SCHEDULER_ENABLED", "true").lower() == "true"

# --- Logging estructurado + trace ID (RNF19) ----------------------------
LOG_DIR = Path(os.getenv("ML_SERVICE_LOG_DIR", SERVICE_DIR / "logs"))
LOG_FILE = LOG_DIR / "ml_service.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Walk-Forward Validation ---------------------------------------------
WF_MIN_TRAIN_DAYS = int(os.getenv("WF_MIN_TRAIN_DAYS", "252"))  # ~1 año hábil
WF_STEP_DAYS = int(os.getenv("WF_STEP_DAYS", "21"))  # ~1 mes hábil por bloque de validación
