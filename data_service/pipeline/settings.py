"""
settings.py
-----------
Fuente única de verdad para configuración del Data Service: rutas de
logs, conexión a base de datos y parámetros del pipeline.

Todo valor sensible o dependiente del entorno (credenciales de base de
datos, versión del pipeline desplegado) se lee de variables de entorno
vía `.env` — nunca se hardcodea en el código (RNF06). Copia
`.env.example` a `.env` en la raíz de `data_service/` y completa los
valores reales; `.env` nunca debe subirse al repositorio.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent  # .../data_service/pipeline
SERVICE_DIR = PACKAGE_DIR.parent  # .../data_service
PROJECT_ROOT = SERVICE_DIR.parent  # raíz del monorepo (si aplica)

# Conexión a PostgreSQL/TimescaleDB. Se valida (con error explícito) la
# primera vez que se usa, en db.get_engine() — no aquí, para que
# importar settings.py nunca falle por falta de configuración.
DATABASE_URL: str | None = os.getenv("DATABASE_URL")

# Directorio y archivo de logs (RF06).
LOG_DIR = Path(os.getenv("DATA_SERVICE_LOG_DIR", PROJECT_ROOT / "logs" / "data_service"))
LOG_FILE = LOG_DIR / "data_pipeline.log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Fecha desde la que se descarga histórico si un ticker no tiene datos
# todavía en raw_ohlc (primera corrida).
DEFAULT_START_DATE = os.getenv("DEFAULT_START_DATE", "2005-01-01")

# Identificador de versión de este pipeline (commit/tag de git,
# inyectado por CI/CD), usado en ingestion_log.pipeline_version para
# trazabilidad de qué código produjo cada corrida (caso de uso 5).
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "unknown")
