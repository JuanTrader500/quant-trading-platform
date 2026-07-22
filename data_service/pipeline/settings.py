"""
settings.py
-----------
Fuente única de verdad para rutas y constantes del DataPipeline.

Desacopla la ubicación física de los datos y artefactos del código
(RNF10), de modo que extraction / validation / preparation nunca
hardcodean rutas: todas se resuelven aquí.

Layout esperado
----------------
sp500_MLops/
├── src/
│   ├── DataPipeline/      <- este paquete
│   └── data/
│       ├── raw/
│       └── processed/
└── models/
    └── artifacts/
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # .../data_service/pipeline
SRC_DIR = PACKAGE_DIR.parent                            # .../data_service
PROJECT_ROOT = SRC_DIR.parent                            # .../sp500_MLops

DATA_DIR = SRC_DIR / "pipeline" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "ml_service" / "artifacts"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

LOG_DIR = PROJECT_ROOT / "logs" / "data_service"
LOG_FILE = LOG_DIR / "data_pipeline.log"

ASSETS_CONFIG_PATH = PACKAGE_DIR / "config" / "assets.yaml"

RETRAINING_INTERVAL_WEEKS = 1
DEFAULT_START_DATE = "2005-01-01"

for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
