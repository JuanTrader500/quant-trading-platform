"""
feature_schema.py
------------------
RNF12: versiona el esquema de features (nombres y orden de columnas).

Cada dataset procesado se guarda junto a un manifiesto JSON con su
versión de esquema y un hash de las columnas. El módulo de entrenamiento
debe comparar ese hash contra el usado por el modelo vigente y forzar
reentrenamiento si difieren, evitando predicciones con features
inconsistentes.
"""

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "1.0.0"

# Orden exacto de columnas esperado por dataset (definido en el
# Data Dictionary del proyecto). Agregar un asset nuevo = agregar una
# entrada aquí; nada más del pipeline necesita cambiar.
FEATURE_COLUMNS: dict[str, list[str]] = {
    "sp500": [
        "sp500_log_return", "sp500_log_range", "sp500_body_log",
        "sp500_upper_wick_log", "sp500_lower_wick_log",
        "sp500_vol_5d", "sp500_vol_10d",
        "vix_log_close", "vix_log_range", "vix_log_return",
        "day_of_week", "target",
    ],
    "nq": [
        "nq_log_return", "nq_log_range", "nq_body_log",
        "nq_upper_wick_log", "nq_lower_wick_log",
        "nq_vol_5d", "nq_vol_10d",
        "vxn_log_close", "vxn_log_range", "vxn_log_return",
        "day_of_week", "target",
    ],
}


def _hash_columns(columns: list[str]) -> str:
    return hashlib.sha256("|".join(columns).encode()).hexdigest()[:12]


def enforce_schema(df, dataset_name: str):
    """Reordena/valida columnas contra el esquema esperado. Falla rápido
    si falta una columna requerida (detecta cambios silenciosos upstream)."""
    expected = FEATURE_COLUMNS[dataset_name]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name}: faltan columnas del esquema esperado: {missing}")
    return df[expected]


def write_schema_manifest(dataset_name: str, output_dir: Path) -> Path:
    columns = FEATURE_COLUMNS[dataset_name]
    manifest = {
        "dataset": dataset_name,
        "schema_version": SCHEMA_VERSION,
        "columns": columns,
        "hash": _hash_columns(columns),
    }
    path = Path(output_dir) / f"{dataset_name}_schema.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path
