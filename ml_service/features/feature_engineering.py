"""
features/feature_engineering.py
--------------------------------
RF14 (modo "Testing"): transforma un OHLC ingresado manualmente por la
persona usuaria en el vector de features que el modelo espera,
replicando el mismo Data Dictionary que el Data Service usa para su
dataset de entrenamiento (`feature_schema.FEATURE_COLUMNS`).

DECISIÓN DE DISEÑO / LIMITACIÓN CONOCIDA
-----------------------------------------
El contrato público del Data Service (`/features/latest`,
`/features/history`) expone únicamente features ya calculadas, nunca
el OHLC crudo histórico. Eso alcanza para features que dependen solo
del día manual (`main_log_range`, `main_body_log`,
`main_upper_wick_log`, `main_lower_wick_log`, `day_of_week`), que se
recalculan aquí de forma exacta a partir del OHLC ingresado.

Pero `main_log_return`, `main_vol_5d`, `main_vol_10d` y las columnas
`vol_idx_*` dependen del cierre de días anteriores (y del VIX, que la
persona usuaria no ingresa en modo Testing según RF14). Como el Data
Service no expone el cierre crudo del día anterior por API, estas
columnas se completan con el valor real más reciente conocido
(`/features/latest`) en lugar de recalcularse para el día hipotético
ingresado manualmente. Es una aproximación explícita y documentada,
no un cálculo con fuga de datos: no se usa ningún dato del futuro,
solo el contexto reciente ya público. Si se quiere una versión exacta,
el Data Service necesitaría un endpoint adicional que exponga el
cierre crudo del día anterior (ej. `GET /raw/latest`).
"""

import math
from datetime import date

from clients.data_service_client import get_latest_features
from core.logging_config import get_logger
from features.ohlc_validation import validate_ohlc

logger = get_logger(__name__)

# Debe coincidir 1:1 (sin contar target) con feature_schema.FEATURE_COLUMNS
# del Data Service, para que el modelo reciba las mismas columnas con las
# que fue entrenado (RNF12).
FEATURE_COLUMNS: list[str] = [
    "main_log_return",
    "main_log_range",
    "main_body_log",
    "main_upper_wick_log",
    "main_lower_wick_log",
    "main_vol_5d",
    "main_vol_10d",
    "vol_idx_log_close",
    "vol_idx_log_range",
    "vol_idx_log_return",
    "day_of_week",
]

# Features que sí se recalculan de forma exacta a partir del OHLC manual.
_SAME_DAY_COLUMNS = [
    "main_log_range",
    "main_body_log",
    "main_upper_wick_log",
    "main_lower_wick_log",
    "day_of_week",
]

# Features que se aproximan con el último valor real conocido (ver
# docstring del módulo).
_CARRIED_FORWARD_COLUMNS = [
    "main_log_return",
    "main_vol_5d",
    "main_vol_10d",
    "vol_idx_log_close",
    "vol_idx_log_range",
    "vol_idx_log_return",
]


def build_testing_features(
    open_: float,
    high: float,
    low: float,
    close: float,
    as_of: date,
    pair_code: str,
) -> dict:
    """Construye el vector de features para modo Testing (RF14),
    validando primero el OHLC (RF20)."""
    validate_ohlc(open_, high, low, close)

    log_open, log_high, log_low, log_close = math.log(open_), math.log(high), math.log(low), math.log(close)

    same_day = {
        "main_log_range": log_high - log_low,
        "main_body_log": log_close - log_open,
        "main_upper_wick_log": log_high - math.log(max(open_, close)),
        "main_lower_wick_log": math.log(min(open_, close)) - log_low,
        "day_of_week": as_of.weekday(),
    }

    try:
        latest = get_latest_features(pair_code)
    except Exception as exc:
        logger.error(f"No se pudo obtener contexto reciente del Data Service para modo Testing: {exc}")
        raise

    carried_forward = {col: latest.get(col) for col in _CARRIED_FORWARD_COLUMNS}
    missing = [c for c, v in carried_forward.items() if v is None]
    if missing:
        raise ValueError(
            f"El Data Service no tiene valores recientes para: {missing}. "
            "Corre /pipeline/run en el Data Service primero."
        )

    features = {**same_day, **carried_forward}
    return {col: features[col] for col in FEATURE_COLUMNS}
