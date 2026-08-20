"""
features/scale_conversion.py
-------------------------------
El modelo predice `target_range_next_day = log(High_{t+1}) - log(Low_{t+1})`
(ver `preparation.py` del Data Service: `(np.log(idx_high) - np.log(idx_low)).shift(-1)`).
Es una magnitud en escala logarítmica, sin unidades de precio.

Esta conversión, pedida explícitamente para vivir en el ML Service (no
en el Web Service/Django), la transforma en algo interpretable como
puntos del activo:

    1. `range_pct   = e^log_range - 1`         (ancho relativo del rango)
    2. `range_points = anchor_close * range_pct` (ancho en puntos)

`anchor_close` es el último cierre real conocido del índice principal
(hoy, si se está prediciendo mañana). Es una aproximación EXPLÍCITA:
usamos el cierre de hoy como referencia del nivel de precio de mañana,
porque el precio real de mañana obviamente todavía no existe. No es
fuga de datos — no se usa ningún valor del día que se predice.

RMSE/MAE/sesgo (`models/metrics.py`) y el walk-forward de
entrenamiento (`training/walk_forward.py`) siguen trabajando en escala
log contra `target_range_next_day` tal cual lo entrega el Data
Service, para que la comparación contra tu análisis en notebook (y
contra GARCH) se mantenga en las mismas unidades que ya validaste. Esta
conversión solo aplica en las respuestas de predicción servidas al
Web Service (`/predict/testing`, `/predict/tomorrow`).
"""

import math
from dataclasses import dataclass


@dataclass
class RangeInPoints:
    log_range: float
    range_pct: float
    anchor_close: float
    range_points: float

    def to_dict(self) -> dict:
        return {
            "predicted_log_range": self.log_range,
            "predicted_range_pct": self.range_pct,
            "anchor_close": self.anchor_close,
            "predicted_range_points": self.range_points,
        }


def convert_log_range_to_points(log_range: float, anchor_close: float) -> RangeInPoints:
    if anchor_close is None or anchor_close <= 0:
        raise ValueError(f"anchor_close debe ser un precio positivo, recibí: {anchor_close!r}")

    range_pct = math.expm1(log_range)  # e^log_range - 1, más estable numéricamente que exp(x)-1
    range_points = anchor_close * range_pct

    return RangeInPoints(
        log_range=log_range,
        range_pct=range_pct,
        anchor_close=anchor_close,
        range_points=range_points,
    )
