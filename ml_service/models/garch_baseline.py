"""
models/garch_baseline.py
--------------------------
RF10: modelo GARCH como baseline de referencia para el mismo objetivo
(predicción de rango/volatilidad diaria), usado en RF19 para comparar
el desempeño del modelo ML contra este baseline.

Se entrena un GARCH(1,1) sobre los retornos logarítmicos diarios
(`main_log_return`) y se usa la desviación estándar condicional
pronosticada a un paso como proxy del rango esperado del día
siguiente — comparable en magnitud con `target_range_next_day`
(también expresado en log-rango).
"""

import numpy as np
from arch import arch_model

from core.logging_config import get_logger

logger = get_logger(__name__)


class GarchBaseline:
    """Envoltorio delgado sobre `arch.arch_model` para exponer la misma
    interfaz `fit`/`predict` que los modelos de `models/algorithms.py`,
    de forma que `training/walk_forward.py` los trate de forma uniforme."""

    def __init__(self) -> None:
        self._result = None

    def fit(self, log_returns: np.ndarray) -> "GarchBaseline":
        # arch_model espera retornos en escala porcentual para converger
        # de forma estable.
        scaled_returns = np.asarray(log_returns, dtype=float) * 100
        model = arch_model(scaled_returns, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        self._result = model.fit(disp="off")
        return self

    def forecast_next_day_range(self) -> float:
        """Devuelve la desviación estándar condicional pronosticada a
        un paso, en la misma escala log que `target_range_next_day`."""
        if self._result is None:
            raise RuntimeError("GarchBaseline no ha sido entrenado (llamar fit() primero).")
        forecast = self._result.forecast(horizon=1, reindex=False)
        variance_pct2 = forecast.variance.values[-1, 0]
        sigma_pct = float(np.sqrt(variance_pct2))
        return sigma_pct / 100  # deshacer el escalado x100 de fit()
