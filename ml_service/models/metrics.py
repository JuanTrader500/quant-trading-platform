"""
models/metrics.py
------------------
RF17: RMSE y MAE del modelo vigente sobre el conjunto de validación
más reciente.

RF18: métrica propia de sesgo direccional, que distingue entre
subestimación (predicción < real) y sobreestimación (predicción >
real) de la volatilidad, en lugar de solo un error promedio que las
compensa entre sí.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationMetrics:
    rmse: float
    mae: float
    directional_bias: float  # promedio de (predicho - real); >0 sobreestima, <0 subestima
    underestimation_rate: float  # fracción de días donde predicho < real
    overestimation_rate: float  # fracción de días donde predicho > real
    n_observations: int

    def to_dict(self) -> dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "directional_bias": self.directional_bias,
            "underestimation_rate": self.underestimation_rate,
            "overestimation_rate": self.overestimation_rate,
            "n_observations": self.n_observations,
        }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise ValueError("No hay observaciones para evaluar.")

    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    directional_bias = float(np.mean(error))
    underestimation_rate = float(np.mean(error < 0))
    overestimation_rate = float(np.mean(error > 0))

    return EvaluationMetrics(
        rmse=rmse,
        mae=mae,
        directional_bias=directional_bias,
        underestimation_rate=underestimation_rate,
        overestimation_rate=overestimation_rate,
        n_observations=len(y_true),
    )
