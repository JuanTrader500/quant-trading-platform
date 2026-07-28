"""
training/walk_forward.py
--------------------------
RF08: Walk-Forward Validation con ventana expansiva ("expanding
window"): en cada bloque, se entrena únicamente con datos estrictamente
anteriores al bloque de validación y se predice sobre ese bloque, nunca
al revés. Así se evita fuga de datos también durante la validación (no
solo en el feature engineering del Data Service, RF03).

Se usa tanto para:
  - el reentrenamiento mensual en producción (`retrain_manager.py`,
    cadencia real: el primero de cada mes), como
  - el backtest histórico 2022 -> actualidad (`historical_backtest.py`),
    que reproduce la MISMA cadencia mensual para estar alineado con lo
    que hubiera pasado en producción.

`target_range_next_day` ya viene calculado por el Data Service con
`shift(-1)` (RF03) — este módulo no vuelve a desplazar nada, solo
respeta el orden temporal al partir train/validación.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.logging_config import get_logger
from models.algorithms import build_candidate_models
from models.garch_baseline import GarchBaseline
from models.metrics import EvaluationMetrics, evaluate

logger = get_logger(__name__)

FEATURE_COLUMNS_FOR_GARCH_INPUT = "main_log_return"


@dataclass
class WalkForwardResult:
    metrics_by_model: dict[str, EvaluationMetrics]
    pooled_predictions: dict[str, list[dict]] = field(default_factory=dict)  # {model: [{date, y_true, y_pred}, ...]}
    best_model_name: str = ""


def _iter_expanding_folds(n_rows: int, min_train_days: int, step_days: int):
    """Genera (train_idx_end, val_idx_start, val_idx_end) para una
    ventana expansiva: el set de entrenamiento siempre arranca en 0 y
    crece; el bloque de validación es el siguiente tramo de `step_days`
    filas, nunca solapado con el entrenamiento."""
    train_end = min_train_days
    while train_end < n_rows:
        val_end = min(train_end + step_days, n_rows)
        yield train_end, train_end, val_end
        train_end = val_end


def run_walk_forward(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    min_train_days: int,
    step_days: int,
) -> WalkForwardResult:
    """Corre walk-forward validation sobre `df` (ordenado por fecha,
    índice = fecha) para el modelo de producción confirmado (ver
    `models/algorithms.py`) y el baseline GARCH (RF10). Devuelve métricas y predicciones pooled por
    modelo, más el nombre del mejor modelo ML (menor RMSE)."""
    df = df.sort_index()
    n_rows = len(df)
    if n_rows <= min_train_days:
        raise ValueError(
            f"No hay suficientes filas ({n_rows}) para el mínimo de entrenamiento "
            f"configurado ({min_train_days}). Reduce WF_MIN_TRAIN_DAYS o espera más datos."
        )

    X_all = df[feature_columns].to_numpy()
    y_all = df[target_column].to_numpy()
    dates_all = df.index.to_numpy()
    returns_all = df[FEATURE_COLUMNS_FOR_GARCH_INPUT].to_numpy()

    candidate_names = list(build_candidate_models().keys()) + ["garch_baseline"]
    pooled: dict[str, list[dict]] = {name: [] for name in candidate_names}

    n_folds = 0
    for train_end, val_start, val_end in _iter_expanding_folds(n_rows, min_train_days, step_days):
        n_folds += 1
        X_train, y_train = X_all[:train_end], y_all[:train_end]
        X_val, y_val = X_all[val_start:val_end], y_all[val_start:val_end]
        dates_val = dates_all[val_start:val_end]

        models = build_candidate_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            for d, yt, yp in zip(dates_val, y_val, preds):
                pooled[name].append({"date": pd.Timestamp(d).date().isoformat(), "y_true": float(yt), "y_pred": float(yp)})

        try:
            garch = GarchBaseline().fit(returns_all[:train_end])
            garch_pred = garch.forecast_next_day_range()
            for d, yt in zip(dates_val, y_val):
                pooled["garch_baseline"].append(
                    {"date": pd.Timestamp(d).date().isoformat(), "y_true": float(yt), "y_pred": garch_pred}
                )
        except Exception as exc:
            logger.error(f"GARCH baseline falló en un fold de walk-forward: {exc}")

    logger.info(f"Walk-Forward Validation completado: {n_folds} bloque(s), {n_rows} filas totales.")

    metrics_by_model: dict[str, EvaluationMetrics] = {}
    for name, records in pooled.items():
        if not records:
            continue
        y_true = np.array([r["y_true"] for r in records])
        y_pred = np.array([r["y_pred"] for r in records])
        metrics_by_model[name] = evaluate(y_true, y_pred)

    ml_only = {k: v for k, v in metrics_by_model.items() if k != "garch_baseline"}
    best_model_name = min(ml_only, key=lambda k: ml_only[k].rmse) if ml_only else ""

    return WalkForwardResult(
        metrics_by_model=metrics_by_model,
        pooled_predictions=pooled,
        best_model_name=best_model_name,
    )
