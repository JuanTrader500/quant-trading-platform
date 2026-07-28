"""
training/historical_backtest.py
----------------------------------
Genera el CSV histórico `date,predict` pedido explícitamente para uso
externo (no es parte del flujo de predicción en producción): simula
qué habría predicho el sistema cada día hábil desde
`HISTORICAL_BACKTEST_START` (2022-01-01 por defecto) hasta hoy, **si
hubiera estado corriendo con la misma cadencia de reentrenamiento
mensual real** (RF08: reentrena el primero de cada mes).

Diseño (para estar alineado con el entorno de producción)
------------------------------------------------------------
Para cada mes calendario en el rango:
  1. Se entrena únicamente con datos con fecha ANTERIOR al primer día
     de ese mes (ventana expansiva, igual que en producción) —
     jamás se usa un dato del mes que se va a predecir ni de meses
     futuros. Sin fuga de datos.
  2. Se predice, con ese modelo "congelado" del mes, cada fila del
     mes cuyo `target_range_next_day` ya sea conocido en el Data
     Service (para poder comparar contra el valor real más adelante).

Ya no aplica la simplificación de "elegir un algoritmo una sola vez":
hay un único modelo confirmado (Gradient Boosting con hiperparámetros
fijos, ver `models/algorithms.py`), así que cada mes simplemente se
reajusta (fit) ese mismo modelo con los datos disponibles hasta ese
punto — sin comparar entre algoritmos.

Uso
---
    python -m training.historical_backtest
    python -m training.historical_backtest --start 2022-01-01 --end 2026-07-26
"""

import argparse
import csv
from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from clients.data_service_client import get_training_dataset
from core.logging_config import get_logger
from core.settings import (
    DATA_SERVICE_PAIR_CODE,
    HISTORICAL_BACKTEST_OUTPUT_DIR,
    HISTORICAL_BACKTEST_START,
    WF_MIN_TRAIN_DAYS,
)
from features.feature_engineering import FEATURE_COLUMNS
from models.algorithms import build_candidate_models

logger = get_logger(__name__)

TARGET_COLUMN = "target_range_next_day"


def _month_starts(start: date, end: date):
    current = date(start.year, start.month, 1)
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


def _the_only_algorithm_name() -> str:
    """Ya no hay selección: un único modelo confirmado (ver
    `models/algorithms.py`). Se deja esta función (en vez de un
    literal esparcido por el código) para que, si en el futuro se
    vuelve a comparar entre varios algoritmos, solo haya que tocar
    este punto."""
    return next(iter(build_candidate_models()))


def run_historical_backtest(pair_code: str, start: date, end: date) -> list[tuple[str, float]]:
    logger.info(f"Descargando histórico completo de {pair_code} desde el Data Service …")
    rows = get_training_dataset(pair_code)
    if not rows:
        raise RuntimeError("El Data Service no devolvió filas para el backtest histórico.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    algorithm_name = _the_only_algorithm_name()
    predictions: list[tuple[str, float]] = []

    for month_start in _month_starts(start, end):
        train_df = df[df.index.date < month_start]
        if len(train_df) < WF_MIN_TRAIN_DAYS:
            logger.info(f"Saltando {month_start.isoformat()}: no hay suficiente histórico previo todavía.")
            continue

        model = build_candidate_models()[algorithm_name]
        model.fit(train_df[FEATURE_COLUMNS].to_numpy(), train_df[TARGET_COLUMN].to_numpy())

        month_end = month_start + relativedelta(months=1)
        month_df = df[(df.index.date >= month_start) & (df.index.date < month_end)]
        if month_df.empty:
            continue

        preds = model.predict(month_df[FEATURE_COLUMNS].to_numpy())
        for target_date, pred in zip(month_df.index, preds):
            predictions.append((target_date.date().isoformat(), float(pred)))

        logger.info(f"Mes {month_start.isoformat()}: {len(month_df)} predicción(es) generada(s).")

    return predictions


def _write_csv(predictions: list[tuple[str, float]], output_path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "predict"])
        writer.writerows(predictions)
    logger.info(f"CSV histórico escrito en {output_path} ({len(predictions)} filas).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest histórico walk-forward date,predict")
    parser.add_argument("--pair-code", default=DATA_SERVICE_PAIR_CODE)
    parser.add_argument("--start", default=HISTORICAL_BACKTEST_START)
    parser.add_argument("--end", default=datetime.utcnow().date().isoformat())
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_path = args.output or (HISTORICAL_BACKTEST_OUTPUT_DIR / f"predictions_{args.pair_code}_{start}_{end}.csv")

    predictions = run_historical_backtest(args.pair_code, start, end)
    _write_csv(predictions, output_path)


if __name__ == "__main__":
    main()
