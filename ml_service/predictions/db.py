"""
predictions/db.py
-------------------
Capa de acceso al SQLite de predicciones día a día (ver
`db/predictions_schema.sql`). Aísla al resto del servicio de SQL
directo, igual que hace el Data Service con su propio `db.py`
(consistencia de estilo entre servicios, RNF10/RNF11).

El archivo físico vive en un volumen Docker nombrado (ver
`PREDICTIONS_DB_PATH` en `core/settings.py`) para persistir entre
reinicios del contenedor.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from core.settings import PREDICTIONS_DB_PATH

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "db" / "predictions_schema.sql"


def init_db() -> None:
    with _connection() as conn:
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(PREDICTIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_prediction(
    target_date: str,
    pair_code: str,
    mode: str,
    predicted_log_range: float,
    predicted_range_pct: float,
    anchor_close: float,
    predicted_range_points: float,
    model_name: str,
    model_version: str | None,
    mlflow_run_id: str | None,
    trace_id: str | None,
) -> int:
    with _connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions
                (target_date, predicted_at, pair_code, mode,
                 predicted_log_range, predicted_range_pct, anchor_close, predicted_range_points,
                 model_name, model_version, mlflow_run_id, trace_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target_date,
                datetime.now(timezone.utc).isoformat(),
                pair_code,
                mode,
                predicted_log_range,
                predicted_range_pct,
                anchor_close,
                predicted_range_points,
                model_name,
                model_version,
                mlflow_run_id,
                trace_id,
            ),
        )
        return cursor.lastrowid


def update_actual_range(target_date: str, pair_code: str, actual_range: float) -> int:
    """Completa `actual_range` una vez que el Data Service ya conoce el
    valor real de ese día (RF24: historial con resultado real)."""
    with _connection() as conn:
        cursor = conn.execute(
            "UPDATE predictions SET actual_range = ? WHERE target_date = ? AND pair_code = ? AND mode = 'automatic'",
            (actual_range, target_date, pair_code),
        )
        return cursor.rowcount


def fetch_history(
    pair_code: str | None = None,
    mode: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 500,
) -> list[dict]:
    query = "SELECT * FROM predictions WHERE 1=1"
    params: list = []
    if pair_code:
        query += " AND pair_code = ?"
        params.append(pair_code)
    if mode:
        query += " AND mode = ?"
        params.append(mode)
    if date_from:
        query += " AND target_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND target_date <= ?"
        params.append(date_to)
    query += " ORDER BY target_date DESC LIMIT ?"
    params.append(limit)

    with _connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]