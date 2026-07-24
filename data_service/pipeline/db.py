"""
db.py
-----
Capa de acceso a datos del Data Service (PostgreSQL + TimescaleDB).

Aísla al resto del pipeline de SQL: `extraction.py` y `preparation.py`
llaman a estas funciones y nunca escriben queries directamente
(RNF10/RNF11) — así, si el día de mañana cambia el motor de base de
datos, solo se toca este archivo.

El connection string se lee de la variable de entorno `DATABASE_URL`
(ver `.env.example`), nunca se hardcodea (RNF06). El engine se crea de
forma perezosa (lazy) la primera vez que se usa, para que importar este
módulo no falle si `.env` todavía no está configurado.

Corresponde 1:1 al esquema de `docs/data_service_schema.sql`.
"""

import json
from contextlib import contextmanager
from datetime import date
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from .feature_schema import FEATURE_COLUMNS
from .logging_config import get_logger
from .settings import DATABASE_URL

load_dotenv()
logger = get_logger(__name__)

_ENGINE: Engine | None = None

# Columnas de la tabla `features` que no son PK ni FK, tal como las
# define el esquema de features.py. Se reutiliza aquí para no mantener
# dos listas de columnas que puedan desincronizarse.
_FEATURE_VALUE_COLUMNS = FEATURE_COLUMNS


def get_engine() -> Engine:
    """Crea (una sola vez) y devuelve el engine de SQLAlchemy.

    Lanza un error explícito y accionable si DATABASE_URL no está
    configurada, en vez de fallar con un traceback críptico de
    SQLAlchemy más adelante.
    """
    global _ENGINE
    if _ENGINE is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL no está definida. Copia .env.example a .env "
                "en la raíz de data_service/ y completa la cadena de conexión."
            )
        _ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _ENGINE


@contextmanager
def _connection() -> Iterator[Connection]:
    """Context manager interno: una conexión con transacción autocommit
    al salir del bloque (o rollback automático si hay excepción)."""
    engine = get_engine()
    with engine.begin() as conn:
        yield conn


# ---------------------------------------------------------------------
# raw_ohlc — datos crudos (caso de uso 1 / RF01)
# ---------------------------------------------------------------------

def get_latest_raw_date(ticker: str) -> date | None:
    """Última fecha ya almacenada en raw_ohlc para un ticker.

    Es la base de la actualización incremental (RF05): extraction.py
    solo pide a Yahoo Finance lo que venga después de esta fecha.
    """
    with _connection() as conn:
        result = conn.execute(
            text("SELECT max(date) FROM raw_ohlc WHERE ticker = :ticker"),
            {"ticker": ticker},
        ).scalar()
    return result


def upsert_raw_ohlc(df: pd.DataFrame, ticker: str) -> int:
    """Inserta o actualiza velas OHLCV de un ticker (upsert idempotente).

    Usa una tabla temporal + INSERT ... ON CONFLICT para poder cargar el
    lote completo de una vez en lugar de fila por fila. Devuelve la
    cantidad de filas escritas.
    """
    if df.empty:
        return 0

    payload = df.assign(ticker=ticker)[["ticker", "date", "open", "high", "low", "close", "volume"]]

    with _connection() as conn:
        conn.execute(text(
            "CREATE TEMP TABLE IF NOT EXISTS _raw_ohlc_staging "
            "(LIKE raw_ohlc INCLUDING DEFAULTS) ON COMMIT DROP"
        ))
        payload.to_sql("_raw_ohlc_staging", conn, if_exists="append", index=False, method="multi", chunksize=500)
        conn.execute(text("""
            INSERT INTO raw_ohlc (ticker, date, open, high, low, close, volume)
            SELECT ticker, date, open, high, low, close, volume FROM _raw_ohlc_staging
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                updated_at = now()
        """))
    return len(payload)


def fetch_raw_ohlc(ticker: str) -> pd.DataFrame:
    """Trae el histórico completo de OHLCV de un ticker, ordenado por
    fecha. Usado por preparation.py para calcular features."""
    with _connection() as conn:
        df = pd.read_sql(
            text("SELECT date, open, high, low, close, volume FROM raw_ohlc "
                 "WHERE ticker = :ticker ORDER BY date"),
            conn, params={"ticker": ticker},
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------
# features — datos procesados (caso de uso 2 / RF02, RF03)
# ---------------------------------------------------------------------

def upsert_features(df: pd.DataFrame, pair_code: str, schema_version: str) -> int:
    """Inserta o actualiza filas de features para un par.

    `df` debe traer columna `date` y todas las columnas de
    `feature_schema.FEATURE_COLUMNS`. Los NaN se convierten a NULL de
    SQL explícitamente (importante para `target_range_next_day`, que
    es NULL en la fila más reciente hasta que se conoce el día
    siguiente — ver preparation.py).
    """
    if df.empty:
        return 0

    payload = df.assign(pair_code=pair_code, schema_version=schema_version)
    payload = payload[["pair_code", "date", "schema_version", *_FEATURE_VALUE_COLUMNS]]
    payload = payload.astype(object).where(pd.notnull(payload), None)

    with _connection() as conn:
        conn.execute(text(
            "CREATE TEMP TABLE IF NOT EXISTS _features_staging "
            "(LIKE features INCLUDING DEFAULTS) ON COMMIT DROP"
        ))
        payload.to_sql("_features_staging", conn, if_exists="append", index=False, method="multi", chunksize=500)

        columns_sql = ", ".join(_FEATURE_VALUE_COLUMNS)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in ["schema_version", *_FEATURE_VALUE_COLUMNS])
        conn.execute(text(f"""
            INSERT INTO features (pair_code, date, schema_version, {columns_sql})
            SELECT pair_code, date, schema_version, {columns_sql} FROM _features_staging
            ON CONFLICT (pair_code, date) DO UPDATE SET
                {set_clause},
                updated_at = now()
        """))
    return len(payload)


def fetch_latest_features(pair_code: str) -> dict | None:
    """Última fila de features de un par (equivalente a la vista
    v_latest_features). Usado por el endpoint /features/latest (RF15)."""
    with _connection() as conn:
        row = conn.execute(
            text("SELECT * FROM features WHERE pair_code = :p ORDER BY date DESC LIMIT 1"),
            {"p": pair_code},
        ).mappings().first()
    return dict(row) if row else None


def fetch_training_dataset(pair_code: str, date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    """Filas de features con target ya conocido, listas para entrenar
    (equivalente a la vista v_training_dataset). Caso de uso 3."""
    query = "SELECT * FROM features WHERE pair_code = :p AND target_range_next_day IS NOT NULL"
    params: dict = {"p": pair_code}
    if date_from:
        query += " AND date >= :date_from"
        params["date_from"] = date_from
    if date_to:
        query += " AND date <= :date_to"
        params["date_to"] = date_to
    query += " ORDER BY date"

    with _connection() as conn:
        rows = conn.execute(text(query), params).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# feature_schema_versions — versionado de esquema (caso de uso 5 / RNF12)
# ---------------------------------------------------------------------

def get_current_schema_version() -> str | None:
    """Versión de esquema marcada como vigente en la base de datos."""
    with _connection() as conn:
        return conn.execute(
            text("SELECT version FROM feature_schema_versions WHERE is_current LIMIT 1")
        ).scalar()


def register_schema_version(version: str, columns: list[str]) -> None:
    """Registra una versión de esquema (si no existía) y la marca como
    vigente, dejando cualquier otra versión anterior como no-vigente.

    Se llama al inicio de cada corrida de preparation.py: si el código
    de feature_schema.py cambió de versión, queda reflejado en la base
    de datos automáticamente, sin pasos manuales.
    """
    with _connection() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM feature_schema_versions WHERE version = :v"), {"v": version}
        ).scalar()
        if not exists:
            conn.execute(text("""
                INSERT INTO feature_schema_versions (version, columns, is_current)
                VALUES (:v, CAST(:cols AS JSONB), false)
            """), {"v": version, "cols": json.dumps({"columns": columns})})
        conn.execute(
            text("UPDATE feature_schema_versions SET is_current = (version = :v)"),
            {"v": version},
        )


# ---------------------------------------------------------------------
# ingestion_log — auditoría de corridas (caso de uso 4 / RF06)
# ---------------------------------------------------------------------

def log_run(
    pipeline_stage: str,
    status: str,
    ticker: str | None = None,
    pair_code: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    rows_affected: int | None = None,
    error_message: str | None = None,
    pipeline_version: str | None = None,
    duration_ms: int | None = None,
) -> None:
    """Registra una corrida del pipeline en ingestion_log.

    `pipeline_stage` es 'extraction' (requiere `ticker`) o
    'feature_engineering' (requiere `pair_code`), reflejando el CHECK
    constraint de la tabla.
    """
    with _connection() as conn:
        conn.execute(text("""
            INSERT INTO ingestion_log
                (pipeline_stage, ticker, pair_code, date_from, date_to,
                 rows_affected, status, error_message, pipeline_version, duration_ms)
            VALUES
                (:stage, :ticker, :pair, :date_from, :date_to,
                 :rows, :status, :error, :pipeline_version, :duration_ms)
        """), {
            "stage": pipeline_stage,
            "ticker": ticker,
            "pair": pair_code,
            "date_from": date_from,
            "date_to": date_to,
            "rows": rows_affected,
            "status": status,
            "error": error_message,
            "pipeline_version": pipeline_version,
            "duration_ms": duration_ms,
        })
