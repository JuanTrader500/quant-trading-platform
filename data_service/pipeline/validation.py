"""
validation.py
--------------
RF04: valida la integridad de los datos descargados (nulos,
duplicados, gaps de fechas por días no hábiles) antes de persistirlos
en `raw_ohlc`.

Todo hallazgo se registra en logs. Solo se corrige lo que compromete
la integridad del dataset (nulos y duplicados); los gaps por feriados
son esperables en un calendario bursátil y solo se reportan, no se
tratan como error.

Esta validación es la primera línea de defensa; la base de datos
aplica una segunda capa independiente mediante CHECK constraints
(high >= low, OHLC dentro de rango) — ver docs/data_service_schema.sql.
"""

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Aplica chequeos de calidad sobre un DataFrame OHLC antes de
    persistirlo en la base de datos."""

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")

    @classmethod
    def validate(cls, df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """Corre la secuencia completa de validaciones y devuelve el
        DataFrame limpio (nulos y duplicados eliminados)."""
        df = cls._drop_nulls(df, asset_name)
        df = cls._drop_duplicates(df, asset_name)
        cls._report_date_gaps(df, asset_name)
        return df

    @staticmethod
    def _drop_nulls(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """Elimina filas con nulos en columnas OHLC obligatorias."""
        cols = [c for c in DataValidator.REQUIRED_COLUMNS if c in df.columns]
        before = len(df)
        df = df.dropna(subset=cols).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.warning(f"{asset_name}: {removed} fila(s) con nulos eliminadas.")
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """Elimina fechas duplicadas, conservando el último valor recibido."""
        before = len(df)
        df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.warning(f"{asset_name}: {removed} fecha(s) duplicada(s) eliminadas.")
        return df

    @staticmethod
    def _report_date_gaps(df: pd.DataFrame, asset_name: str) -> None:
        """Reporta (sin corregir) días hábiles sin dato — feriados de
        mercado u otros días no-trading esperables."""
        if df.empty:
            return
        expected = pd.bdate_range(df["date"].min(), df["date"].max())
        missing = expected.difference(pd.to_datetime(df["date"]))
        if len(missing):
            logger.info(
                f"{asset_name}: {len(missing)} día(s) hábil(es) sin dato "
                f"(feriados de mercado u otros no-trading days)."
            )
