"""
validation.py
--------------
RF04: valida la integridad de los datos descargados (nulos, duplicados,
gaps de fechas por días no hábiles) antes de persistirlos.

Todo hallazgo se registra en logs. Solo se corrige lo que compromete la
integridad del dataset (nulos y duplicados); los gaps por feriados son
esperables en un calendario bursátil y solo se reportan.
"""

import pandas as pd

from .logging_config import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Aplica chequeos de calidad sobre un DataFrame OHLC antes de guardarlo."""

    REQUIRED_COLUMNS = ("date", "open", "high", "low", "close")

    @classmethod
    def validate(cls, df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        df = cls._drop_nulls(df, asset_name)
        df = cls._drop_duplicates(df, asset_name)
        cls._report_date_gaps(df, asset_name)
        return df

    @staticmethod
    def _drop_nulls(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        cols = [c for c in DataValidator.REQUIRED_COLUMNS if c in df.columns]
        before = len(df)
        df = df.dropna(subset=cols).reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.warning(f"{asset_name}: {removed} fila(s) con nulos eliminadas.")
        return df

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
        removed = before - len(df)
        if removed:
            logger.warning(f"{asset_name}: {removed} fecha(s) duplicada(s) eliminadas.")
        return df

    @staticmethod
    def _report_date_gaps(df: pd.DataFrame, asset_name: str) -> None:
        if df.empty:
            return
        expected = pd.bdate_range(df["date"].min(), df["date"].max())
        missing = expected.difference(pd.to_datetime(df["date"]))
        if len(missing):
            logger.info(
                f"{asset_name}: {len(missing)} día(s) hábil(es) sin dato "
                f"(feriados de mercado u otros no-trading days)."
            )
