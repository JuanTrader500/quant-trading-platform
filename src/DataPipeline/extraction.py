"""
extraction.py
-------------
RF01: descarga velas diarias OHLC del SP500/NQ y sus índices de
volatilidad (VIX/VXN) vía yfinance.
RF05: actualización incremental — si el CSV ya existe, se mergea con lo
nuevo preservando fechas históricas que la API ya no devuelva.
RF04: valida integridad (nulos, duplicados, gaps) antes de persistir.
RF06: registra fecha de ejecución, rango de fechas obtenido y errores
de conexión con la API externa.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from .logging_config import get_logger
from .settings import ASSETS_CONFIG_PATH, RAW_DATA_DIR, DEFAULT_START_DATE
from .validation import DataValidator

logger = get_logger(__name__)

FILENAME_MAP: dict[str, str] = {
    "sp500": "sp500_data_daily.csv",
    "vix": "vix_data_daily.csv",
    "nq": "nq_data_daily.csv",
    "vxn": "vxn_data_daily.csv",
}


class DataExtractor:
    """Descarga y persiste OHLCV diario para una lista de activos."""

    def __init__(self, start_date: str = DEFAULT_START_DATE, data_dir: str | Path | None = None):
        self.start_date = start_date
        self.end_date = datetime.now()
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_asset(self, ticker: str, asset_name: str, is_volatility: bool = False) -> bool:
        run_ts = datetime.now().isoformat(timespec="seconds")
        logger.info(f"[{asset_name}] Ejecución {run_ts} — descargando {ticker} …")
        try:
            raw = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
        except Exception as exc:
            logger.error(f"[{asset_name}] Error de conexión con Yahoo Finance: {exc}")
            return False

        df = self._process(raw, asset_name)
        if df is None or df.empty:
            logger.error(f"[{asset_name}] Sin datos utilizables tras el procesamiento.")
            return False

        df = DataValidator.validate(df, asset_name)
        df = self._add_derived_columns(df, is_volatility)

        file_path = self.data_dir / FILENAME_MAP.get(asset_name, f"{asset_name}_data_daily.csv")
        if file_path.exists():
            df = self._merge_with_integrity(df, file_path, asset_name)

        df.to_csv(file_path, index=False)
        logger.info(
            f"[{asset_name}] Rango obtenido {df['date'].min().date()} → "
            f"{df['date'].max().date()}  ({len(df):,} filas) → {file_path.name}"
        )
        return True

    def download_all(self, assets_config: list[dict]) -> dict[str, bool]:
        return {
            asset["name"]: self.download_asset(
                ticker=asset["ticker"],
                asset_name=asset["name"],
                is_volatility=asset.get("vol", False),
            )
            for asset in assets_config
        }

    @staticmethod
    def load_config(config_path: str | Path | None = None) -> list[dict]:
        path = Path(config_path) if config_path else ASSETS_CONFIG_PATH
        with open(path) as f:
            return yaml.safe_load(f)["assets"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process(data: pd.DataFrame, asset_name: str) -> pd.DataFrame | None:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = pd.DataFrame({
                "date": data.index,
                "open": data["Open"].values,
                "high": data["High"].values,
                "low": data["Low"].values,
                "close": data["Close"].values,
            })
            if "Volume" in data.columns:
                df["volume"] = data["Volume"].values
            return df.dropna().reset_index(drop=True)
        except Exception as exc:
            logger.error(f"[{asset_name}] Error procesando respuesta de la API: {exc}")
            return None

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame, is_volatility: bool) -> pd.DataFrame:
        if is_volatility:
            df["avg_hl"] = (df["high"] + df["low"]) / 2
        else:
            df["pct_move"] = (df["high"] - df["low"]) / df["open"] * 100
        return df

    @staticmethod
    def _merge_with_integrity(new_df: pd.DataFrame, existing_path: Path, asset_name: str) -> pd.DataFrame:
        existing = pd.read_csv(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])
        new_df["date"] = pd.to_datetime(new_df["date"])

        missing = set(existing["date"]) - set(new_df["date"])
        if missing:
            logger.info(f"[{asset_name}] {len(missing)} fecha(s) histórica(s) preservadas del disco.")

        return (
            pd.concat([existing[existing["date"].isin(missing)], new_df])
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )


if __name__ == "__main__":
    extractor = DataExtractor()
    results = extractor.download_all(DataExtractor.load_config())
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error(f"Activos que fallaron: {failed}")
        raise SystemExit(1)
    logger.info("Todos los activos descargados correctamente.")
