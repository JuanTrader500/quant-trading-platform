"""
extraction.py
-------------
RF01: descarga velas diarias OHLC de los activos definidos en
`registry.py` (SP500, NASDAQ y sus índices de volatilidad VIX/VXN) vía
yfinance y las persiste en la tabla `raw_ohlc` (PostgreSQL/TimescaleDB).

RF05: actualización incremental — antes de descargar, consulta en base
de datos la última fecha ya almacenada para ese ticker (`db.get_latest_raw_date`)
y solo pide a Yahoo Finance el rango faltante, en vez de re-descargar
la serie completa.

RF04: valida integridad (nulos, duplicados, gaps) antes de persistir,
delegado en `validation.DataValidator`.

RF06: cada corrida (una por ticker) queda registrada en la tabla
`ingestion_log` vía `db.log_run`, con fecha de ejecución, rango de
fechas obtenido, filas afectadas y cualquier error de conexión con la
API externa.
"""

import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

from . import db
from .logging_config import get_logger
from .registry import AssetInfo, all_assets
from .settings import DEFAULT_START_DATE, PIPELINE_VERSION
from .validation import DataValidator

logger = get_logger(__name__)


class DataExtractor:
    """Descarga y persiste OHLCV diario para los activos del registry."""

    def __init__(self, default_start_date: str = DEFAULT_START_DATE):
        self.default_start_date = default_start_date

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def download_asset(self, asset: AssetInfo) -> bool:
        """Descarga incrementalmente un único activo y lo persiste en BD.

        Devuelve True si la corrida terminó en un estado utilizable
        (incluye el caso "ya estaba al día, no había nada nuevo"), y
        False solo si hubo un error real (conexión, validación vacía).
        """
        start_ts = time.monotonic()
        start_date, end_date = self._resolve_date_range(asset.ticker)

        if start_date > end_date:
            logger.info(f"[{asset.name}] Ya está al día (última fecha {start_date - timedelta(days=1)}).")
            return True

        logger.info(f"[{asset.name}] Descargando {asset.ticker} desde {start_date} hasta {end_date} …")
        try:
            # yf.download usa `end` exclusivo: sumamos 1 día para incluir
            # el día de hoy en el rango descargado.
            raw = yf.download(
                asset.ticker,
                start=str(start_date),
                end=str(end_date + timedelta(days=1)),
                progress=False,
            )
        except Exception as exc:
            logger.error(f"[{asset.name}] Error de conexión con Yahoo Finance: {exc}")
            db.log_run(
                "extraction", status="error", ticker=asset.ticker,
                date_from=start_date, date_to=end_date, error_message=str(exc),
                pipeline_version=PIPELINE_VERSION, duration_ms=self._elapsed_ms(start_ts),
            )
            return False

        df = self._process(raw, asset.name)
        if df is None or df.empty:
            logger.warning(f"[{asset.name}] Sin datos nuevos en el rango solicitado.")
            db.log_run(
                "extraction", status="success", ticker=asset.ticker,
                date_from=start_date, date_to=end_date, rows_affected=0,
                pipeline_version=PIPELINE_VERSION, duration_ms=self._elapsed_ms(start_ts),
            )
            return True

        df = DataValidator.validate(df, asset.name)
        if df.empty:
            logger.error(f"[{asset.name}] Todas las filas fueron descartadas en validación.")
            db.log_run(
                "extraction", status="error", ticker=asset.ticker,
                date_from=start_date, date_to=end_date,
                error_message="Todas las filas descartadas en validación (RF04).",
                pipeline_version=PIPELINE_VERSION, duration_ms=self._elapsed_ms(start_ts),
            )
            return False

        rows = db.upsert_raw_ohlc(df, asset.ticker)
        logger.info(
            f"[{asset.name}] {rows} fila(s) escritas en raw_ohlc "
            f"({df['date'].min().date()} → {df['date'].max().date()})."
        )
        db.log_run(
            "extraction", status="success", ticker=asset.ticker,
            date_from=df["date"].min().date(), date_to=df["date"].max().date(),
            rows_affected=rows, pipeline_version=PIPELINE_VERSION,
            duration_ms=self._elapsed_ms(start_ts),
        )
        return True

    def download_all(self, assets: list[AssetInfo] | None = None) -> dict[str, bool]:
        """Descarga todos los activos del registry, o los indicados
        explícitamente. Agregar un activo nuevo no requiere tocar este
        método (RNF11): basta con agregarlo en registry.py."""
        assets = assets if assets is not None else all_assets()
        return {asset.name: self.download_asset(asset) for asset in assets}

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _resolve_date_range(self, ticker: str) -> tuple[date, date]:
        """RF05: calcula desde dónde descargar según lo ya almacenado
        en base de datos. Si el ticker no tiene datos aún, arranca
        desde DEFAULT_START_DATE (primera carga histórica completa)."""
        latest = db.get_latest_raw_date(ticker)
        start = latest + timedelta(days=1) if latest else date.fromisoformat(self.default_start_date)
        end = datetime.now().date()
        return start, end

    @staticmethod
    def _process(data: pd.DataFrame, asset_name: str) -> pd.DataFrame | None:
        """Normaliza la respuesta de yfinance (posible MultiIndex de
        columnas) a columnas planas OHLCV con nombres en minúscula."""
        try:
            if data.empty:
                return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = pd.DataFrame({
                "date": data.index,
                "open": data["Open"].values,
                "high": data["High"].values,
                "low": data["Low"].values,
                "close": data["Close"].values,
            })
            df["volume"] = data["Volume"].values if "Volume" in data.columns else None
            return df.dropna(subset=["date", "open", "high", "low", "close"]).reset_index(drop=True)
        except Exception as exc:
            logger.error(f"[{asset_name}] Error procesando respuesta de la API: {exc}")
            return None

    @staticmethod
    def _elapsed_ms(start_ts: float) -> int:
        """Milisegundos transcurridos desde `start_ts` (time.monotonic()),
        usado para poblar ingestion_log.duration_ms."""
        return int((time.monotonic() - start_ts) * 1000)


if __name__ == "__main__":
    extractor = DataExtractor()
    results = extractor.download_all()
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error(f"Activos que fallaron: {failed}")
        raise SystemExit(1)
    logger.info("Todos los activos descargados correctamente.")
