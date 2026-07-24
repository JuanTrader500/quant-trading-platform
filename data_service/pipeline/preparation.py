"""
preparation.py
---------------
RF02/RF03: calcula las features del Data Dictionary usando únicamente
información disponible hasta el cierre de t para predecir t+1 (sin
data leakage): todas las columnas de features se calculan con datos
≤ t; `target_range_next_day` es la única columna que mira hacia
adelante (shift(-1)), y queda NULL para el día más reciente hasta que
el día siguiente ya tiene dato en `raw_ohlc`.

RNF12: al inicio de cada corrida registra/confirma la versión de
esquema vigente en base de datos (`feature_schema.register_current_version`)
y, antes de persistir, fuerza el orden de columnas del esquema
(`feature_schema.enforce_schema`).

Lee de `raw_ohlc` y escribe en `features` vía `db.py` — no toca
archivos en disco (reemplaza el flujo anterior basado en CSV).

Decisión de diseño — sin filtrado de outliers (IQR)
----------------------------------------------------
Se evaluó filtrar outliers del índice de volatilidad (IQR sobre
`vol_high`) antes del feature engineering. Se descartó deliberadamente:
en un dataset cuyo target ES la volatilidad futura, los días que el
filtro marcaba como "atípicos" son precisamente los eventos de alta
volatilidad que el modelo necesita ver para aprender a anticiparlos.
El error de validación se mantuvo estable con y sin el filtro, así que
se conserva la serie completa.
"""

import time

import numpy as np
import pandas as pd

from . import db
from .feature_schema import FEATURE_COLUMNS, SCHEMA_VERSION, enforce_schema, register_current_version
from .logging_config import get_logger
from .registry import ASSETS, PairInfo, all_pairs
from .settings import PIPELINE_VERSION

logger = get_logger(__name__)

# Columnas de features que deben estar completas para conservar una
# fila. `target_range_next_day` se excluye a propósito: el día más
# reciente legítimamente todavía no lo conoce (RF03).
_REQUIRED_NON_NULL = [c for c in FEATURE_COLUMNS if c != "target_range_next_day"]

# Columnas OHLCV renombradas con estos prefijos tras el merge, para no
# confundir "el índice principal" con "su índice de volatilidad" al
# calcular columnas derivadas.
_OHLCV_COLS = ("open", "high", "low", "close", "volume")


class DataPreparer:
    """Genera features para cada par índice + índice de volatilidad
    definido en `registry.py`."""

    def run_pipeline(self, pairs: list[PairInfo] | None = None) -> dict[str, pd.DataFrame]:
        """Corre la preparación para todos los pares (o los indicados).
        Devuelve {pair_code: DataFrame} de los pares que sí se pudieron
        procesar; los que fallan quedan registrados en logs + ingestion_log
        pero no detienen a los demás (RNF11)."""
        register_current_version()
        pairs = pairs if pairs is not None else all_pairs()

        results: dict[str, pd.DataFrame] = {}
        for pair in pairs:
            logger.info(f"[{pair.pair_code}] Iniciando preparación …")
            try:
                results[pair.pair_code] = self._prepare_pair(pair)
            except Exception as exc:
                logger.error(f"[{pair.pair_code}] Error en el pipeline: {exc}", exc_info=True)
                db.log_run(
                    "feature_engineering", status="error", pair_code=pair.pair_code,
                    error_message=str(exc), pipeline_version=PIPELINE_VERSION,
                )
        return results

    # ------------------------------------------------------------------
    # Por par
    # ------------------------------------------------------------------

    def _prepare_pair(self, pair: PairInfo) -> pd.DataFrame:
        """Ejecuta la preparación completa de un par: carga, feature
        engineering, validación de esquema, persistencia y logging."""
        start_ts = time.monotonic()

        index_ticker = ASSETS[pair.index_asset].ticker
        vol_ticker = ASSETS[pair.volatility_asset].ticker

        df = self._load_and_merge(index_ticker, vol_ticker)
        df = self._engineer_features(df)
        df = enforce_schema(df)

        rows = db.upsert_features(df.reset_index(), pair.pair_code, SCHEMA_VERSION)
        logger.info(f"[{pair.pair_code}] {rows} fila(s) escritas en features (shape {df.shape}).")

        db.log_run(
            "feature_engineering", status="success", pair_code=pair.pair_code,
            date_from=df.index.min().date() if not df.empty else None,
            date_to=df.index.max().date() if not df.empty else None,
            rows_affected=rows, pipeline_version=PIPELINE_VERSION,
            duration_ms=int((time.monotonic() - start_ts) * 1000),
        )
        return df

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    @staticmethod
    def _load_and_merge(index_ticker: str, vol_ticker: str) -> pd.DataFrame:
        """Trae de la base de datos el histórico OHLCV del índice
        principal y de su índice de volatilidad, y los une por fecha."""
        eq = db.fetch_raw_ohlc(index_ticker).rename(columns={c: f"idx_{c}" for c in _OHLCV_COLS})
        vol = db.fetch_raw_ohlc(vol_ticker).rename(columns={c: f"vol_{c}" for c in _OHLCV_COLS})

        if eq.empty or vol.empty:
            raise ValueError(
                f"Sin datos crudos suficientes para {index_ticker}/{vol_ticker}. "
                f"Corre extraction.py primero (RNF10: etapas independientes)."
            )
        return eq.merge(vol, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Feature engineering (RF02 / RF03)
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calcula las columnas de `feature_schema.FEATURE_COLUMNS` a
        partir del OHLCV ya unido del par. Devuelve un DataFrame
        indexado por `date`."""
        df = df.copy()

        positive_cols = ["idx_open", "idx_high", "idx_low", "idx_close", "vol_high", "vol_low", "vol_close"]
        df = df[(df[positive_cols] > 0).all(axis=1)]

        # Target: rango logarítmico de mañana (t+1). Es la única columna
        # que usa shift(-1); todas las demás usan solo datos ≤ t (RF03).
        df["target_range_next_day"] = (np.log(df["idx_high"]) - np.log(df["idx_low"])).shift(-1)

        log_close, log_open = np.log(df["idx_close"]), np.log(df["idx_open"])
        log_high, log_low = np.log(df["idx_high"]), np.log(df["idx_low"])

        df["main_log_return"] = log_close - log_close.shift(1)
        df["main_log_range"] = log_high - log_low
        df["main_body_log"] = log_close - log_open
        df["main_upper_wick_log"] = log_high - np.log(np.maximum(df["idx_open"], df["idx_close"]))
        df["main_lower_wick_log"] = np.log(np.minimum(df["idx_open"], df["idx_close"])) - log_low
        df["main_vol_5d"] = df["main_log_return"].rolling(5).std()
        df["main_vol_10d"] = df["main_log_return"].rolling(10).std()

        log_vol_close = np.log(df["vol_close"])
        df["vol_idx_log_close"] = log_vol_close
        df["vol_idx_log_range"] = np.log(df["vol_high"]) - np.log(df["vol_low"])
        df["vol_idx_log_return"] = log_vol_close - log_vol_close.shift(1)

        df["day_of_week"] = df["date"].dt.dayofweek
        df = df.set_index("date")

        # Se descartan solo las filas con features incompletas (warm-up
        # de las ventanas rolling); la fila más reciente se conserva
        # aunque el target todavía sea NULL (RF03 — se completa un día
        # después, cuando raw_ohlc ya tenga el dato de t+1).
        return df.dropna(subset=_REQUIRED_NON_NULL)


if __name__ == "__main__":
    results = DataPreparer().run_pipeline()
    if not results:
        raise SystemExit(1)
