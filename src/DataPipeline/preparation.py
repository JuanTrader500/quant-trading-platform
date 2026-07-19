"""
preparation.py
---------------
RF02/RF03: calcula las features del Data Dictionary usando únicamente
información disponible hasta el cierre de t para predecir t+1 (sin
data leakage). Ver Readme del proyecto para la justificación de cada
variable.
RNF12: al final de cada corrida, fuerza el orden de columnas del
esquema versionado y persiste su manifiesto (feature_schema.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .feature_schema import enforce_schema, write_schema_manifest
from .logging_config import get_logger
from .settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = get_logger(__name__)

# Agregar un nuevo par de activos = agregar una entrada aquí (RNF11).
DATASET_CONFIGS: list[dict] = [
    {"name": "sp500", "equity_file": "sp500_data_daily.csv", "vol_file": "vix_data_daily.csv",
     "vol_prefix": "vix", "output_file": "processed_sp500.csv"},
    {"name": "nq", "equity_file": "nq_data_daily.csv", "vol_file": "vxn_data_daily.csv",
     "vol_prefix": "vxn", "output_file": "processed_nq.csv"},
]


class DataPreparer:
    """Limpia y genera features para un par activo + índice de volatilidad."""

    def __init__(self, raw_data_dir: str | Path | None = None, processed_data_dir: str | Path | None = None):
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else RAW_DATA_DIR
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for cfg in DATASET_CONFIGS:
            name = cfg["name"]
            logger.info(f"[{name.upper()}] Iniciando preparación …")
            try:
                df = self._load_and_merge(cfg)
                df = self._clean(df, vol_prefix=cfg["vol_prefix"])
                df = self._engineer_features(df, asset_prefix=name, vol_prefix=cfg["vol_prefix"])
                df = enforce_schema(df, name)

                output_path = self.processed_data_dir / cfg["output_file"]
                df.to_csv(output_path, index=True)
                write_schema_manifest(name, self.processed_data_dir)

                logger.info(f"[{name.upper()}] Guardado → {output_path.name}  (shape {df.shape})")
                results[name] = df
            except FileNotFoundError as exc:
                logger.error(f"[{name.upper()}] Archivo crudo faltante: {exc}. Se omite este dataset.")
            except Exception as exc:
                logger.error(f"[{name.upper()}] Error en el pipeline: {exc}", exc_info=True)
        return results

    # ------------------------------------------------------------------
    # Carga
    # ------------------------------------------------------------------

    def _load_and_merge(self, cfg: dict) -> pd.DataFrame:
        equity_path = self.raw_data_dir / cfg["equity_file"]
        vol_path = self.raw_data_dir / cfg["vol_file"]
        for p in (equity_path, vol_path):
            if not p.exists():
                raise FileNotFoundError(p)

        eq = pd.read_csv(equity_path, parse_dates=["date"])
        eq = eq.rename(columns={c: f"{cfg['name']}_{c}" for c in eq.columns if c != "date"})

        vol = pd.read_csv(vol_path, parse_dates=["date"])
        vol = vol.rename(columns={c: f"{cfg['vol_prefix']}_{c}" for c in vol.columns if c != "date"})

        return eq.merge(vol, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Limpieza
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Limpieza (DESACTIVADA PARA MODELADO DE VOLATILIDAD)
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame, vol_prefix: str, train_cutoff: str = "2020-12-31") -> pd.DataFrame:
        # Devolvemos el DataFrame intacto sin aplicar ningún filtro IQR
        logger.info("Filtro de outliers desactivado. Pasando toda la data cruda al pipeline.")
        return df

    # ------------------------------------------------------------------
    # Feature engineering (RF02 / RF03 — ver Readme para justificación)
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame, asset_prefix: str, vol_prefix: str) -> pd.DataFrame:
        df = df.copy()

        eq_open, eq_high, eq_low, eq_close = (f"{asset_prefix}_{c}" for c in ("open", "high", "low", "close"))
        vol_high, vol_low, vol_close = (f"{vol_prefix}_{c}" for c in ("high", "low", "close"))

        required_positive = [c for c in [eq_open, eq_high, eq_low, eq_close, vol_high, vol_low, vol_close]
                              if c in df.columns]
        df = df[(df[required_positive] > 0).all(axis=1)]

        # Target: rango logarítmico de mañana (t+1), calculado con datos futuros
        # y desplazado para no filtrarse como feature de hoy.
        df["target"] = (np.log(df[eq_high]) - np.log(df[eq_low])).shift(-1)

        log_close, log_open = np.log(df[eq_close]), np.log(df[eq_open])
        log_high, log_low = np.log(df[eq_high]), np.log(df[eq_low])

        df[f"{asset_prefix}_log_return"] = log_close - log_close.shift(1)
        df[f"{asset_prefix}_log_range"] = log_high - log_low
        df[f"{asset_prefix}_body_log"] = log_close - log_open
        df[f"{asset_prefix}_upper_wick_log"] = log_high - np.log(np.maximum(df[eq_open], df[eq_close]))
        df[f"{asset_prefix}_lower_wick_log"] = np.log(np.minimum(df[eq_open], df[eq_close])) - log_low
        df[f"{asset_prefix}_vol_5d"] = df[f"{asset_prefix}_log_return"].rolling(5).std()
        df[f"{asset_prefix}_vol_10d"] = df[f"{asset_prefix}_log_return"].rolling(10).std()

        log_vol_close = np.log(df[vol_close])
        df[f"{vol_prefix}_log_close"] = log_vol_close
        df[f"{vol_prefix}_log_range"] = np.log(df[vol_high]) - np.log(df[vol_low])
        df[f"{vol_prefix}_log_return"] = log_vol_close - log_vol_close.shift(1)

        df["day_of_week"] = df["date"].dt.dayofweek
        df = df.set_index("date")
        return df.dropna()


if __name__ == "__main__":
    results = DataPreparer().run_pipeline()
    if not results:
        raise SystemExit(1)
