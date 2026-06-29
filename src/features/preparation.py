"""
preparation.py
--------------
Transforms raw OHLCV CSVs into feature-engineered datasets ready for model
training.

Two independent pipelines run in sequence:
  • SP500 (^GSPC)  +  VIX  → processed_sp500.csv
  • NQ   (^NDX)    +  VXN  → processed_nq.csv

Both share the same cleaning / feature-engineering logic defined once in
DataPreparer.  Outputs land in  data/processed/.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset definitions
# Extend this list to add new asset pairs — nothing else needs to change.
# ---------------------------------------------------------------------------
DATASET_CONFIGS: list[dict] = [
    {
        "name":        "sp500",
        "equity_file": "sp500_data_daily.csv",
        "vol_file":    "vix_data_daily.csv",
        "vol_prefix":  "vix",
        "output_file": "processed_sp500.csv",
    },
    {
        "name":        "nq",
        "equity_file": "nq_data_daily.csv",
        "vol_file":    "vxn_data_daily.csv",
        "vol_prefix":  "vxn",
        "output_file": "processed_nq.csv",
    },
]


class DataPreparer:
    """Cleans and feature-engineers raw CSVs for a given asset + volatility pair."""

    def __init__(
        self,
        raw_data_dir: str | Path | None = None,
        processed_data_dir: str | Path | None = None,
    ):
        # If explicit paths are provided (e.g. from PipelineManager), use them directly.
        # Only fall back to auto-detection when called standalone.
        if raw_data_dir and processed_data_dir:
            self.raw_data_dir       = Path(raw_data_dir)
            self.processed_data_dir = Path(processed_data_dir)
        else:
            # Walk up from this file until we find the project root
            # (identified by having both a 'src' and a 'data' sibling directory).
            project_root = self._find_project_root(Path(__file__).resolve())
            self.raw_data_dir       = Path(raw_data_dir)       if raw_data_dir       else project_root / "data" / "raw"
            self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else project_root / "data" / "processed"

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataPreparer — raw: {self.raw_data_dir}")
        logger.info(f"DataPreparer — processed: {self.processed_data_dir}")

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        """Walk up directory tree until a folder containing both 'src' and 'data' is found."""
        for parent in [start, *start.parents]:
            if (parent / "src").is_dir() and (parent / "data").is_dir():
                return parent
        # Fallback: parents[2] relative to this file
        return start.parents[2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict[str, pd.DataFrame]:
        """
        Execute the full preparation pipeline for every dataset config.
        Returns a dict  { dataset_name: processed_df }.
        """
        results: dict[str, pd.DataFrame] = {}
        for cfg in DATASET_CONFIGS:
            logger.info(f"[{cfg['name'].upper()}] Starting preparation pipeline …")
            try:
                df = self._load_and_merge(cfg)
                df = self._clean(df, vol_prefix=cfg["vol_prefix"])
                df = self._engineer_features(df, asset_prefix=cfg["name"], vol_prefix=cfg["vol_prefix"])
                output_path = self.processed_data_dir / cfg["output_file"]
                df.to_csv(output_path, index=False)
                logger.info(f"[{cfg['name'].upper()}] Saved → {output_path}  (shape {df.shape})")
                results[cfg["name"]] = df
            except FileNotFoundError as exc:
                logger.error(f"[{cfg['name'].upper()}] Raw file missing: {exc}. Skipping this dataset.")
            except Exception as exc:
                logger.error(f"[{cfg['name'].upper()}] Pipeline error: {exc}", exc_info=True)
        return results

    # ------------------------------------------------------------------
    # Private — loading
    # ------------------------------------------------------------------

    def _load_and_merge(self, cfg: dict) -> pd.DataFrame:
        """Load equity + volatility CSVs and inner-join on date."""
        equity_path = self.raw_data_dir / cfg["equity_file"]
        vol_path    = self.raw_data_dir / cfg["vol_file"]
        vol_prefix  = cfg["vol_prefix"]
        asset_name  = cfg["name"]

        for p in (equity_path, vol_path):
            if not p.exists():
                raise FileNotFoundError(p)

        # ---- equity ----
        eq = pd.read_csv(equity_path, parse_dates=["date"])
        eq = eq.rename(columns={c: f"{asset_name}_{c}" for c in eq.columns if c != "date"})

        # ---- volatility ----
        vol = pd.read_csv(vol_path, parse_dates=["date"])
        vol = vol.rename(columns={c: f"{vol_prefix}_{c}" for c in vol.columns if c != "date"})

        merged = eq.merge(vol, on="date", how="inner").sort_values("date").reset_index(drop=True)
        logger.info(
            f"[{asset_name.upper()}] Merged {len(eq):,} equity rows × {len(vol):,} vol rows "
            f"→ {len(merged):,} common dates"
        )
        return merged

    # ------------------------------------------------------------------
    # Private — cleaning
    # ------------------------------------------------------------------

    def _clean(self, df: pd.DataFrame, vol_prefix: str, train_cutoff: str = "2020-12-31") -> pd.DataFrame:
        """IQR outlier removal on the volatility high, fitted on the training slice only."""
        vol_high_col = f"{vol_prefix}_high"
        if vol_high_col not in df.columns:
            logger.warning(f"Column {vol_high_col!r} not found — skipping outlier cleaning.")
            return df

        df = df.copy().sort_values("date")
        train_mask = df["date"] <= train_cutoff
        Q1, Q3 = df.loc[train_mask, vol_high_col].quantile([0.25, 0.75])
        IQR     = Q3 - Q1
        lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        before = len(df)
        df = df[(df[vol_high_col] >= lo) & (df[vol_high_col] <= hi)].reset_index(drop=True)
        logger.info(f"Outlier removal on {vol_high_col}: {before:,} → {len(df):,} rows")
        return df

    # ------------------------------------------------------------------
    # Private — feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(
        self,
        df: pd.DataFrame,
        asset_prefix: str,
        vol_prefix: str,
    ) -> pd.DataFrame:
        """
        Build stationary, leak-free features for one equity+vol pair.

        Naming convention
        -----------------
        All raw prefixed columns are dropped at the end.
        Engineered columns use plain names (close_log, log_return, …) so the
        downstream model code is asset-agnostic.
        """
        logger.info(f"Engineering features for {asset_prefix.upper()} + {vol_prefix.upper()} …")
        df = df.copy()

        eq_close = f"{asset_prefix}_close"
        eq_high  = f"{asset_prefix}_high"
        eq_low   = f"{asset_prefix}_low"

        vol_close = f"{vol_prefix}_close"
        vol_high  = f"{vol_prefix}_high"
        vol_low   = f"{vol_prefix}_low"

        # Guard: all required equity prices must be positive
        required_positive = [c for c in [eq_close, eq_high, eq_low, vol_close, vol_high, vol_low]
                             if c in df.columns]
        df = df[(df[required_positive] > 0).all(axis=1)]

        # ----------------------------------------------------------------
        # 1. Log prices
        # ----------------------------------------------------------------
        df["close_log"] = np.log(df[eq_close])
        df["high_log"]  = np.log(df[eq_high])
        df["low_log"]   = np.log(df[eq_low])

        # ----------------------------------------------------------------
        # 2. Targets  (no lag — these ARE what the model predicts)
        # ----------------------------------------------------------------
        df["target_high"] = df["high_log"]  - df["close_log"].shift(1)   # log-distance to today's high
        df["target_low"]  = df["low_log"]   - df["close_log"].shift(1)   # log-distance to today's low
        df["log_target"]  = df["close_log"].diff()                        # daily log-return

        # ----------------------------------------------------------------
        # 3. Lagged equity features  (t-1 → no leakage)
        # ----------------------------------------------------------------
        df["log_return"]      = df["close_log"].diff().shift(1)
        df["close_log_lag1"]  = df["close_log"].shift(1)
        df["Upper_Wick_lag1"] = (df["high_log"]  - df["close_log"]).shift(1)
        df["Lower_Wick_lag1"] = (df["close_log"] - df["low_log"]).shift(1)

        # ----------------------------------------------------------------
        # 4. Lagged volatility features
        # ----------------------------------------------------------------
        if vol_close in df.columns:
            # Normalised daily vol: VIX-equivalent / (100 * sqrt(252))
            df["Vol_Diaria_lag1"] = (df[vol_close] / (100 * np.sqrt(252))).shift(1)

        if vol_high in df.columns and vol_low in df.columns:
            # Log range of the volatility index
            df["Vol_Rango_Log_lag1"] = (np.log(df[vol_high]) - np.log(df[vol_low])).shift(1)

        # ----------------------------------------------------------------
        # 5. Drop raw prefixed columns (prevent leakage / keep frame lean)
        # ----------------------------------------------------------------
        raw_prefixes = (asset_prefix, vol_prefix)
        cols_to_drop = [c for c in df.columns if any(c.startswith(f"{p}_") for p in raw_prefixes)]
        df = df.drop(columns=cols_to_drop, errors="ignore").dropna()

        logger.info(f"Feature engineering complete → shape {df.shape}")
        return df


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    preparer = DataPreparer()
    results  = preparer.run_pipeline()
    if not results:
        raise SystemExit(1)