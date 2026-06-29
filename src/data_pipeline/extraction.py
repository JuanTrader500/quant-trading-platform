"""
extraction.py
-------------
Downloads OHLCV data from yfinance for all assets defined in config/assets.yaml
and persists them as daily CSVs under data/raw/.

Supports incremental updates: if a CSV already exists it merges new data with
the existing file, preserving any dates that the API may no longer return
(e.g. far-back history).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical filename for every known asset
# Add new assets here — no other file needs to change.
# ---------------------------------------------------------------------------
FILENAME_MAP: dict[str, str] = {
    "sp500": "sp500_data_daily.csv",
    "vix":   "vix_data_daily.csv",
    "nq":    "nq_data_daily.csv",
    "vxn":   "vxn_data_daily.csv",
}


class DataExtractor:
    """Downloads and persists daily OHLCV data for a list of assets."""

    def __init__(self, start_date: str = "2005-01-01", data_dir: str | Path | None = None):
        self.start_date = start_date
        self.end_date   = datetime.now()
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            project_root  = self._find_project_root(Path(__file__).resolve())
            self.data_dir = project_root / "data" / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataExtractor — raw data dir: {self.data_dir}")

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        """Walk up until a folder containing both 'src' and 'data' is found."""
        for parent in [start, *start.parents]:
            if (parent / "src").is_dir() and (parent / "data").is_dir():
                return parent
        return start.parents[2]  # fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_asset(self, ticker: str, asset_name: str, is_volatility: bool = False) -> bool:
        """Download a single asset and save / merge its CSV."""
        logger.info(f"Downloading {asset_name} ({ticker}) …")
        try:
            raw = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            df  = self._process(raw, asset_name)
            if df is None:
                return False

            df = self._add_derived_columns(df, is_volatility)

            file_path = self.data_dir / FILENAME_MAP.get(asset_name, f"{asset_name}_data_daily.csv")
            if file_path.exists():
                df = self._merge_with_integrity(df, file_path, asset_name)

            df.to_csv(file_path, index=False)
            logger.info(f"Saved {file_path.name}  ({len(df):,} rows)")
            return True

        except Exception as exc:
            logger.error(f"Failed to download {asset_name}: {exc}")
            return False

    def download_all(self, assets_config: list[dict]) -> dict[str, bool]:
        """Download every asset in the config list."""
        return {
            asset["name"]: self.download_asset(
                ticker=asset["ticker"],
                asset_name=asset["name"],
                is_volatility=asset.get("vol", False),
            )
            for asset in assets_config
        }

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_config(config_path: str | Path | None = None) -> list[dict]:
        project_root = Path(__file__).resolve().parents[2]
        config_path  = Path(config_path) if config_path else project_root / "config" / "assets.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg["assets"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _process(data: pd.DataFrame, asset_name: str) -> pd.DataFrame | None:
        """Flatten a yfinance MultiIndex DataFrame into a clean OHLCV frame."""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = pd.DataFrame(
                {
                    "date":  data.index,
                    "open":  data["Open"].values,
                    "high":  data["High"].values,
                    "low":   data["Low"].values,
                    "close": data["Close"].values,
                }
            )
            if "Volume" in data.columns:
                df["volume"] = data["Volume"].values

            return df.dropna().reset_index(drop=True)
        except Exception as exc:
            logger.error(f"Error processing {asset_name}: {exc}")
            return None

    @staticmethod
    def _add_derived_columns(df: pd.DataFrame, is_volatility: bool) -> pd.DataFrame:
        if is_volatility:
            df["avg_hl"] = (df["high"] + df["low"]) / 2
        else:
            df["pct_move"] = (df["high"] - df["low"]) / df["open"] * 100
        return df

    @staticmethod
    def _merge_with_integrity(
        new_df: pd.DataFrame,
        existing_path: Path,
        asset_name: str,
    ) -> pd.DataFrame:
        """
        Merge the freshly downloaded frame with the on-disk CSV.
        Dates already on disk but absent from the API response are preserved
        (they may have been delisted or the API window may not reach them).
        New rows and updated rows overwrite the old ones.
        """
        existing = pd.read_csv(existing_path)
        existing["date"] = pd.to_datetime(existing["date"])
        new_df["date"]   = pd.to_datetime(new_df["date"])

        missing = set(existing["date"]) - set(new_df["date"])
        if missing:
            logger.warning(
                f"{asset_name}: {len(missing)} historical date(s) not in API response — preserved from disk."
            )

        merged = (
            pd.concat([existing[existing["date"].isin(missing)], new_df])
            .drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        return merged


# ---------------------------------------------------------------------------
# Entry-point (can also be called directly for one-off refreshes)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    extractor = DataExtractor()
    config    = DataExtractor.load_config()
    results   = extractor.download_all(config)
    failed    = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error(f"Assets that failed to download: {failed}")
        raise SystemExit(1)
    logger.info("All assets downloaded successfully.")