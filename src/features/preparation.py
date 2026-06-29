import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, raw_data_dir=None, processed_data_dir=None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else self.project_root / "data" / "raw"
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else self.project_root / "data" / "processed"
        
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def clean_outliers_iqr(self, df, column, train_cutoff="2020-12-31"):
        """
        Robust cleaning using IQR calculated only on the training set to avoid leakage.
        """
        logger.info(f"Cleaning outliers for column: {column}")
        df_copy = df.copy()
        
        # Ensure date is datetime and sorted
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy = df_copy.sort_values('date')
        
        # Calculate IQR on training slice
        train_df = df_copy.loc[df_copy['date'] <= train_cutoff]
        Q1 = train_df[column].quantile(0.25)
        Q3 = train_df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_cleaned = df_copy[(df_copy[column] >= lower_bound) & (df_copy[column] <= upper_bound)]
        logger.info(f"Original rows: {len(df)} | Rows after cleaning {column}: {len(df_cleaned)}")
        return df_cleaned

    def engineer_features(self, df):
        """
        Transforms raw prices into stationary log-relative features and targets.
        """
        logger.info("Engineering financial features...")
        df = df.copy()
        # Ensure required price columns are positive before taking logs
        positive_columns = ["close", "high", "low"]
        # detect if prefixed (merged) columns exist for main asset (sp500)
        main_prefix = "sp500_"
        main_cols = [f"{main_prefix}close", f"{main_prefix}high", f"{main_prefix}low"]
        vix_cols = ["vix_close", "vix_high", "vix_low"]
        required = [c for c in main_cols + vix_cols if c in df.columns]
        if required:
            df = df[(df[required] > 0).all(axis=1)]

        # 1. Log transformation of base prices (use main asset if available)
        if f"{main_prefix}close" in df.columns:
            df["close_log"] = np.log(df[f"{main_prefix}close"])
            df["high_log"] = np.log(df[f"{main_prefix}high"])
            df["low_log"] = np.log(df[f"{main_prefix}low"])
        else:
            df["close_log"] = np.log(df["close"])
            df["high_log"] = np.log(df["high"])
            df["low_log"] = np.log(df["low"])
        
        # 2. Target Generation (Stationary)
        # Distance from yesterday's close to today's extremes
        df["target_high"] = df["high_log"] - df["close_log"].shift(1)
        df["target_low"] = df["low_log"] - df["close_log"].shift(1)
        df["log_target"] = df["close_log"].diff()
        
        # 3. Lagged Features (t-1) to prevent data leakage
        # Log Return of yesterday
        df["log_return"] = df["close_log"].diff().shift(1)
        
        # Wicks (Price Action Psychology)
        df["Upper_Wick_lag1"] = (df["high_log"] - df["close_log"]).shift(1)
        df["Lower_Wick_lag1"] = (df["close_log"] - df["low_log"]).shift(1)
        
        # VIX relative features
        # Normalized Daily Volatility: VIX / (100 * sqrt(252))
        if "vix_close" in df.columns:
            df["VIX_Vol_Diaria_lag1"] = (df["vix_close"] / (100 * np.sqrt(252))).shift(1)
        if "vix_high" in df.columns and "vix_low" in df.columns:
            df["VIX_Rango_Log_lag1"] = (np.log(df["vix_high"]) - np.log(df["vix_low"]) ).shift(1)

        df["close_log_lag1"] = df["close_log"].shift(1)
        
        # Drop raw prices and temporary columns to avoid leakage
        # Drop both prefixed and unprefixed variants
        cols_to_drop = [
            "vix_open", "vix_high", "vix_low", "vix_close", "vix_avg_hl", "vix_volume",
            "sp500_open", "sp500_high", "sp500_low", "sp500_close", "sp500_volume", "pct_move",
            "nq_open", "nq_high", "nq_low", "nq_close", "nq_volume",
            "vxn_open", "vxn_high", "vxn_low", "vxn_close", "vxn_volume"
        ]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        df = df.dropna()
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def run_pipeline(self, asset_file="sp500_df_data_daily.csv", vix_file="vix_2026_data_daily.csv"):
        """
        Full execution from raw CSVs to processed features.
        """
        logger.info("Starting data preparation pipeline...")
        
        # Backwards compatible: if specific asset_file and vix_file are provided (tests), use the legacy flow
        asset_path = self.raw_data_dir / asset_file
        vix_path = self.raw_data_dir / vix_file
        if asset_path.exists() and vix_path.exists():
            es_df = pd.read_csv(asset_path)
            vix_df = pd.read_csv(vix_path)
            es_df["date"] = pd.to_datetime(es_df["date"])
            vix_df["date"] = pd.to_datetime(vix_df["date"])
            vix_df = vix_df.rename(columns={
                "open": "vix_open", "high": "vix_high", "low": "vix_low", 
                "close": "vix_close", "volume": "vix_volume", "avg_hl": "vix_avg_hl"
            })
            df = es_df.merge(vix_df, on="date", how="inner")
        else:
            # Load multiple assets and merge them on date. Default set includes sp500 and vix plus vxn and nq.
            assets = ["sp500", "vix", "vxn", "nq"]
            filename_map = {
                "sp500": "sp500_df_data_daily.csv",
                "vix": "vix_2026_data_daily.csv",
                "nq": "nq_data_daily.csv",
                "vxn": "vxn_data_daily.csv",
            }

            df = None
            for asset in assets:
                path = self.raw_data_dir / filename_map.get(asset, f"{asset}_data_daily.csv")
                if not path.exists():
                    logger.warning(f"Expected raw file for {asset} not found at {path}. Skipping.")
                    continue
                tmp = pd.read_csv(path)
                tmp["date"] = pd.to_datetime(tmp["date"])
                # Rename VIX explicitly to vix_ prefix
                if asset == "vix":
                    tmp = tmp.rename(columns={
                        "open": "vix_open", "high": "vix_high", "low": "vix_low", 
                        "close": "vix_close", "volume": "vix_volume", "avg_hl": "vix_avg_hl"
                    })
                else:
                    # Prefix other asset columns to avoid collisions
                    cols = [c for c in tmp.columns if c != "date"]
                    tmp = tmp.rename(columns={c: f"{asset}_{c}" for c in cols})

                if df is None:
                    df = tmp
                else:
                    df = df.merge(tmp, on="date", how="inner")
        
        # Clean (if vix_high present)
        if "vix_high" in df.columns:
            df = self.clean_outliers_iqr(df, "vix_high")

        # Engineer
        df_final = self.engineer_features(df)
        
        # Save
        output_path = self.processed_data_dir / "processed_features.csv"
        df_final.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

        return df_final

if __name__ == "__main__":
    preparer = DataPreparer()
    preparer.run_pipeline()
