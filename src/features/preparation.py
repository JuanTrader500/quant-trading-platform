import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreparer:
    def __init__(self, raw_data_dir="data/raw", processed_data_dir="data/processed"):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        if not os.path.exists(self.processed_data_dir):
            os.makedirs(self.processed_data_dir)

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
        
        # 1. Log transformation of base prices
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
        df["VIX_Vol_Diaria_lag1"] = (df["vix_close"] / (100 * np.sqrt(252))).shift(1)
        # Log range of VIX
        df["VIX_Rango_Log_lag1"] = (np.log(df["vix_high"]) - np.log(df["vix_low"])).shift(1)
        
        df["close_log_lag1"] = df["close_log"].shift(1)
        
        # Drop raw prices and temporary columns to avoid leakage
        cols_to_drop = ["vix_open", "vix_high", "vix_low", "vix_close", "vix_avg_hl", "volume", "pct_move", "vix_volume"]
        df = df.drop(columns=cols_to_drop, errors="ignore")
        
        df = df.dropna()
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df

    def run_pipeline(self, asset_file="sp500_df_data_daily.csv", vix_file="vix_2026_data_daily.csv"):
        """
        Full execution from raw CSVs to processed features.
        """
        logger.info("Starting data preparation pipeline...")
        
        # Load
        es_df = pd.read_csv(os.path.join(self.raw_data_dir, asset_file))
        vix_df = pd.read_csv(os.path.join(self.raw_data_dir, vix_file))
        
        # Rename VIX columns
        vix_df = vix_df.rename(columns={
            "open": "vix_open", "high": "vix_high", "low": "vix_low", 
            "close": "vix_close", "volume": "vix_volume", "avg_hl": "vix_avg_hl"
        })
        
        # Merge
        df = es_df.merge(vix_df, on="date", how="inner")
        
        # Clean
        df = self.clean_outliers_iqr(df, "vix_high")
        
        # Engineer
        df_final = self.engineer_features(df)
        
        # Save
        output_path = os.path.join(self.processed_data_dir, "processed_features.csv")
        df_final.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
        
        return df_final

if __name__ == "__main__":
    preparer = DataPreparer()
    preparer.run_pipeline()
