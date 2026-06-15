import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, start_date="2005-01-01", data_dir="data/raw"):
        self.start_date = start_date
        self.end_date = datetime.now()
        self.data_dir = data_dir
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _process_yfinance_data(self, data, asset_name):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = pd.DataFrame()
            df['date'] = data.index
            df['open'] = data['Open'].values
            df['high'] = data['High'].values
            df['low'] = data['Low'].values
            df['close'] = data['Close'].values
            if 'Volume' in data.columns:
                df['volume'] = data['Volume'].values

            df = df.dropna().reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error processing data for {asset_name}: {e}")
            return None

    def download_asset_data(self, ticker, asset_name, is_volatility=False):
        logger.info(f"Downloading data for {asset_name} ({ticker})...")
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            df = self._process_yfinance_data(data, asset_name)
            
            if df is not None:
                if not is_volatility:
                    df['pct_move'] = (df['high'] - df['low']) / df['open'] * 100
                else:
                    df['avg_hl'] = (df['high'] + df['low']) / 2

                file_path = os.path.join(self.data_dir, f"{asset_name}_data_daily.csv")
                df.to_csv(file_path, index=False)
                logger.info(f"Successfully exported: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to download {asset_name}: {e}")
        return False

    def download_multiple(self, assets_config):
        """
        assets_config: list of dicts [{'ticker': '^GSPC', 'name': 'SP500', 'vol': False}, ...]
        """
        results = {}
        for asset in assets_config:
            success = self.download_asset_data(
                asset['ticker'], 
                asset['name'], 
                is_volatility=asset.get('vol', False)
            )
            results[asset['name']] = success
        return results

if __name__ == "__main__":
    # Example usage
    config = [
        {'ticker': '^GSPC', 'name': 'sp500', 'vol': False},
        {'ticker': '^VIX', 'name': 'vix', 'vol': True},
        {'ticker': '^VXN', 'name': 'vxn', 'vol': True}
    ]
    extractor = DataExtractor()
    extractor.download_multiple(config)
