import unittest
import os
import pandas as pd
import numpy as np
from src.data_pipeline.extraction import DataExtractor
from src.features.preparation import DataPreparer

class TestMLOpsPipeline(unittest.TestCase):
    def setUp(self):
        # Setup temporary paths for tests
        self.test_raw_dir = "tests/data/raw"
        self.test_proc_dir = "tests/data/processed"
        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_proc_dir, exist_ok=True)
        
        # Create dummy data
        self.asset_name = "test_asset"
        self.vix_name = "test_vix"
        
        dates = pd.date_range(start="2020-01-01", periods=100)
        df_asset = pd.DataFrame({
            'date': dates,
            'open': np.random.rand(100)*100,
            'high': np.random.rand(100)*100 + 1,
            'low': np.random.rand(100)*100 - 1,
            'close': np.random.rand(100)*100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        df_vix = pd.DataFrame({
            'date': dates,
            'open': np.random.rand(100)*20,
            'high': np.random.rand(100)*20 + 1,
            'low': np.random.rand(100)*20 - 1,
            'close': np.random.rand(100)*20,
            'volume': np.random.randint(0, 100, 100),
            'avg_hl': np.random.rand(100)*20
        })
        
        self.asset_path = os.path.join(self.test_raw_dir, f"{self.asset_name}.csv")
        self.vix_path = os.path.join(self.test_raw_dir, f"{self.vix_name}.csv")
        df_asset.to_csv(self.asset_path, index=False)
        df_vix.to_csv(self.vix_path, index=False)

    def test_data_extractor_logic(self):
        """Test if the extractor logic handles dataframe structure correctly"""
        extractor = DataExtractor(data_dir=self.test_raw_dir)
        # Test dummy data processing
        df = pd.DataFrame({'Open': [10], 'High': [11], 'Low': [9], 'Close': [10], 'Volume': [100]}, index=[pd.Timestamp('2020-01-01')])
        processed = extractor._process_yfinance_data(df, "test")
        self.assertIsNotNone(processed)
        self.assertEqual(len(processed), 1)

    def test_feature_preparation_pipeline(self):
        """Test the full preparation flow: raw -> processed"""
        preparer = DataPreparer(raw_data_dir=self.test_raw_dir, processed_data_dir=self.test_proc_dir)
        # Run pipeline with dummy files
        result = preparer.run_pipeline(asset_file=f"{self.asset_name}.csv", vix_file=f"{self.vix_name}.csv")
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('log_target', result.columns)
        self.assertIn('Upper_Wick_lag1', result.columns)
        self.assertTrue(os.path.exists(os.path.join(self.test_proc_dir, "processed_features.csv")))

    def tearDown(self):
        import shutil
        shutil.rmtree("tests/data", ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
