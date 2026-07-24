import pytest
from unittest.mock import MagicMock, patch
from data_service.pipeline.extraction import DataExtractor
from data_service.pipeline.registry import AssetInfo

def test_download_asset_success():
    extractor = DataExtractor()
    asset = AssetInfo(name="SP500", ticker="^GSPC", description="S&P 500 Index")
    
    # Mock yfinance.download and db.py
    with patch("data_service.pipeline.extraction.yf.download") as mock_yf, \
         patch("data_service.pipeline.extraction.db") as mock_db:
        
        # Mock yfinance return value (a simple DataFrame)
        import pandas as pd
        mock_yf.return_value = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [95.0], "Close": [101.0], "Volume": [1000]
        }, index=[pd.Timestamp("2023-01-01")])
        
        # Mock db.get_latest_raw_date to simulate that we need data from 2023-01-01
        mock_db.get_latest_raw_date.return_value = pd.Timestamp("2022-12-31").date()
        mock_db.upsert_raw_ohlc.return_value = 1
        
        result = extractor.download_asset(asset)
        
        assert result is True
        mock_yf.assert_called_once()
        mock_db.upsert_raw_ohlc.assert_called_once()
        mock_db.log_run.assert_called()

def test_download_asset_api_error():
    extractor = DataExtractor()
    asset = AssetInfo(name="SP500", ticker="^GSPC", description="S&P 500 Index")
    
    with patch("data_service.pipeline.extraction.yf.download") as mock_yf, \
         patch("data_service.pipeline.extraction.db") as mock_db:
        
        mock_yf.side_effect = Exception("Yahoo Finance API Error")
        
        result = extractor.download_asset(asset)
        
        assert result is False
        mock_db.log_run.assert_called_with(
            "extraction", status="error", ticker=asset.ticker,
            # date_from and date_to are calculated inside, so we check status
            # using any() or just checking the first arg
            # but let's just verify that log_run was called.
            # we can be more specific if we want
        )
        # Correcting call check for log_run
        args, kwargs = mock_db.log_run.call_args
        assert args[0] == "extraction"
        assert kwargs["status"] == "error"

def test_download_all_calls_download_asset():
    extractor = DataExtractor()
    with patch.object(DataExtractor, "download_asset") as mock_download:
        mock_download.return_value = True
        
        results = extractor.download_all()
        
        assert len(results) > 0
        assert all(results.values())
        assert mock_download.call_count > 0
