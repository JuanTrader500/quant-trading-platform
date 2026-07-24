import pandas as pd
import pytest
from data_service.pipeline.validation import DataValidator

def test_drop_nulls():
    # Create a dataframe with some nulls in required columns
    df = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-03")],
        "open": [100.0, None, 102.0],
        "high": [105.0, 106.0, None],
        "low": [95.0, 96.0, 97.0],
        "close": [101.0, 102.0, 103.0]
    })
    
    cleaned_df = DataValidator.validate(df, "TestAsset")
    
    # Only the first row is complete
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["date"] == pd.Timestamp("2023-01-01")

def test_drop_duplicates():
    # Create a dataframe with duplicate dates
    df = pd.DataFrame({
        "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [95.0, 96.0, 97.0],
        "close": [101.0, 102.0, 103.0]
    })
    
    cleaned_df = DataValidator.validate(df, "TestAsset")
    
    # Should have 2 unique dates, keeping the last one for the duplicate
    assert len(cleaned_df) == 2
    # Check that the last occurrence of 2023-01-01 was kept (open=101.0)
    row_01_01 = cleaned_df[cleaned_df["date"] == pd.Timestamp("2023-01-01")]
    assert row_01_01.iloc[0]["open"] == 101.0

def test_empty_dataframe():
    df = pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    cleaned_df = DataValidator.validate(df, "TestAsset")
    assert cleaned_df.empty
