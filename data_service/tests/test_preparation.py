import pandas as pd
import numpy as np
import pytest
from data_service.pipeline.preparation import DataPreparer

def test_engineer_features_math():
    # Create a synthetic dataset
    # 15 days to ensure rolling windows (5d, 10d) are filled
    dates = pd.date_range(start="2023-01-01", periods=15)
    df = pd.DataFrame({
        "date": dates,
        "idx_open": [100 + i for i in range(15)],
        "idx_high": [105 + i for i in range(15)],
        "idx_low": [95 + i for i in range(15)],
        "idx_close": [102 + i for i in range(15)],
        "vol_open": [20 + i for i in range(15)],
        "vol_high": [25 + i for i in range(15)],
        "vol_low": [15 + i for i in range(15)],
        "vol_close": [22 + i for i in range(15)],
    })

    features_df = DataPreparer._engineer_features(df)

    # 1. Check target_range_next_day: (log(high_t+1) - log(low_t+1))
    first_date = features_df.index[0]
    first_idx = df.index[df["date"] == first_date][0]
    expected_target = np.log(df.loc[first_idx + 1, "idx_high"]) - np.log(df.loc[first_idx + 1, "idx_low"])
    assert np.isclose(features_df["target_range_next_day"].iloc[0], expected_target)

    # 2. Check main_log_return: log(close_t) - log(close_t-1)
    second_date = features_df.index[1]
    second_idx = df.index[df["date"] == second_date][0]
    expected_return = np.log(df.loc[second_idx, "idx_close"]) - np.log(df.loc[second_idx - 1, "idx_close"])
    assert np.isclose(features_df["main_log_return"].iloc[1], expected_return)

    # 3. Check target for the last row is NaN (RF03)
    assert np.isnan(features_df["target_range_next_day"].iloc[-1])

    # 4. Check day_of_week
    # 2023-01-01 is Sunday (6), but since we drop initial rows due to rolling, 
    # the first index will be 10th day.
    assert "day_of_week" in features_df.columns

def test_engineer_features_invalid_data():
    # DataFrame with negative values (should be filtered out by the positive_cols check)
    df = pd.DataFrame({
        "date": pd.date_range(start="2023-01-01", periods=2),
        "idx_open": [100, -100],
        "idx_high": [105, 105],
        "idx_low": [95, 95],
        "idx_close": [102, 102],
        "vol_high": [25, 25],
        "vol_low": [15, 15],
        "vol_close": [22, 22],
    })
    
    features_df = DataPreparer._engineer_features(df)
    # The row with -100 should be removed
    assert len(features_df) < 2
