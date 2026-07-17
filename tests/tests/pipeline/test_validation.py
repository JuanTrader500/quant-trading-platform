"""Tests de DataValidator — RF04: integridad de datos crudos."""

import pandas as pd

from DataPipeline.validation import DataValidator


def test_drop_nulls_removes_incomplete_rows():
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": [1, 2, None, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close": [1, 2, 3, 4, 5],
    })
    cleaned = DataValidator._drop_nulls(df, "test_asset")
    assert len(cleaned) == 4
    assert cleaned["open"].isna().sum() == 0


def test_drop_duplicates_keeps_last_occurrence():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "open": [1, 999, 2],
        "high": [1, 999, 2],
        "low": [1, 999, 2],
        "close": [1, 999, 2],
    })
    cleaned = DataValidator._drop_duplicates(df, "test_asset")
    assert len(cleaned) == 2
    kept = cleaned.loc[cleaned["date"] == "2024-01-01", "open"].iloc[0]
    assert kept == 999  # keep="last"


def test_report_date_gaps_does_not_mutate_dataframe():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),  # faltan días hábiles entre medio
        "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2],
    })
    before = df.copy()
    DataValidator._report_date_gaps(df, "test_asset")
    pd.testing.assert_frame_equal(df, before)


def test_validate_runs_full_pipeline_and_returns_clean_df():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]),
        "open": [1, 1, None, 3],
        "high": [1, 1, 2, 3],
        "low": [1, 1, 2, 3],
        "close": [1, 1, 2, 3],
    })
    cleaned = DataValidator.validate(df, "test_asset")
    # fila duplicada colapsa a 1, fila con null se elimina → quedan 2
    assert len(cleaned) == 2
    assert cleaned["open"].isna().sum() == 0
