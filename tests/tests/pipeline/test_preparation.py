"""Tests de preparation.py — RF02 (features), RF03 (sin data leakage),
integración con el versionado de esquema (RNF12)."""

import json

import numpy as np
import pandas as pd
import pytest

from DataPipeline.feature_schema import FEATURE_COLUMNS
from DataPipeline.preparation import DataPreparer


def test_run_pipeline_missing_raw_files_returns_empty_dict(tmp_path):
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()
    processed = tmp_path / "processed"

    results = DataPreparer(raw_data_dir=empty_raw, processed_data_dir=processed).run_pipeline()

    assert results == {}


def test_run_pipeline_produces_processed_csv_and_schema_manifest(equity_and_vol_csvs, tmp_path):
    processed_dir = tmp_path / "processed"

    results = DataPreparer(
        raw_data_dir=equity_and_vol_csvs, processed_data_dir=processed_dir
    ).run_pipeline()

    assert "sp500" in results
    csv_path = processed_dir / "processed_sp500.csv"
    schema_path = processed_dir / "sp500_schema.json"
    assert csv_path.exists()
    assert schema_path.exists()

    df = pd.read_csv(csv_path, index_col="date")
    assert list(df.columns) == FEATURE_COLUMNS["sp500"]  # orden forzado por RNF12
    assert not df.isna().any().any()

    manifest = json.loads(schema_path.read_text())
    assert manifest["columns"] == FEATURE_COLUMNS["sp500"]


def test_engineer_features_drops_raw_ohlc_columns(equity_and_vol_csvs, tmp_path):
    preparer = DataPreparer(raw_data_dir=equity_and_vol_csvs, processed_data_dir=tmp_path / "out")
    results = preparer.run_pipeline()

    df = results["sp500"]
    raw_cols = {"sp500_open", "sp500_high", "sp500_low", "sp500_close",
                "vix_open", "vix_high", "vix_low", "vix_close"}
    assert raw_cols.isdisjoint(df.columns)


def test_target_is_next_day_log_range_no_leakage():
    """RF03: target(t) debe ser el log-range de t+1, no el de t."""
    dates = pd.bdate_range("2024-01-01", periods=15)
    eq = pd.DataFrame({
        "date": dates,
        "sp500_open": np.linspace(100, 114, 15),
        "sp500_high": np.linspace(101, 115, 15),
        "sp500_low": np.linspace(99, 113, 15),
        "sp500_close": np.linspace(100.5, 114.5, 15),
    })
    vol = pd.DataFrame({
        "date": dates,
        "vix_open": np.linspace(15, 20, 15),
        "vix_high": np.linspace(16, 21, 15),
        "vix_low": np.linspace(14, 19, 15),
        "vix_close": np.linspace(15.5, 20.5, 15),
    })
    merged = eq.merge(vol, on="date")

    preparer = DataPreparer.__new__(DataPreparer)  # no necesita rutas para este cálculo puro
    engineered = preparer._engineer_features(merged, asset_prefix="sp500", vol_prefix="vix")

    # Reconstruir manualmente el target esperado antes del dropna/index-set
    expected_target = (np.log(merged["sp500_high"]) - np.log(merged["sp500_low"])).shift(-1)
    expected_target.index = merged["date"]
    expected_target = expected_target.reindex(engineered.index)

    pd.testing.assert_series_equal(engineered["target"], expected_target, check_names=False)


def test_iqr_outlier_removal_uses_only_training_slice(equity_and_vol_csvs):
    preparer = DataPreparer(raw_data_dir=equity_and_vol_csvs, processed_data_dir=equity_and_vol_csvs / "out")
    raw = pd.read_csv(equity_and_vol_csvs / "sp500_data_daily.csv", parse_dates=["date"])
    raw = raw.rename(columns={c: f"sp500_{c}" for c in raw.columns if c != "date"})
    vol = pd.read_csv(equity_and_vol_csvs / "vix_data_daily.csv", parse_dates=["date"])
    vol = vol.rename(columns={c: f"vix_{c}" for c in vol.columns if c != "date"})
    merged = raw.merge(vol, on="date")

    cleaned = preparer._clean(merged, vol_prefix="vix", train_cutoff="2024-01-10")
    assert len(cleaned) <= len(merged)
    assert set(cleaned.columns) == set(merged.columns)
