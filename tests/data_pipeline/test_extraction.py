"""
Tests de extraction.py — RF01 (descarga), RF05 (update incremental),
RF06 (logging de errores de conexión).

`yfinance.download` siempre se mockea: los tests de CI no deben
depender de la red ni de la disponibilidad de la API externa.
"""

import pandas as pd
import pytest

from DataPipeline.extraction import DataExtractor


def _fake_yf_response(dates) -> pd.DataFrame:
    """Emula el DataFrame que retorna yf.download (índice = fechas)."""
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(len(dates))],
            "High": [101.0 + i for i in range(len(dates))],
            "Low": [99.0 + i for i in range(len(dates))],
            "Close": [100.5 + i for i in range(len(dates))],
            "Volume": [1000 + i for i in range(len(dates))],
        },
        index=pd.DatetimeIndex(dates),
    )


def test_download_asset_success_persists_csv(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=5)
    monkeypatch.setattr(
        "DataPipeline.extraction.yf.download",
        lambda *a, **k: _fake_yf_response(dates),
    )

    extractor = DataExtractor(start_date="2024-01-01", data_dir=tmp_path)
    ok = extractor.download_asset(ticker="^GSPC", asset_name="sp500", is_volatility=False)

    assert ok is True
    out_file = tmp_path / "sp500_data_daily.csv"
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 5
    assert "pct_move" in df.columns  # is_volatility=False → agrega pct_move


def test_download_asset_volatility_adds_avg_hl(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=3)
    monkeypatch.setattr(
        "DataPipeline.extraction.yf.download",
        lambda *a, **k: _fake_yf_response(dates),
    )

    extractor = DataExtractor(start_date="2024-01-01", data_dir=tmp_path)
    extractor.download_asset(ticker="^VIX", asset_name="vix", is_volatility=True)

    df = pd.read_csv(tmp_path / "vix_data_daily.csv")
    assert "avg_hl" in df.columns
    assert "pct_move" not in df.columns


def test_download_asset_connection_error_returns_false(tmp_path, monkeypatch):
    def _raise(*a, **k):
        raise ConnectionError("Yahoo Finance no responde")

    monkeypatch.setattr("DataPipeline.extraction.yf.download", _raise)

    extractor = DataExtractor(start_date="2024-01-01", data_dir=tmp_path)
    ok = extractor.download_asset(ticker="^GSPC", asset_name="sp500")

    assert ok is False
    assert not (tmp_path / "sp500_data_daily.csv").exists()


def test_download_asset_incremental_update_preserves_old_dates(tmp_path, monkeypatch):
    # Estado inicial en disco: incluye una fecha que la API "ya no devuelve"
    existing = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-29", "2024-01-01"]),
        "open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2],
        "pct_move": [0.0, 0.0],
    })
    existing.to_csv(tmp_path / "sp500_data_daily.csv", index=False)

    new_dates = pd.bdate_range("2024-01-01", periods=2)  # no incluye 2023-12-29
    monkeypatch.setattr(
        "DataPipeline.extraction.yf.download",
        lambda *a, **k: _fake_yf_response(new_dates),
    )

    extractor = DataExtractor(start_date="2024-01-01", data_dir=tmp_path)
    extractor.download_asset(ticker="^GSPC", asset_name="sp500")

    result = pd.read_csv(tmp_path / "sp500_data_daily.csv", parse_dates=["date"])
    assert pd.Timestamp("2023-12-29") in set(result["date"])  # preservada
    assert len(result) == 3  # 1 preservada + 2 nuevas


def test_download_all_reports_per_asset_status(tmp_path, monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=2)
    monkeypatch.setattr(
        "DataPipeline.extraction.yf.download",
        lambda ticker, **k: _fake_yf_response(dates) if ticker == "^GSPC" else (_ for _ in ()).throw(ConnectionError()),
    )

    extractor = DataExtractor(start_date="2024-01-01", data_dir=tmp_path)
    config = [
        {"ticker": "^GSPC", "name": "sp500", "vol": False},
        {"ticker": "^VIX", "name": "vix", "vol": True},
    ]
    results = extractor.download_all(config)

    assert results == {"sp500": True, "vix": False}


def test_load_config_reads_yaml(tmp_path):
    config_file = tmp_path / "assets.yaml"
    config_file.write_text(
        "assets:\n  - ticker: \"^GSPC\"\n    name: \"sp500\"\n    vol: false\n"
    )
    assets = DataExtractor.load_config(config_file)
    assert assets == [{"ticker": "^GSPC", "name": "sp500", "vol": False}]
