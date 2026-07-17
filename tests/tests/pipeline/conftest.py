"""
conftest.py
-----------
Fixtures compartidas por los tests del DataPipeline.

Agrega src/ al sys.path (los tests viven en tests/pipeline/, el código
en src/DataPipeline/) y provee generadores de datos OHLC sintéticos
para no depender de la API de Yahoo Finance ni de datos reales en CI.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _synthetic_ohlc(n_rows: int, start: str, seed: int, base_price: float) -> pd.DataFrame:
    """Genera un DataFrame OHLC determinístico (random walk positivo)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_rows)
    close = base_price + np.cumsum(rng.normal(0, 1, n_rows))
    close = np.abs(close) + base_price / 2  # nunca <= 0

    open_ = close + rng.normal(0, 0.5, n_rows)
    open_ = np.abs(open_) + 1
    high = np.maximum(open_, close) + np.abs(rng.normal(0.5, 0.3, n_rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.5, 0.3, n_rows))
    low = np.maximum(low, 0.1)

    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


@pytest.fixture
def make_ohlc_df():
    return _synthetic_ohlc


@pytest.fixture
def equity_and_vol_csvs(tmp_path, make_ohlc_df):
    """Crea CSVs crudos de equity + volatilidad ya persistidos en disco,
    suficientes en longitud para superar los rolling windows (10 días)
    de preparation.py y dejar filas no-NaN tras el dropna."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    equity = make_ohlc_df(n_rows=40, start="2024-01-02", seed=1, base_price=4500)
    vol = make_ohlc_df(n_rows=40, start="2024-01-02", seed=2, base_price=18)

    equity_path = raw_dir / "sp500_data_daily.csv"
    vol_path = raw_dir / "vix_data_daily.csv"
    equity.to_csv(equity_path, index=False)
    vol.to_csv(vol_path, index=False)

    return raw_dir
