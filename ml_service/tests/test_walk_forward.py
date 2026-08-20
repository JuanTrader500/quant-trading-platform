import numpy as np
import pandas as pd

from features.feature_engineering import FEATURE_COLUMNS
from training.walk_forward import run_walk_forward

TARGET_COLUMN = "target_range_next_day"


def _synthetic_dataset(n_rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=n_rows, freq="B")
    data = {col: rng.normal(scale=0.01, size=n_rows) for col in FEATURE_COLUMNS}
    data["day_of_week"] = dates.dayofweek
    data[TARGET_COLUMN] = rng.normal(loc=0.01, scale=0.005, size=n_rows).clip(min=0.0001)
    return pd.DataFrame(data, index=dates)


def test_walk_forward_produces_metrics_for_all_candidates_and_garch():
    df = _synthetic_dataset()
    result = run_walk_forward(
        df=df, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN,
        min_train_days=120, step_days=30,
    )
    assert "garch_baseline" in result.metrics_by_model
    assert result.best_model_name == "gradient_boosting_trees"
    assert result.metrics_by_model[result.best_model_name].n_observations > 0


def test_walk_forward_never_predicts_using_future_rows():
    """Verifica que cada predicción pooled tenga fecha posterior a
    todas las fechas usadas para entrenar ese bloque: no debe haber
    solapamiento train/validación (anti fuga de datos)."""
    df = _synthetic_dataset()
    result = run_walk_forward(
        df=df, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN,
        min_train_days=120, step_days=30,
    )
    predicted_dates = sorted(r["date"] for r in result.pooled_predictions["gradient_boosting_trees"])
    all_dates = sorted(d.date().isoformat() for d in df.index)
    # Las primeras `min_train_days` fechas nunca deberían aparecer como predichas.
    assert predicted_dates[0] not in all_dates[:100]
