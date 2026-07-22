import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true, y_pred):
    """Calcula el Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mae(y_true, y_pred):
    """Calcula el Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))

def calculate_directional_bias(y_true, y_pred):
    """
    Calcula el sesgo direccional (RF18).
    Distingue entre subestimación (pred < real) y sobreestimación (pred > real).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_pred - y_true
    
    under = error[error < 0]
    over = error[error > 0]
    
    return {
        "bias_medio": float(error.mean()),
        "pct_subestimado": float((error < 0).mean() * 100),
        "pct_sobreestimado": float((error > 0).mean() * 100),
        "magnitud_media_subestimacion": float(-under.mean()) if len(under) > 0 else 0.0,
        "magnitud_media_sobreestimacion": float(over.mean()) if len(over) > 0 else 0.0,
    }

def walk_forward_splits(n_samples, initial_train_size, step_size):
    """
    Generador de splits para validación Walk-Forward expandida.
    """
    train_end = initial_train_size
    while train_end < n_samples:
        test_end = min(train_end + step_size, n_samples)
        yield np.arange(0, train_end), np.arange(train_end, test_end)
        train_end = test_end
