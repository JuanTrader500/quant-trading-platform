import pytest
import numpy as np
import pandas as pd
from src.RetrainingPipeline.validation_utils import (
    calculate_rmse, calculate_mae, calculate_directional_bias, walk_forward_splits
)

def test_calculate_rmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    # rmse = sqrt((0.1^2 + (-0.1)^2 + 0.2^2)/3) = sqrt(0.06/3) = sqrt(0.02) approx 0.1414
    assert calculate_rmse(y_true, y_pred) == pytest.approx(0.14142, abs=1e-4)

def test_calculate_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    # mae = (0.1 + 0.1 + 0.2)/3 = 0.4/3 approx 0.1333
    assert calculate_mae(y_true, y_pred) == pytest.approx(0.13333, abs=1e-4)

def test_calculate_directional_bias():
    # Caso 1: Subestimación total
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([0.5, 1.5])
    bias = calculate_directional_bias(y_true, y_pred)
    assert bias["pct_subestimado"] == 100.0
    assert bias["pct_sobreestimado"] == 0.0
    assert bias["magnitud_media_subestimacion"] == 0.5

    # Caso 2: Sobreestimación total
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    bias = calculate_directional_bias(y_true, y_pred)
    assert bias["pct_subestimado"] == 0.0
    assert bias["pct_sobreestimado"] == 100.0
    assert bias["magnitud_media_sobreestimacion"] == 0.5

def test_walk_forward_splits():
    n_samples = 100
    initial_train = 70
    step = 10
    splits = list(walk_forward_splits(n_samples, initial_train, step))
    
    # Esperamos (100-70)/10 = 3 splits
    assert len(splits) == 3
    
    # Primer split: train [0, 70), test [70, 80)
    train_0, test_0 = splits[0]
    assert len(train_0) == 70
    assert len(test_0) == 10
    assert test_0[0] == 70
    
    # Último split: train [0, 90), test [90, 100)
    train_last, test_last = splits[-1]
    assert len(train_last) == 90
    assert len(test_last) == 10
    assert test_last[-1] == 99
