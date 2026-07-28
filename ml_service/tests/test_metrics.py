import numpy as np

from models.metrics import evaluate


def test_perfect_predictions_have_zero_error():
    y = np.array([0.01, 0.02, 0.015])
    result = evaluate(y, y)
    assert result.rmse == 0
    assert result.mae == 0
    assert result.directional_bias == 0


def test_overestimation_is_detected():
    y_true = np.array([0.01, 0.01, 0.01])
    y_pred = np.array([0.02, 0.02, 0.02])
    result = evaluate(y_true, y_pred)
    assert result.directional_bias > 0
    assert result.overestimation_rate == 1.0
    assert result.underestimation_rate == 0.0


def test_underestimation_is_detected():
    y_true = np.array([0.02, 0.02, 0.02])
    y_pred = np.array([0.01, 0.01, 0.01])
    result = evaluate(y_true, y_pred)
    assert result.directional_bias < 0
    assert result.underestimation_rate == 1.0
