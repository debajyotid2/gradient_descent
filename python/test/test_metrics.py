"""
Unit tests for metrics.py.
"""

import pytest
import numpy as np
from src.metrics import mse, mae, r2_score

@pytest.fixture
def y_fix() -> np.ndarray:
    return np.array([[-273.51853194],
                     [-223.31372354],
                     [-162.879998  ],
                     [-219.93701874],
                     [-273.19760952],
                     [-252.06560385],
                     [-176.92895107],
                     [-299.11200384]])

@pytest.fixture
def y_pred_fix() -> np.ndarray:
    return np.array([[-272.60326403],
                     [-224.72540201],
                     [-163.07964395],
                     [-224.0407238 ],
                     [-271.2857721 ],
                     [-250.54698885],
                     [-174.61458858],
                     [-300.0570572 ]])

def test_mse(y_fix, y_pred_fix):
    y, y_pred = y_fix, y_pred_fix
    diff = y - y_pred
    expected = (diff.T @ diff)[0, 0] / y.shape[0]
    assert expected == mse(y, y_pred)

def test_mae(y_fix, y_pred_fix):
    y, y_pred = y_fix, y_pred_fix
    expected = np.mean(np.abs(y - y_pred))
    assert expected == mae(y, y_pred)

def test_r2_score(y_fix, y_pred_fix):
    y, y_pred = y_fix, y_pred_fix
    y_mean = np.mean(y)
    diff = y - y_pred
    ss_res = (diff.T @ diff)[0, 0]
    ss_tot = ((y - y_mean).T @ (y - y_mean))[0, 0]
    expected = 1.0 - ss_res / ss_tot
    assert expected == r2_score(y, y_pred)
