"""
Unit tests for losses.py.
"""
import numpy as np
import pytest
from src.losses import l2_grad_theta, l2_grad_with_reg, l2_loss, l2_reg, l2_reg_grad_theta

np.random.seed(234)

@pytest.fixture
def x_fix() -> np.ndarray:
    return np.array([[ 4.01232075,  2.12612775],
                 [10.4475947 ,  5.87423023],
                 [-0.1031163 , 10.46195664],
                 [ 6.9540509 ,  5.88248986],
                 [ 8.43081452,  2.28290053],
                 [ 8.6667661 ,  3.87416444],
                 [ 6.29859297,  9.65950334],
                 [ 4.51188484,  0.02989535]])

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

@pytest.fixture
def theta_fix() -> np.ndarray:
    return np.random.random(size=(2, 1))

def reg_grad_fn(theta_arg: np.ndarray) -> np.ndarray:
    return 2 * theta_arg

def test_l2_grad_theta(x_fix, y_fix, theta_fix):
    x, y, theta = x_fix, y_fix, theta_fix
    expected = 2 * x.T @ (x @ theta - y)
    np.testing.assert_array_almost_equal(expected, l2_grad_theta(x, y, theta))

def test_l2_grad_with_reg(x_fix, y_fix, theta_fix):
    x, y, theta = x_fix, y_fix, theta_fix
    lamda = 0.1
    expected = 2 * (x.T @ (x @ theta - y)) + lamda * reg_grad_fn(theta)
    np.testing.assert_array_almost_equal(expected, l2_grad_with_reg(x, y, theta, reg_grad_fn, lamda))

def test_l2_loss(y_fix, y_pred_fix):
    y, y_pred = y_fix, y_pred_fix
    diff = y - y_pred
    expected = (np.dot(diff.T, diff))[0, 0] / y.shape[0]
    np.testing.assert_array_almost_equal(expected, l2_loss(y, y_pred))

def test_l2_reg(theta_fix):
    theta = theta_fix
    expected = (theta.T @ theta)[0, 0]
    np.testing.assert_array_almost_equal(expected, l2_reg(theta))

def test_l2_reg_grad_theta(theta_fix):
    theta = theta_fix
    expected = 2 * theta
    np.testing.assert_array_almost_equal(expected, l2_reg_grad_theta(theta))
