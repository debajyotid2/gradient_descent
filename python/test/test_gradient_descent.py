"""
Unit tests for gradient_descent.py.
"""
import numpy as np
import pytest
from src.gradient_descent import forward, update_theta, gradient_descent, minibatch_sgd

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
def theta_fix() -> np.ndarray:
    return np.random.random(size=(2, 1))

def grad_fn_theta(x_arg: np.ndarray, y_arg: np.ndarray, theta_arg: np.ndarray) -> np.ndarray:
    return 2 * x_arg.T @ (x_arg @ theta_arg - y_arg)

def loss_fn(y_arg: np.ndarray, y_pred_arg: np.ndarray) -> np.ndarray:
    diff = y_arg - y_pred_arg
    return (np.dot(diff.T, diff))[0, 0] / y_arg.shape[0]

def test_forward(x_fix, theta_fix):
    x, theta = x_fix, theta_fix
    np.testing.assert_array_almost_equal(x @ theta, forward(x, theta))

def test_update_theta(x_fix, y_fix, theta_fix):
    x, y, theta = x_fix, y_fix, theta_fix
    eta = 0.001
    expected = theta - (eta/y.shape[0]) * grad_fn_theta(x, y, theta)
    np.testing.assert_array_almost_equal(expected, update_theta(x, y, theta, grad_fn_theta, eta))

def test_gradient_descent(x_fix, y_fix, theta_fix):
    x, y, theta = x_fix, y_fix, theta_fix
    eta = 0.001
    expected_theta = np.array([[-0.1651129 ],
                               [13.05738274]])
    expected_bias = np.array([-299.70244188])
    expected_final_loss = 3.9901898
    got_theta, got_bias, got_losses = gradient_descent(x, y, theta, loss_fn, grad_fn_theta, eta)
    np.testing.assert_array_almost_equal(expected_theta, got_theta)
    np.testing.assert_array_almost_equal(expected_bias, got_bias)
    np.testing.assert_almost_equal(expected_final_loss, got_losses[-1])

def test_minibatch_sgd(x_fix, y_fix, theta_fix):
    x, y, theta = x_fix, y_fix, theta_fix
    batch_size, eta = 1, 0.001
    expected_theta = np.array([[-0.064183],
                               [13.210066]])
    expected_bias = np.array([-301.090467])
    expected_final_loss = 0.012341761
    got_theta, got_bias, got_losses = minibatch_sgd(x, y, theta, batch_size, loss_fn, grad_fn_theta, eta)
    np.testing.assert_array_almost_equal(expected_theta, got_theta)
    np.testing.assert_array_almost_equal(expected_bias, got_bias)
    np.testing.assert_almost_equal(expected_final_loss, got_losses[-1])
