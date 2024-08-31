"""
Unit tests for helpers.py.
"""

import numpy as np
from src.helpers import make_regression_dataset, split_into_train_test

def test_make_regression_dataset():
    n_samples, n_features, bias, noise_intensity = 100, 5, 123.33, 1.2
    x, y = make_regression_dataset(n_samples, n_features, bias, noise_intensity)
    assert x.shape == (n_samples, n_features)
    assert y.shape == (n_samples, 1)

def test_split_into_train_test():
    num_idxs = 100
    idxs = np.arange(num_idxs)
    test_frac = 0.2
    train_idxs, test_idxs = split_into_train_test(idxs, test_frac)
    assert test_idxs.shape[0] == int(test_frac * num_idxs)
    assert len(train_idxs) + len(test_idxs) == num_idxs
