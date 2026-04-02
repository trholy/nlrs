import numpy as np
import cvxpy as cp
import scipy.special
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error, mean_pinball_loss
import pytest
from nlrs.objectives.losses import (
    epsilon_insensitive,
    squared_epsilon_insensitive,
    huber,
    quantile,
    mean_absolute_error as nlrs_mean_absolute_error
)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = X @ np.array([1.0, -2.0, 3.0, -4.0, 5.0]) + np.random.randn(20) * 0.1
    beta = cp.Variable(5)
    beta.value = np.array([0.5, -1.0, 1.5, -2.0, 2.5])
    return X, y, beta


def test_huber_loss(regression_data):
    X, y, beta = regression_data
    M = 1.0
    
    expr = huber(X, y, beta, M=M)
    cvx_val = expr.value
    
    preds = X @ beta.value
    res = y - preds
    
    # Using scipy.special.huber
    np_val = np.mean(scipy.special.huber(M, res))

    np.testing.assert_allclose(cvx_val, np_val)


def test_quantile_loss(regression_data):
    X, y, beta = regression_data
    q = 0.75
    
    expr = quantile(X, y, beta, q=q)
    cvx_val = expr.value
    
    preds = X @ beta.value
    np_val = mean_pinball_loss(y, preds, alpha=q)
    
    np.testing.assert_allclose(cvx_val, np_val)


def test_mae_loss(regression_data):
    X, y, beta = regression_data
    
    expr = nlrs_mean_absolute_error(X, y, beta)
    cvx_val = expr.value
    
    preds = X @ beta.value
    np_val = sk_mean_absolute_error(y, preds)
    
    np.testing.assert_allclose(cvx_val, np_val)
    
    # Test that mae is exactly equal to 2 * quantile(q=0.5)
    expr_q = quantile(X, y, beta, q=0.5)
    np.testing.assert_allclose(cvx_val, 2 * expr_q.value)


def test_invalid_quantile(regression_data):
    X, y, beta = regression_data
    with pytest.raises(ValueError):
        quantile(X, y, beta, q=-0.1)
    
    with pytest.raises(ValueError):
        quantile(X, y, beta, q=1.5)


def test_invalid_epsilon(regression_data):
    X, y, beta = regression_data
    with pytest.raises(ValueError, match="non-negative"):
        epsilon_insensitive(X, y, beta, epsilon=-0.1)
    with pytest.raises(ValueError, match="non-negative"):
        squared_epsilon_insensitive(X, y, beta, epsilon=-0.1)
