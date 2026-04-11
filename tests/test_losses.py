import cvxpy as cp
import numpy as np
import pytest
import scipy.special
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_pinball_loss

from nlrs.objectives.losses import (
    epsilon_insensitive,
    get_loss_expr,
    huber,
    quantile,
    squared_epsilon_insensitive,
    squared_error,
)


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 6)
    y = X @ np.array([1.5, -2.2, 0.8, -0.4, 3.1, -1.0]) + rng.randn(30) * 0.2

    beta = cp.Variable(6)
    beta.value = np.array([0.3, -1.2, 0.7, -0.2, 2.5, -0.7])

    intercept = cp.Variable()
    intercept.value = 0.35
    return X, y, beta, intercept


def _preds(X, beta, intercept=None):
    preds = X @ beta.value
    if intercept is not None:
        preds = preds + intercept.value
    return preds


@pytest.mark.parametrize("use_intercept", [False, True])
def test_squared_error_matches_numpy(regression_data, use_intercept):
    X, y, beta, intercept = regression_data
    intercept_var = intercept if use_intercept else None

    expr = squared_error(X, y, beta, intercept=intercept_var)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept_var)
    np_val = 0.5 * np.mean((preds - y) ** 2)

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("use_intercept", [False, True])
def test_squared_error_matches_numpy_via_factory(regression_data, use_intercept):
    X, y, beta, intercept = regression_data
    intercept_var = intercept if use_intercept else None

    expr = get_loss_expr("squared_error", X, y, beta, intercept=intercept_var)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept_var)
    np_val = 0.5 * np.mean((preds - y) ** 2)

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("use_intercept", [False, True])
def test_sum_squares_matches_numpy_via_factory(regression_data, use_intercept):
    X, y, beta, intercept = regression_data
    intercept_var = intercept if use_intercept else None

    expr = get_loss_expr("sum_squares", X, y, beta, intercept=intercept_var)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept_var)
    np_val = np.sum((preds - y) ** 2)

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5])
def test_epsilon_insensitive_matches_numpy(regression_data, epsilon):
    X, y, beta, intercept = regression_data

    expr = epsilon_insensitive(X, y, beta, epsilon=epsilon, intercept=intercept)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept)
    np_val = np.mean(np.maximum(np.abs(preds - y) - epsilon, 0.0))

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5])
def test_squared_epsilon_insensitive_matches_numpy(regression_data, epsilon):
    X, y, beta, intercept = regression_data

    expr = squared_epsilon_insensitive(X, y, beta, epsilon=epsilon, intercept=intercept)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept)
    np_val = np.mean(np.maximum(np.abs(preds - y) - epsilon, 0.0) ** 2)

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("M", [0.25, 1.0, 2.0])
def test_huber_matches_scipy(regression_data, M):
    X, y, beta, intercept = regression_data

    expr = huber(X, y, beta, M=M, intercept=intercept)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept)
    residuals = y - preds
    np_val = np.mean(scipy.special.huber(M, residuals))

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_quantile_matches_sklearn_pinball(regression_data, q):
    X, y, beta, intercept = regression_data

    expr = quantile(X, y, beta, q=q, intercept=intercept)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept)
    np_val = mean_pinball_loss(y, preds, alpha=q)

    np.testing.assert_allclose(cvx_val, np_val)


@pytest.mark.parametrize("loss_name", ["mae", "mean_absolute_error"])
def test_mae_matches_sklearn_and_quantile(loss_name, regression_data):
    X, y, beta, intercept = regression_data

    expr = get_loss_expr(loss_name, X, y, beta, intercept=intercept)
    cvx_val = expr.value

    preds = _preds(X, beta, intercept)
    sk_val = sk_mean_absolute_error(y, preds)

    q50 = quantile(X, y, beta, q=0.5, intercept=intercept).value
    np.testing.assert_allclose(cvx_val, sk_val)
    np.testing.assert_allclose(cvx_val, 2 * q50)


@pytest.mark.parametrize(
    "loss_name,kwargs",
    [
        ("epsilon_insensitive", {"epsilon": 0.2}),
        ("squared_epsilon_insensitive", {"epsilon": 0.2}),
        ("huber", {"M": 1.3}),
        ("quantile", {"q": 0.8}),
        ("mae", {}),
    ],
)
def test_loss_factory_multiply_by_n(loss_name, kwargs, regression_data):
    X, y, beta, intercept = regression_data
    n = X.shape[0]

    base = get_loss_expr(loss_name, X, y, beta, intercept=intercept, **kwargs).value
    scaled = get_loss_expr(
        loss_name,
        X,
        y,
        beta,
        intercept=intercept,
        multiply_by_n=True,
        **kwargs,
    ).value

    np.testing.assert_allclose(scaled, n * base)


@pytest.mark.parametrize("invalid_q", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_quantile(regression_data, invalid_q):
    X, y, beta, _ = regression_data
    with pytest.raises(ValueError):
        quantile(X, y, beta, q=invalid_q)


@pytest.mark.parametrize("invalid_eps", [-1.0, -0.1, np.inf, np.nan])
def test_invalid_epsilon(regression_data, invalid_eps):
    X, y, beta, _ = regression_data
    with pytest.raises(ValueError, match="non-negative"):
        epsilon_insensitive(X, y, beta, epsilon=invalid_eps)
    with pytest.raises(ValueError, match="non-negative"):
        squared_epsilon_insensitive(X, y, beta, epsilon=invalid_eps)


def test_unknown_loss_name(regression_data):
    X, y, beta, intercept = regression_data
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss_expr("not_a_real_loss", X, y, beta, intercept=intercept)
