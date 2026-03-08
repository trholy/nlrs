import numpy as np
import pytest
from sklearn.linear_model import MultiTaskElasticNet as SklearnMultiTaskElasticNet
from sklearn.linear_model import MultiTaskLasso as SklearnMultiTaskLasso
from nlrs.linear_model import MultiTaskRegressor


@pytest.fixture
def multitask_data():
    rng = np.random.RandomState(42)
    n_samples, n_features, n_tasks = 100, 30, 40
    n_relevant_features = 5
    coef = np.zeros((n_tasks, n_features))
    times = np.linspace(0, 2 * np.pi, n_tasks)
    for k in range(n_relevant_features):
        coef[:, k] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))
    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
    return X, Y


def test_multitask_elasticnet(multitask_data):
    X, Y = multitask_data
    alpha = 1.0
    l1_ratio = 0.5
    
    sk_model = SklearnMultiTaskElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        random_state=42
    )
    sk_model.fit(X, Y)
    
    model = MultiTaskRegressor(
        loss="squared_error",
        penalty="l1_l2",
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        solver="SCS"
    )
    model.fit(X, Y)
    
    # ECOS/SCS vs CD coord descent tolerance
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_multitask_lasso(multitask_data):
    X, Y = multitask_data
    alpha = 1.0
    
    sk_model = SklearnMultiTaskLasso(
        alpha=alpha,
        fit_intercept=True,
        random_state=42
    )
    sk_model.fit(X, Y)
    
    model = MultiTaskRegressor(
        loss="squared_error",
        penalty="l1",
        alpha=alpha,
        fit_intercept=True,
        solver="SCS"
    )
    model.fit(X, Y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)
