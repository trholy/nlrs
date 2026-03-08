import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from nlrs.linear_model import Lasso


@pytest.fixture
def adaptive_data():
    X, y, coef = make_regression(
        n_samples=50,
        n_features=10,
        n_informative=5,
        n_targets=1,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def test_adaptive_toggle(adaptive_data):
    X, y = adaptive_data
    alpha = 0.5

    model_base = Lasso(alpha=alpha, adaptive=False, solver="SCS")
    model_base.fit(X, y)

    model_adaptive = Lasso(alpha=alpha, adaptive=True, solver="SCS")
    model_adaptive.fit(X, y)

    # Ensure they produce different coefficients due to the varying penalty weights
    assert not np.allclose(model_base.coef_, model_adaptive.coef_, rtol=1e-2)


def test_adaptive_uniform_weights(adaptive_data):
    X, y = adaptive_data
    alpha = 0.5
    p = X.shape[1]

    # Manual weights of 1.0 everywhere should make it exactly identical to standard Lasso
    uniform_weights = np.ones(p)

    model_base = Lasso(alpha=alpha, adaptive=False, solver="SCS")
    model_base.fit(X, y)

    model_manual = Lasso(alpha=alpha, adaptive=True, adaptive_weights=uniform_weights, solver="SCS")
    model_manual.fit(X, y)

    np.testing.assert_allclose(model_base.coef_, model_manual.coef_, atol=1e-5)


def test_adaptive_p_greater_n():
    # Make data where p > n
    X, y, _ = make_regression(
        n_samples=20,
        n_features=50,
        n_informative=10,
        random_state=42,
        coef=True
    )
    X = StandardScaler().fit_transform(X)

    # lin_reg fails smoothly (provides zeros / non-unique solutions)
    # RidgeCV provides a stable set of initial weights
    model = Lasso(alpha=0.1, adaptive=True, adaptive_weights_model="cv_ridge", solver="SCS")

    # Verify it completes without divide by zero exceptions or optimization failures
    try:
        model.fit(X, y)
        assert model.coef_ is not None
        assert model.coef_.shape == (50,)
    except Exception as e:
        pytest.fail(f"Adaptive Lasso failed in p>n with error: {e}")


def test_adaptive_weights_power(adaptive_data):
    X, y = adaptive_data
    alpha = 0.5
    
    model_pow_05 = Lasso(
        alpha=alpha, 
        adaptive=True, 
        adaptive_weights_power=0.5,
        solver="SCS"
    )
    model_pow_05.fit(X, y)
    
    model_pow_20 = Lasso(
        alpha=alpha, 
        adaptive=True, 
        adaptive_weights_power=2.0,
        solver="SCS"
    )
    model_pow_20.fit(X, y)
    
    # Assert coefficients are different due to different weight powers
    assert not np.allclose(model_pow_05.coef_, model_pow_20.coef_, rtol=1e-2)
