import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso as SklearnLasso
from nlrs.utils.selection import AdaptiveWeights

from nlrs.linear_model import Lasso
import asgl


@pytest.fixture
def regression_data():
    X, y, coef = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_targets=1,
        bias=0.0,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, coef


def test_adaptive_lasso_asgl(regression_data):
    X, y, _ = regression_data
    alpha = 0.05
    
    # 1. ASGL Baseline
    asgl_model = asgl.Regressor(model='lm', penalization='alasso', fit_intercept=False, lambda1=alpha)
    asgl_model.fit(X, y)
    
    # asgl returns a list of arrays for coef_ under the hood, usually shape (1, n_features+1). 
    # even when fit_intercept=False, index 0 is the intercept (0.0).
    asgl_coef_ = np.array(asgl_model.coef_).flatten()
    if len(asgl_coef_) > len(asgl_model.coef_):
        asgl_coef = asgl_coef_[1:] # Drop intercept
    else:
        asgl_coef = asgl_coef_
        
    # 2. Sklearn Simulation of Adaptive Lasso
    # To simulate Adaptive Lasso computationally, we divide the design matrix (X) by the weights.
    weights = AdaptiveWeights(model="lin_reg", fit_intercept=False).get_weights(X, y)
    X_scaled = X / weights
    sk_mock = SklearnLasso(alpha=alpha, fit_intercept=False)
    sk_mock.fit(X_scaled, y)
    sk_mock_coef = sk_mock.coef_ / weights
    
    # 3. nlrs Adaptive Lasso
    model = Lasso(
        alpha=alpha, 
        fit_intercept=False, 
        adaptive=True, 
        adaptive_weights_model="lin_reg",
        solver="CLARABEL"
    )
    model.fit(X, y)
    
    # Compare
    # nlrs native vs asgl (Tolerance is wide enough to account for cvxpy vs internal asgl solver differences)
    np.testing.assert_allclose(model.coef_, asgl_coef, atol=1e-1)
    
    # nlrs native vs sklearn mathematical scaling mock (Tolerance is tight since both represent pure coordinate/gradient optimizations)
    np.testing.assert_allclose(model.coef_, sk_mock_coef, atol=1e-2)

    # sklearn vs asgl
    np.testing.assert_allclose(sk_mock_coef, asgl_coef, atol=1e-1)
