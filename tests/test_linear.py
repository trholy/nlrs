import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnOLS
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.preprocessing import StandardScaler

from nlrs.linear_model import LinearRegression as OLS
from nlrs.linear_model import Ridge as Ridge
from nlrs.linear_model import Lasso as Lasso
from nlrs.linear_model import ElasticNet as ElasticNet


@pytest.fixture
def regression_data():
    X, y, coef = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=10,
        n_targets=1,
        bias=0.0,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, coef


def test_linear_regression(regression_data):
    X, y, _ = regression_data
    
    sk_model = SklearnOLS(fit_intercept=True)
    sk_model.fit(X, y)
    
    model = OLS(fit_intercept=True, solver="SCS") # ECOS occasionally has issues with completely unregularized problems, but SCS is usually fine
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_ridge_regression(regression_data):
    X, y, _ = regression_data
    
    alpha = 1.0
    sk_model = SklearnRidge(alpha=alpha, fit_intercept=True)
    sk_model.fit(X, y)
    
    model = Ridge(alpha=alpha, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_lasso_regression(regression_data):
    X, y, _ = regression_data
    
    alpha = 1.0
    sk_model = SklearnLasso(alpha=alpha, fit_intercept=True)
    sk_model.fit(X, y)
    
    model = Lasso(alpha=alpha, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_elasticnet_regression(regression_data):
    X, y, _ = regression_data
    
    alpha = 1.0
    l1_ratio = 0.5
    sk_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    sk_model.fit(X, y)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_lasso_positive_constraint(regression_data):
    X, y, _ = regression_data
    
    alpha = 0.5
    sk_model = SklearnLasso(alpha=alpha, fit_intercept=True, positive=True, random_state=42)
    sk_model.fit(X, y)
    
    model = Lasso(alpha=alpha, fit_intercept=True, positive=True, solver="SCS")
    model.fit(X, y)
    
    assert np.all(model.coef_ >= -1e-5)
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    if sk_model.fit_intercept:
        np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)


def test_ridge_no_intercept(regression_data):
    X, y, _ = regression_data
    X_shifted = X + 5.0 
    
    alpha = 1.0
    sk_model = SklearnRidge(alpha=alpha, fit_intercept=False, random_state=42)
    sk_model.fit(X_shifted, y)
    
    model = Ridge(alpha=alpha, fit_intercept=False, solver="SCS")
    model.fit(X_shifted, y)
    
    assert model.intercept_ is None or model.intercept_ == 0.0
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)


@pytest.mark.parametrize("l1_ratio", [0.1, 0.3, 0.7, 0.9])
def test_elasticnet_l1_ratio_sweep(regression_data, l1_ratio):
    X, y, _ = regression_data
    alpha = 0.5
    
    sk_model = SklearnElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
    sk_model.fit(X, y)
    
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-2)
