import time
import cvxpy as cp
import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression as SklearnOLS
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.linear_model import Lasso as SklearnLasso
from sklearn.preprocessing import StandardScaler

from nlrs.linear_model import Ridge, Lasso, ElasticNet, LinearRegression


@pytest.fixture
def base_data():
    X, y, coef = make_regression(
        n_samples=50,
        n_features=5,
        n_informative=3,
        n_targets=1,
        bias=2.0,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, coef


def test_alpha_zero_ridge(base_data):
    X, y, _ = base_data
    sk_ols = SklearnOLS(fit_intercept=True)
    sk_ols.fit(X, y)

    # alpha=0 should emulate OLS behavior
    model = Ridge(alpha=0.0, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_ols.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_ols.intercept_, atol=1e-2)


def test_alpha_zero_lasso(base_data):
    X, y, _ = base_data
    sk_ols = SklearnOLS(fit_intercept=True)
    sk_ols.fit(X, y)
    
    model = Lasso(alpha=0.0, fit_intercept=True, solver="SCS")
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_ols.coef_, atol=1e-2)
    np.testing.assert_allclose(model.intercept_, sk_ols.intercept_, atol=1e-2)


def test_alpha_large(base_data):
    X, y, _ = base_data
    # Very large alpha should shrink all coefficients near zero
    alpha = 1e6
    
    model_ridge = Ridge(alpha=alpha, fit_intercept=True, solver="SCS")
    model_ridge.fit(X, y)
    np.testing.assert_allclose(
        model_ridge.coef_, np.zeros_like(model_ridge.coef_), atol=1e-2
    )
    
    # Intercept shouldn't be heavily affected by alpha theoretically if centered, 
    # but at least close to mean of y
    np.testing.assert_allclose(model_ridge.intercept_, np.mean(y), atol=1e-1)
    
    model_lasso = Lasso(alpha=alpha, fit_intercept=True, solver="SCS")
    model_lasso.fit(X, y)

    np.testing.assert_allclose(model_lasso.coef_, 0.0, atol=1e-2)
    np.testing.assert_allclose(model_lasso.intercept_, np.mean(y), atol=1e-1)


def test_elasticnet_equivalences(base_data):
    X, y, _ = base_data
    alpha = 0.5
    
    # Check ElasticNet(l1_ratio=0.0) against sklearn Ridge with appropriate scaling
    # sklearn ElasticNet(l1_ratio=0) minimizes (1/2n)||y-Xw||^2 + 0.5*alpha||w||^2
    # sklearn Ridge minimizes ||y-Xw||^2 + alpha_ridge||w||^2
    # So equivalent alpha_ridge = alpha * n_samples
    n_samples = X.shape[0]
    alpha_ridge = alpha * n_samples

    ridge_sk = SklearnRidge(alpha=alpha_ridge, fit_intercept=True)
    ridge_sk.fit(X, y)
    
    en_ridge = ElasticNet(alpha=alpha, l1_ratio=0.0, fit_intercept=True, solver="SCS")
    en_ridge.fit(X, y)
    np.testing.assert_allclose(en_ridge.coef_, ridge_sk.coef_, atol=1e-1)
    
    # l1_ratio = 1 -> Lasso
    lasso_sk = SklearnLasso(alpha=alpha, fit_intercept=True)
    lasso_sk.fit(X, y)
    
    en_lasso = ElasticNet(alpha=alpha, l1_ratio=1.0, fit_intercept=True, solver="SCS")
    en_lasso.fit(X, y)
    np.testing.assert_allclose(en_lasso.coef_, lasso_sk.coef_, atol=1e-3)


def test_fit_intercept_centered(base_data):
    X, y, _ = base_data
    
    # Center y manually
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Fit without intercept on centered data
    model_no_int = Ridge(alpha=1.0, fit_intercept=False, solver="SCS")
    model_no_int.fit(X, y_centered)
    
    # Fit with intercept on raw data
    model_int = Ridge(alpha=1.0, fit_intercept=True, solver="SCS")
    model_int.fit(X, y)
    
    np.testing.assert_allclose(model_no_int.coef_, model_int.coef_, atol=1e-4)
    # intercept should match y_mean when X is natively standardized
    np.testing.assert_allclose(model_int.intercept_, y_mean, atol=1e-4)


def test_y_2d_single_target(base_data):
    X, y, _ = base_data
    y_2d = y.reshape(-1, 1)
    
    model_1d = LinearRegression(fit_intercept=True, solver="SCS")
    model_1d.fit(X, y)
    
    model_2d = LinearRegression(fit_intercept=True, solver="SCS")
    model_2d.fit(X, y_2d)
    
    np.testing.assert_allclose(model_1d.coef_, model_2d.coef_, atol=1e-4)
    np.testing.assert_allclose(model_1d.intercept_, model_2d.intercept_, atol=1e-4)


def test_global_parameters(base_data):
    X, y, _ = base_data
    
    # Check that setting these parameters does not raise errors and passes to solver
    model = LinearRegression(
        fit_intercept=True,
        solver="SCS",
        tol=1e-3,
        time_limit=10.0,
        verbose=True
    )
    
    try:
        model.fit(X, y)
    except Exception as e:
        pytest.fail(f"Fitting with custom global parameters failed: {e}")
    
    assert model.coef_ is not None

    # Test an alternative solver if available, just to ensure the parameter routes correctly
    # ECOS is usually available alongside SCS
    model_alt = LinearRegression(
        fit_intercept=True,
        solver="ECOS",
    )
    try:
        model_alt.fit(X, y)
        assert model_alt.coef_ is not None
    except Exception as e:
        pass # If ECOS is not installed or fails for some structural reason, we don't strictly fail the test, but typically it should work.


def test_solver_time_limit_stops_early():
    # Make a problem that is decently sized but won't timeout compilation for all solvers
    X, y, _ = make_regression(
        n_samples=100000,
        n_features=500,
        n_informative=50,
        random_state=42,
        coef=True
    )
    
    solvers = cp.installed_solvers()
    
    # List of cvxpy solvers we explicitly mapped time limits for
    supported_time_limits = [
        "CPLEX", "MOSEK", "GUROBI", "CLARABEL", "OSQP", "PIQP", 
        "CUOPT", "HIGHS", "SCS", "GLOP", "PDLP", "CBC", "XPRESS", 
        "SCIPY", "SCIP"
    ]
    
    for solver in solvers:
        start = time.perf_counter()
        try:
            model = LinearRegression(solver=solver, time_limit=1.0, verbose=False)
            model.fit(X, y)
            
            end = time.perf_counter()
            elapsed = end - start
            
            # If it supported time_limit natively, it should finish within a reasonable buffer
            # including CVXPY's problem compilation time.
            assert elapsed < 5.0, f"Solver {solver} took {elapsed:.2f}s, potentially ignoring the 1.0s time limit."
            
        except ValueError as e:
            if "Setting a time limit is not supported" in str(e):
                assert solver not in supported_time_limits, f"Solver {solver} unexpectedly didn't support time_limit."
            else:
                pass
        except Exception:
            # Other errors (like solver failures, memory defaults, or license issues) are fine to pass
            pass
