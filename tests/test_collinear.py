import numpy as np
from nlrs.linear_model import Ridge, LinearRegression


def test_collinear_features():
    rng = np.random.RandomState(42)
    X_base = rng.randn(50, 2)
    # create third col identical to col 1
    X_collinear = np.column_stack((X_base, X_base[:, 0]))
    y = X_base[:, 0] * 3.0 + X_base[:, 1] * 1.5 + rng.randn(50) * 0.1
    
    # Ridge strictly assigns roughly equal weight to collinear features due to l2 penalty strictly convex
    ridge = Ridge(alpha=1.0, solver="SCS")
    ridge.fit(X_collinear, y)
    
    # Feature 0 and 2 should have same coefficient
    np.testing.assert_allclose(ridge.coef_[0], ridge.coef_[2], rtol=1e-2)
    
    # OLS coefficients for collinear might be ambiguous and depend on solver's underlying least-squares min norm routine, 
    # but the summation should equal the true effect
    ols = LinearRegression(solver="SCS")
    ols.fit(X_collinear, y)
    
    # The sum of their coefficients should track roughly the true `3.0` feature effect
    assert abs(abs(ols.coef_[0] + ols.coef_[2]) - 3.0) < 0.2
