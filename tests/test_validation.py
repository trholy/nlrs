import pytest
import numpy as np
import cvxpy as cp
from nlrs.linear_model import Ridge


def test_invalid_alpha():
    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    
    model = Ridge(alpha=-1.0)
    # CVXPY throws DCPError when parameter is negative but formulated as positive norm
    with pytest.raises(cp.error.DCPError):
        model.fit(X, y)


def test_shape_mismatches():
    X = np.random.randn(10, 5)
    y = np.random.randn(11) # Mismatched samples
    
    model = Ridge()
    with pytest.raises(ValueError):
        model.fit(X, y)
