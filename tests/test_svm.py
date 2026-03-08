import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR as SklearnLinearSVR

from nlrs.linear_model.svm import LinearSVR


@pytest.fixture
def regression_data():
    X, y, coef = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_targets=1,
        bias=0.0,
        noise=0.1,
        random_state=42,
        coef=True,
    )
    X = StandardScaler().fit_transform(X)
    return X, y, coef


def test_linear_svr(regression_data):
    X, y, _ = regression_data
    epsilon = 0.1
    C = 1.0 # C relates to alpha
    # sklearn optimizes min 0.5 * ||w||^2 + C * sum(loss).
    # nlrs optimizes min sum(loss) + alpha * ||w||^2  (since it passes alpha * 2 to l2_penalty, which halves it).
    # So alpha = 0.5 / C = 1.0 / (2 * C)
    alpha = 1.0 / (2 * C)
    
    sk_model = SklearnLinearSVR(
        epsilon=epsilon,
        C=C,
        fit_intercept=True,
        intercept_scaling=1.0,
        dual=True,
        random_state=42
    )
    sk_model.fit(X, y)
    
    model = LinearSVR(
        alpha=alpha,
        epsilon=epsilon,
        penalty="l2",
        loss="epsilon_insensitive",
        fit_intercept=True,
        solver="SCS"
    )
    model.fit(X, y)
    
    # Tolerances are somewhat wide because LinearSVR solves heavily via Dual gradient descent in liblinear vs cvxpy IPMs
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-1)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-1)


def test_linearsvr_squared_loss(regression_data):
    X, y, _ = regression_data
    epsilon = 0.1
    C = 1.0
    alpha = 1.0 / (2 * C)
    
    sk_model = SklearnLinearSVR(
        epsilon=epsilon,
        C=C,
        loss="squared_epsilon_insensitive",
        fit_intercept=True,
        dual=False,
        random_state=42
    )
    sk_model.fit(X, y)
    
    model = LinearSVR(
        alpha=alpha,
        epsilon=epsilon,
        penalty="l2",
        loss="squared_epsilon_insensitive",
        fit_intercept=True,
        solver="SCS"
    )
    model.fit(X, y)
    
    np.testing.assert_allclose(model.coef_, sk_model.coef_, atol=1e-1)
    np.testing.assert_allclose(model.intercept_, sk_model.intercept_, atol=1e-1)


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "l1_l2"])
def test_linearsvr_parameters(regression_data, epsilon, loss, penalty):
    X, y, _ = regression_data
    
    # Just checking if it runs successfully with these standard combinations
    model = LinearSVR(
        alpha=0.1,
        l1_ratio=0.5 if penalty == "l1_l2" else 0.0,
        epsilon=epsilon,
        loss=loss,
        penalty=penalty,
        fit_intercept=True,
        solver="SCS"
    )
    
    try:
        model.fit(X, y)
        assert model.coef_ is not None
    except Exception as e:
        pytest.fail(
            f"LinearSVR failed with epsilon={epsilon}, loss={loss}, penalty={penalty}: {e}"
        )
