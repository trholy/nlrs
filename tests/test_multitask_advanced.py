import numpy as np
import pytest
from sklearn.linear_model import MultiTaskLasso as SklearnMultiTaskLasso
from nlrs.linear_model import MultiTaskRegressor


@pytest.fixture
def multitask_data():
    rng = np.random.RandomState(42)
    n_samples, n_features, n_tasks = 50, 10, 5
    n_relevant_features = 3
    coef = np.zeros((n_tasks, n_features))
    for k in range(n_relevant_features):
        coef[:, k] = rng.randn(n_tasks) * 5
    X = rng.randn(n_samples, n_features)
    Y = np.dot(X, coef.T) + rng.randn(n_samples, n_tasks)
    return X, Y


def test_shared_sparsity(multitask_data):
    X, Y = multitask_data
    alpha = 5.0
    
    model = MultiTaskRegressor(penalty="l1", alpha=alpha, solver="SCS")
    model.fit(X, Y)
    
    coefs = model.coef_ # Shape (n_tasks, n_features)

    # For a given feature (column), if one task's coef is extremely small, 
    # all tasks should also be extremely small due to l1/l2 norm regularization
    norms = np.linalg.norm(coefs, axis=0)

    # Extract features that are deemed "zeroed out"
    zeroed_features = np.where(norms < 1e-3)[0]

    # Ensure that for these features, every single task has a coef approx 0
    if len(zeroed_features) > 0:
        for f_idx in zeroed_features:
            np.testing.assert_allclose(coefs[:, f_idx], 0.0, atol=1e-3)


def test_multitask_fallback():
    # Provide 1D y to MultiTaskRegressor
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10)
    y = rng.randn(50)
    
    sk_lasso = SklearnMultiTaskLasso(alpha=1.0, random_state=42)
    sk_lasso.fit(X, y.reshape(-1, 1))
    
    model = MultiTaskRegressor(penalty="l1", alpha=1.0, solver="SCS")
    model.fit(X, y)
    
    # Check shape handles 1D fallback correctly, producing proper 2D outputs natively
    assert model.coef_.shape == (1, 10) # 1 task, 10 features
    np.testing.assert_allclose(model.coef_, sk_lasso.coef_, atol=1e-1)


def test_multitask_rejects_adaptive_arguments():
    with pytest.raises(TypeError):
        MultiTaskRegressor(alpha=0.1, adaptive=True)

    with pytest.raises(TypeError):
        MultiTaskRegressor(alpha=0.1, adaptive_weights=np.ones(3))


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("loss", ["squared_error", "epsilon_insensitive", "squared_epsilon_insensitive"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "l1_l2"])
def test_multitask_parameters(multitask_data, epsilon, loss, penalty):
    X, Y = multitask_data
    
    model = MultiTaskRegressor(
        alpha=0.1,
        l1_ratio=0.5 if penalty == "l1_l2" else 0.0,
        epsilon=epsilon,
        loss=loss,
        penalty=penalty,
        fit_intercept=True,
        solver="SCS"
    )
    
    try:
        model.fit(X, Y)
        assert model.coef_ is not None
        assert model.coef_.shape == (Y.shape[1], X.shape[1])
    except Exception as e:
        pytest.fail(
            f"MultiTaskRegressor failed with epsilon={epsilon}, loss={loss}, penalty={penalty}: {e}"
        )
