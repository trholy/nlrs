import cvxpy as cp
import numpy as np
import pytest

from nlrs.objectives.regularizers import l1_l2_penalty, l1_penalty, l2_penalty


def test_l1_penalty_multitask_group_lasso():
    beta_val = np.array([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]])
    alpha = 0.7

    beta = cp.Variable((2, 3))
    expr = l1_penalty(beta, alpha)
    beta.value = beta_val

    expected = alpha * np.sum(np.linalg.norm(beta_val, axis=1))
    np.testing.assert_allclose(expr.value, expected, atol=1e-10)


def test_l2_penalty_multitask_group_ridge():
    beta_val = np.array([[1.0, -2.0, 0.5], [0.5, 1.5, -1.0]])
    alpha = 0.7

    beta = cp.Variable((2, 3))
    expr = l2_penalty(beta, alpha)
    beta.value = beta_val

    expected = alpha * 0.5 * np.sum(np.sum(beta_val ** 2, axis=1))
    np.testing.assert_allclose(expr.value, expected, atol=1e-10)


def test_l1_l2_penalty_1d_weights_applied_only_to_l1_component():
    beta_val = np.array([1.0, -2.0, 0.5])
    alpha = 0.9
    l1_ratio = 0.3
    adaptive_weights = np.array([2.0, 0.5, 1.5])

    beta = cp.Variable(3)
    expr = l1_l2_penalty(beta, alpha, l1_ratio, adaptive_weights)
    beta.value = beta_val

    expected = (
        l1_penalty(beta, alpha * l1_ratio, adaptive_weights).value
        + l2_penalty(beta, alpha * (1 - l1_ratio)).value
    )
    np.testing.assert_allclose(expr.value, expected, atol=1e-10)


def test_l1_l2_penalty_multitask_rejects_adaptive_weights():
    beta = cp.Variable((2, 3))
    with pytest.raises(ValueError, match="not supported for multitask"):
        l1_l2_penalty(beta, alpha=1.0, l1_ratio=0.5, adaptive_weights=np.ones(2))


def test_penalty_rejects_negative_alpha():
    beta = cp.Variable(3)
    with pytest.raises(ValueError, match="non-negative"):
        l1_penalty(beta, -0.1)
    with pytest.raises(ValueError, match="non-negative"):
        l2_penalty(beta, -0.1)


def test_l1_l2_rejects_invalid_l1_ratio():
    beta = cp.Variable(3)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        l1_l2_penalty(beta, alpha=1.0, l1_ratio=-0.2)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        l1_l2_penalty(beta, alpha=1.0, l1_ratio=1.2)


def test_adaptive_weights_reject_negative_or_nonfinite():
    beta = cp.Variable(3)
    with pytest.raises(ValueError, match="non-negative"):
        l1_penalty(beta, alpha=1.0, adaptive_weights=np.array([1.0, -1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        l2_penalty(beta, alpha=1.0, adaptive_weights=np.array([1.0, np.inf, 2.0]))


def test_adaptive_weights_reject_shape_mismatch():
    beta_1d = cp.Variable(3)
    with pytest.raises(ValueError, match="shape"):
        l1_penalty(beta_1d, alpha=1.0, adaptive_weights=np.ones(2))

    beta_2d = cp.Variable((2, 3))
    with pytest.raises(ValueError, match="not supported for multitask"):
        l1_penalty(beta_2d, alpha=1.0, adaptive_weights=np.ones(3))
    with pytest.raises(ValueError, match="not supported for multitask"):
        l2_penalty(beta_2d, alpha=1.0, adaptive_weights=np.ones((3, 2)))
