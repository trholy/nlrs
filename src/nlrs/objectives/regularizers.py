import cvxpy as cp
import numpy as np
from typing import Optional


def l1_penalty(
        beta: cp.Variable,
        alpha: float,
        adaptive_weights: Optional[np.ndarray] = None
) -> cp.Expression:
    """
    Calculate the L1 penalty.
    
    Args:
        beta (cp.Variable): Coefficients.
        alpha (float): Constant that multiplies the penalty.
        adaptive_weights (Optional[np.ndarray]): Adaptive weights.
        
    Returns:
        cp.Expression: The L1 penalty expression.
    """
    _validate_penalty_alpha(alpha, "L1")

    if len(beta.shape) == 1:
        if adaptive_weights is not None:
            adaptive_weights = _validate_adaptive_weights_1d(beta, adaptive_weights, "L1")
            return alpha * cp.sum(cp.multiply(adaptive_weights, cp.abs(beta)))
        return alpha * cp.sum(cp.abs(beta))
    else:
        # Multitask
        if adaptive_weights is not None:
            raise ValueError("Adaptive weights are not supported for multitask penalties.")
        return alpha * cp.sum(cp.norm(beta, p=2, axis=1))


def l2_penalty(
        beta: cp.Variable,
        alpha: float,
        adaptive_weights: Optional[np.ndarray] = None
) -> cp.Expression:
    """
    Calculate the L2 penalty.
    
    Args:
        beta (cp.Variable): Coefficients.
        alpha (float): Constant that multiplies the penalty.
        adaptive_weights (Optional[np.ndarray]): Adaptive weights.
        
    Returns:
        cp.Expression: The L2 penalty expression.
    """
    _validate_penalty_alpha(alpha, "L2")

    if len(beta.shape) == 1:
        if adaptive_weights is not None:
            adaptive_weights = _validate_adaptive_weights_1d(beta, adaptive_weights, "L2")
            return alpha * 0.5 * cp.sum(cp.multiply(adaptive_weights, cp.square(beta)))
        return alpha * 0.5 * cp.sum_squares(beta)
    else:
        # Multitask
        if adaptive_weights is not None:
            raise ValueError("Adaptive weights are not supported for multitask penalties.")
        return alpha * 0.5 * cp.sum(cp.power(cp.norm(beta, p=2, axis=1), 2))


def l1_l2_penalty(
        beta: cp.Variable,
        alpha: float,
        l1_ratio: float,
        adaptive_weights: Optional[np.ndarray] = None
) -> cp.Expression:
    """
    Calculate the ElasticNet (L1 + L2) penalty.
    
    Args:
        beta (cp.Variable): Coefficients.
        alpha (float): Constant that multiplies the penalty.
        l1_ratio (float): The ElasticNet mixing parameter.
        adaptive_weights (Optional[np.ndarray]): Adaptive weights.
        
    Returns:
        cp.Expression: The L1 + L2 penalty expression.
    """
    _validate_penalty_alpha(alpha, "L1+L2")
    if not np.isscalar(l1_ratio) or not np.isfinite(l1_ratio):
        raise ValueError("ElasticNet l1_ratio must be a finite scalar.")
    if not (0 <= l1_ratio <= 1):
        raise ValueError("ElasticNet l1_ratio must be in [0, 1].")

    l1_term = l1_penalty(beta, alpha * l1_ratio, adaptive_weights)
    l2_term = l2_penalty(beta, alpha * (1 - l1_ratio))
    return l1_term + l2_term
