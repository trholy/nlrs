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
    if len(beta.shape) == 1:
        if adaptive_weights is not None:
            return alpha * cp.sum(cp.multiply(adaptive_weights, cp.abs(beta)))
        return alpha * cp.sum(cp.abs(beta))
    else:
        # Multitask
        if adaptive_weights is not None:
            return alpha * cp.sum(cp.norm(cp.multiply(adaptive_weights, cp.abs(beta)), p=2, axis=1))
        return alpha * cp.sum(cp.norm(beta, p=2, axis=1))


def l2_penalty(
        beta: cp.Variable,
        alpha: float
) -> cp.Expression:
    """
    Calculate the L2 penalty.
    
    Args:
        beta (cp.Variable): Coefficients.
        alpha (float): Constant that multiplies the penalty.
        
    Returns:
        cp.Expression: The L2 penalty expression.
    """
    if len(beta.shape) == 1:
        return alpha * 0.5 * cp.sum_squares(beta)
    else:
        # Multitask
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
    l1_term = l1_penalty(beta, alpha * l1_ratio, adaptive_weights)
    l2_term = l2_penalty(beta, alpha * (1 - l1_ratio))
    return l1_term + l2_term
