import cvxpy as cp
import numpy as np
from typing import Optional


def squared_error(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the squared error loss.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The squared error loss expression.
    """
    n = X.shape[0]
    preds = X @ beta + intercept if intercept is not None else X @ beta
    return (1 / (2 * n)) * cp.sum_squares(preds - y)


def epsilon_insensitive(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        epsilon: float,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the epsilon-insensitive loss.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        epsilon (float): Epsilon value.
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The epsilon-insensitive loss expression.
    """
    n = X.shape[0]
    preds = X @ beta + intercept if intercept is not None else X @ beta
    return (1 / n) * cp.sum(cp.pos(cp.abs(preds - y) - epsilon))


def squared_epsilon_insensitive(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        epsilon: float,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the squared epsilon-insensitive loss.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        epsilon (float): Epsilon value.
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The squared epsilon-insensitive loss expression.
    """
    n = X.shape[0]
    preds = X @ beta + intercept if intercept is not None else X @ beta
    return (1 / n) * cp.sum(cp.power(cp.pos(cp.abs(preds - y) - epsilon), 2))


def huber(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        M: float = 1.0,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the Huber loss.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        M (float): Huber threshold parameter.
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The Huber loss expression.
    """
    n = X.shape[0]
    preds = X @ beta + intercept if intercept is not None else X @ beta
    return (1 / (2 * n)) * cp.sum(cp.huber(preds - y, M))


def quantile(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        q: float = 0.5,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the quantile loss.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        q (float): Quantile parameter in (0, 1).
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The quantile loss expression.
    """
    if not (0 < q < 1):
        raise ValueError("Quantile q must be strictly between 0 and 1.")
    n = X.shape[0]
    preds = X @ beta + intercept if intercept is not None else X @ beta
    res = y - preds
    return (1 / n) * cp.sum(q * cp.pos(res) + (1 - q) * cp.neg(res))


def mean_absolute_error(
        X: np.ndarray,
        y: np.ndarray,
        beta: cp.Variable,
        intercept: Optional[cp.Variable] = None
) -> cp.Expression:
    """
    Calculate the mean absolute error (MAE) loss using quantile loss as a special case.

    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        beta (cp.Variable): Coefficients.
        intercept (Optional[cp.Variable]): Intercept term.
        
    Returns:
        cp.Expression: The MAE loss expression.
    """
    return 2 * quantile(X, y, beta, q=0.5, intercept=intercept)

mae = mean_absolute_error


def get_loss_expr(
        loss_name: str,
        X: np.ndarray,
        y: np.ndarray,
        coef: cp.Variable,
        intercept: Optional[cp.Variable] = None,
        epsilon: float = 0.0,
        multiply_by_n: bool = False,
        M: float = 1.0,
        q: float = 0.5
) -> cp.Expression:
    """
    Factory function to get the appropriate loss expression from its string name.
    """
    n = X.shape[0]
    preds = X @ coef + intercept if intercept is not None else X @ coef
    
    if loss_name == "squared_error":
        expr = squared_error(X, y, coef, intercept)
    elif loss_name == "sum_squares":
        expr = cp.sum_squares(preds - y)
    elif loss_name == "epsilon_insensitive":
        expr = epsilon_insensitive(X, y, coef, epsilon, intercept)
        if multiply_by_n:
            expr = expr * n
    elif loss_name == "squared_epsilon_insensitive":
        expr = squared_epsilon_insensitive(X, y, coef, epsilon, intercept)
        if multiply_by_n:
            expr = expr * n
    elif loss_name == "huber":
        expr = huber(X, y, coef, M, intercept)
        if multiply_by_n:
            expr = expr * n
    elif loss_name == "quantile":
        expr = quantile(X, y, coef, q, intercept)
        if multiply_by_n:
            expr = expr * n
    elif loss_name in ["mae", "mean_absolute_error"]:
        expr = mean_absolute_error(X, y, coef, intercept)
        if multiply_by_n:
            expr = expr * n
    else:
        raise ValueError(f"Unknown loss {loss_name}")
        
    return expr
