import logging
from typing import Optional
import cvxpy as cp
import numpy as np

from nlrs.linear_model.base import BaseConvexLinearModel
from nlrs.objectives.losses import get_loss_expr
from nlrs.objectives.regularizers import l1_penalty, l2_penalty, l1_l2_penalty
from nlrs.utils.validation import check_X_y
from nlrs.utils.selection import AdaptiveWeights

logger = logging.getLogger(__name__)


class LinearRegression(BaseConvexLinearModel):
    def __init__(
            self,
            fit_intercept: bool = True,
            positive: bool = False,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the linear regression model.
        
        Args:
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            fit_intercept=fit_intercept,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )
        self.positive = positive

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the linear regression model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: Returns an instance of self.
        """
        X, y = check_X_y(X, y)
        n, p = X.shape
        
        coef = cp.Variable(p)
        intercept = cp.Variable() if self.fit_intercept else None
        
        constraints = [coef >= 0] if self.positive else []
        obj = get_loss_expr("squared_error", X, y, coef, intercept)

        return self._solve_and_extract(obj, constraints, coef, intercept)


class _PenalizedLinearModel(BaseConvexLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            positive: bool = False,
            adaptive: bool = False,
            adaptive_weights: Optional[np.ndarray] = None,
            adaptive_weights_model: str = "lin_reg",
            adaptive_weights_power: float = 1.0,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the penalized linear model.
        
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            adaptive (bool): Whether to use adaptive weights.
            adaptive_weights (Optional[np.ndarray]): Array of weights for adaptive penalty.
            adaptive_weights_model (str): Model to use for computing adaptive weights if not provided.
            adaptive_weights_power (float): Power to raise the weights.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            fit_intercept=fit_intercept,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )
        self.alpha = alpha
        self.positive = positive
        self.adaptive = adaptive
        self.adaptive_weights = adaptive_weights
        self.adaptive_weights_model = adaptive_weights_model
        self.adaptive_weights_power = adaptive_weights_power

    def _get_adaptive_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute or return adaptive weights.
        
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
            
        Returns:
            np.ndarray: Computed adaptive weights.
        """
        if self.adaptive_weights is not None:
            return self.adaptive_weights
            
        model = AdaptiveWeights(
            model=self.adaptive_weights_model,
            fit_intercept=self.fit_intercept,
            scoring="neg_mean_squared_error",
            alphas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            adaptive_weights_power=self.adaptive_weights_power,
            tol=1e-6,
            random_state=42,
            n_splits=5,
            shuffle=True,
        )
        return model.get_weights(X, y)

    def _fit_linear_penalized(
            self,
            X: np.ndarray,
            y: np.ndarray,
            penalty_fn,
            loss: str = "squared_error",
            epsilon: float = 0.0,
            **penalty_kwargs
    ):
        """
        Fit a linear model with a specific penalty and loss function.
        
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
            penalty_fn (callable): Penalty function to use.
            loss (str): Loss function to use.
            epsilon (float): Epsilon for epsilon-insensitive loss.
            **penalty_kwargs: Additional arguments for the penalty function.
            
        Returns:
            self: Returns an instance of self.
        """
        X_copy, y_copy = check_X_y(X, y)
        n, p = X_copy.shape

        coef = cp.Variable(p)
        intercept = cp.Variable() if self.fit_intercept else None
        
        constraints = [coef >= 0] if self.positive else []
        
        loss_expr = get_loss_expr(loss, X_copy, y_copy, coef, intercept, epsilon, multiply_by_n=True)
            
        reg = penalty_fn(coef, **penalty_kwargs)
        obj = loss_expr + reg

        return self._solve_and_extract(obj, constraints, coef, intercept)


class Lasso(_PenalizedLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            positive: bool = False,
            adaptive: bool = False,
            adaptive_weights: Optional[np.ndarray] = None,
            adaptive_weights_model: str = "lin_reg",
            adaptive_weights_power: float = 1.0,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the Lasso model.

        Args:
            alpha (float): Constant that multiplies the penalty terms.
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            adaptive (bool): Whether to use adaptive weights.
            adaptive_weights (Optional[np.ndarray]): Array of weights for adaptive penalty.
            adaptive_weights_model (str): Model to use for computing adaptive weights if not provided.
            adaptive_weights_power (float): Power to raise the weights.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            positive=positive,
            adaptive=adaptive,
            adaptive_weights=adaptive_weights,
            adaptive_weights_model=adaptive_weights_model,
            adaptive_weights_power=adaptive_weights_power,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Lasso model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: Returns an instance of self.
        """
        weights = self._get_adaptive_weights(X, y) if self.adaptive else None
        return self._fit_linear_penalized(
            X, y, l1_penalty, loss="squared_error",
            alpha=self.alpha, adaptive_weights=weights
        )


class Ridge(_PenalizedLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            fit_intercept: bool = True,
            positive: bool = False,
            adaptive: bool = False,
            adaptive_weights: Optional[np.ndarray] = None,
            adaptive_weights_model: str = "lin_reg",
            adaptive_weights_power: float = 1.0,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the Ridge model.

        Args:
            alpha (float): Constant that multiplies the penalty terms.
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            adaptive (bool): Whether to use adaptive weights.
            adaptive_weights (Optional[np.ndarray]): Array of weights for adaptive penalty.
            adaptive_weights_model (str): Model to use for computing adaptive weights if not provided.
            adaptive_weights_power (float): Power to raise the weights.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            positive=positive,
            adaptive=adaptive,
            adaptive_weights=adaptive_weights,
            adaptive_weights_model=adaptive_weights_model,
            adaptive_weights_power=adaptive_weights_power,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Ridge model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: Returns an instance of self.
        """
        weights = self._get_adaptive_weights(X, y) if self.adaptive else None
        return self._fit_linear_penalized(
            X, y, l2_penalty, loss="sum_squares", alpha=self.alpha * 2, adaptive_weights=weights
        )


class ElasticNet(_PenalizedLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            l1_ratio: float = 0.5,
            fit_intercept: bool = True,
            positive: bool = False,
            adaptive: bool = False,
            adaptive_weights: Optional[np.ndarray] = None,
            adaptive_weights_model: str = "lin_reg",
            adaptive_weights_power: float = 1.0,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the ElasticNet model.
        
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            l1_ratio (float): The ElasticNet mixing parameter.
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            adaptive (bool): Whether to use adaptive weights.
            adaptive_weights (Optional[np.ndarray]): Array of weights for adaptive penalty.
            adaptive_weights_model (str): Model to use for computing adaptive weights if not provided.
            adaptive_weights_power (float): Power to raise the weights.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            positive=positive,
            adaptive=adaptive,
            adaptive_weights=adaptive_weights,
            adaptive_weights_model=adaptive_weights_model,
            adaptive_weights_power=adaptive_weights_power,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )
        self.l1_ratio = l1_ratio

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ElasticNet model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: Returns an instance of self.
        """
        weights = self._get_adaptive_weights(X, y) if self.adaptive else None
        return self._fit_linear_penalized(
            X, y, l1_l2_penalty, loss="squared_error",
            alpha=self.alpha, l1_ratio=self.l1_ratio, adaptive_weights=weights
        )


class MultiTaskRegressor(_PenalizedLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            l1_ratio: float = 0.5,
            loss: str = "squared_error",
            penalty: str = "l1",
            epsilon: float = 0.0,
            fit_intercept: bool = True,
            positive: bool = False,
            adaptive: bool = False,
            adaptive_weights: Optional[np.ndarray] = None,
            adaptive_weights_model: str = "lin_reg",
            adaptive_weights_power: float = 1.0,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the MultiTaskRegressor model.
        
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            l1_ratio (float): The ElasticNet mixing parameter.
            loss (str): Loss function to use ('squared_error', 'epsilon_insensitive', 'squared_epsilon_insensitive').
            penalty (str): Penalty function to use ('l1', 'l2', 'l1_l2').
            epsilon (float): Epsilon for epsilon-insensitive loss.
            fit_intercept (bool): Whether to calculate the intercept.
            positive (bool): When set to True, forces the coefficients to be positive.
            adaptive (bool): Whether to use adaptive weights.
            adaptive_weights (Optional[np.ndarray]): Array of weights for adaptive penalty.
            adaptive_weights_model (str): Model to use for computing adaptive weights if not provided.
            adaptive_weights_power (float): Power to raise the weights.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            positive=positive,
            adaptive=adaptive,
            adaptive_weights=adaptive_weights,
            adaptive_weights_model=adaptive_weights_model,
            adaptive_weights_power=adaptive_weights_power,
            solver=solver,
            tol=tol,
            warm_start=warm_start,
            time_limit=time_limit,
            verbose=verbose
        )
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.penalty = penalty
        self.epsilon = epsilon

    def _get_adaptive_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute or return adaptive weights for multitask setting.
        
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
            
        Returns:
            np.ndarray: Computed adaptive weights.
        """
        if self.adaptive_weights is not None:
            return self.adaptive_weights
            
        model = AdaptiveWeights(
            model=self.adaptive_weights_model,
            fit_intercept=self.fit_intercept,
            scoring="neg_mean_squared_error",
            alphas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            adaptive_weights_power=self.adaptive_weights_power,
            tol=1e-6,
            random_state=42,
            n_splits=5,
            shuffle=True,
        )
        return model.get_weights(X, y).T

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the MultiTaskRegressor model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples, n_tasks).
            
        Returns:
            self: Returns an instance of self.
        """
        X_copy, y_copy = check_X_y(X, y, allow_multi_output=True)
        if y_copy.ndim == 1:
            y_copy = y_copy.reshape(-1, 1)

        n, p = X_copy.shape
        n_tasks = y_copy.shape[1]
        
        coef = cp.Variable((p, n_tasks))
        intercept = cp.Variable(n_tasks) if self.fit_intercept else None
        
        constraints = [coef >= 0] if self.positive else []
        weights = self._get_adaptive_weights(X, y_copy) if self.adaptive else None
        
        loss_expr = get_loss_expr(
            self.loss, X_copy, y_copy,
            coef, intercept, self.epsilon, multiply_by_n=False
        )
            
        if self.penalty == "l1":
            reg_expr = l1_penalty(coef, self.alpha, weights)
        elif self.penalty == "l2":
            reg_expr = l2_penalty(coef, self.alpha, weights)
        elif self.penalty == "l1_l2":
            reg_expr = l1_l2_penalty(coef, self.alpha, self.l1_ratio, weights)
        else:
            raise ValueError(f"Unknown penalty {self.penalty}")

        obj = loss_expr + reg_expr

        self._solve_and_extract(obj, constraints, coef, intercept)
        
        if hasattr(self, "coef_") and self.coef_ is not None:
            self.coef_ = self.coef_.T
            
        return self
