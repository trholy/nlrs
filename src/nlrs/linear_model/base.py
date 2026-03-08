import numpy as np
from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod
from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
import cvxpy as cp


class BaseConvexLinearModel(BaseEstimator, RegressorMixin, ABC):
    def __init__(
            self,
            fit_intercept: bool = True,
            solver: str = "CLARABEL",
            tol: float = 1e-5,
            warm_start: bool = False,
            time_limit: Optional[float] = None,
            verbose: bool = False
    ):
        """
        Initialize the base convex linear model.
        
        Args:
            fit_intercept (bool): Whether to calculate the intercept.
            solver (str): The solver to use.
            tol (float): Tolerance for the solver.
            warm_start (bool): Whether to reuse the solution of the previous call to fit.
            time_limit (Optional[float]): Time limit for the solver in seconds.
            verbose (bool): Whether to print verbose output from the solver.
        """
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.tol = tol
        self.warm_start = warm_start
        self.time_limit = time_limit
        self.verbose = verbose

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to the training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,) or (n_samples, n_targets).
            
        Returns:
            self: Returns an instance of self.
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Args:
            X (np.ndarray): Samples to predict on, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted values.
        """
        check_is_fitted(self, "coef_")
        # In multitask (MultiTaskRegressor), coef_ is stored as (n_targets, n_features).
        # We need to transpose it for prediction: X @ coef_.T
        w = self.coef_.T if self.coef_.ndim == 2 else self.coef_
        if hasattr(self, "intercept_") and self.intercept_ is not None:
            # For multitask, intercept is a row vector or scalar. Using broadasting.
            return X @ w + self.intercept_
        return X @ w

    def _solve_and_extract(
            self,
            obj: cp.Expression,
            constraints: list,
            coef: cp.Variable,
            intercept: Optional[cp.Variable] = None
    ):
        """
        Solve the convex problem and extract the values to the class attributes.
        """
        from nlrs.solvers.base import solve_convex_problem
        import logging
        logger = logging.getLogger(__name__)

        solution = solve_convex_problem(
            obj, constraints,
            self.solver, self.verbose, self.warm_start, self.time_limit
        )
        if solution is None:
            return self

        self.solver_kwargs_ = solution.solver_kwargs
        self.solver_status_ = solution.status
        self.solver_stats_ = solution.stats

        if coef.value is None:
            logger.warning("Optimization failed to return coefficient values.")
            return self

        coef_val = np.where(np.abs(coef.value) < self.tol, 0.0, coef.value)
        
        is_positive = getattr(self, "positive", False)
        if is_positive:
            if np.min(coef_val) < 0:
                logger.warning(f"Coefficients below zero, min: {np.min(coef_val)}.")
            coef_val = np.where(coef_val < 0, 0, coef_val)
            
        self.coef_ = coef_val

        if intercept is not None and intercept.value is not None:
            # intercept is a scalar or array depending on formulation
            int_val = intercept.value
            if isinstance(int_val, np.ndarray) and int_val.size == 1:
                int_val = float(int_val.item())
            
            if isinstance(int_val, float):
                int_val = 0.0 if abs(int_val) < self.tol else int_val
            else:
                int_val = np.where(np.abs(int_val) < self.tol, 0.0, int_val)
                
            self.intercept_ = int_val
        else:
            self.intercept_ = None
            
        return self
