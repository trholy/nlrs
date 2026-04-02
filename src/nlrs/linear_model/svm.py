import logging
from typing import Optional
import numpy as np

from nlrs.linear_model.linear import _PenalizedLinearModel
from nlrs.objectives.regularizers import l1_penalty, l2_penalty, l1_l2_penalty

logger = logging.getLogger(__name__)


class LinearSVR(_PenalizedLinearModel):
    def __init__(
            self,
            alpha: float = 1.0,
            epsilon: float = 0.0,
            l1_ratio: float = 0.5,
            loss: str = "epsilon_insensitive",
            penalty: str = "l2",
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
        Initialize the LinearSVR model.

        Note:
            This implementation does not regularize the intercept term.
        
        Args:
            alpha (float): Constant that multiplies the penalty terms.
            epsilon (float): Epsilon for epsilon-insensitive loss.
            l1_ratio (float): The ElasticNet mixing parameter.
            loss (str): Loss function to use.
            penalty (str): Penalty function to use ('l1', 'l2', 'l1_l2').
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
        self.epsilon = epsilon
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.penalty = penalty

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the LinearSVR model.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
            
        Returns:
            self: Returns an instance of self.
        """
        weights = self._get_adaptive_weights(X, y) if self.adaptive else None
        
        if self.loss in [
            "huber", "mae", "mean_absolute_error", "quantile", "squared_error", "sum_squares"
        ]:
            raise ValueError(f"LinearSVR does not support '{self.loss}' loss.")
        
        if self.penalty == "l1":
            return self._fit_linear_penalized(
                X, y, l1_penalty, loss=self.loss,
                epsilon=self.epsilon, alpha=self.alpha,
                adaptive_weights=weights
            )
        elif self.penalty == "l2":
            return self._fit_linear_penalized(
                X, y, l2_penalty, loss=self.loss,
                epsilon=self.epsilon, alpha=self.alpha * 2
            )
        elif self.penalty == "l1_l2":
            return self._fit_linear_penalized(
                X, y, l1_l2_penalty, loss=self.loss,
                epsilon=self.epsilon, alpha=self.alpha, l1_ratio=self.l1_ratio,
                adaptive_weights=weights
            )
        else:
            raise ValueError(f"Unknown penalty {self.penalty}")
