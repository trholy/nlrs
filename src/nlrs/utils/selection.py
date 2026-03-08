from typing import Union

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold


class AdaptiveWeights:
    def __init__(
        self,
        model: str = "lin_reg",
        fit_intercept: bool = True,
        scoring: str = "neg_mean_squared_error",
        alphas: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
        adaptive_weights_power: float = 1.0,
        tol: float = 1e-6,
        random_state: Union[int, None] = None,
        n_splits: int = 5,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the AdaptiveWeights class.
        
        Args:
            model (str): Model to use for computing weights ('lin_reg' or 'cv_ridge').
            fit_intercept (bool): Whether to calculate the intercept.
            scoring (str): Scoring method for CV if CV model is used.
            alphas (tuple[float, ...]): Alphas for RidgeCV if used.
            adaptive_weights_power (float): Power to raise the absolute coefficients to.
            tol (float): Tolerance to prevent division by zero for very small coefficients.
            random_state (Union[int, None]): Random state for CV split.
            n_splits (int): Number of splits for CV.
            shuffle (bool): Whether to shuffle in CV.
        """
        self.adaptive_weights_power = adaptive_weights_power
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.alphas = np.array(alphas)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.scoring = scoring
        self.model = model
        self.tol = tol

    def get_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the adaptive weights based on the specified model.
        
        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
            
        Returns:
            np.ndarray: Computed adaptive weights.
        """
        if (not isinstance(self.model, str)
                or self.model not in [
                    "lin_reg", "cv_ridge"
                ]):
            msg = "'model' must be 'lin_reg' or 'cv_ridge'."
            raise ValueError(msg)

        if (not isinstance(self.adaptive_weights_power, (int, float))
                or self.adaptive_weights_power <= 0):
            msg = "'adaptive_weights_power' must be a positive numeric value."
            raise ValueError(msg)

        X_copy, y_copy = np.copy(X), np.copy(y)

        if self.model == "cv_ridge":
            estimator = RidgeCV(
                alphas=self.alphas,
                fit_intercept=self.fit_intercept,
                scoring=self.scoring,
                cv=KFold(
                    n_splits=self.n_splits,
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                ),
                gcv_mode="auto",
                alpha_per_target=False,
            )

        elif self.model == "lin_reg":
            estimator = LinearRegression(
                fit_intercept=self.fit_intercept,
                copy_X=True,
                n_jobs=None,
                positive=False,
            )

        estimator.fit(X_copy, y_copy)

        adaptive_weights = estimator.coef_
        adaptive_weights = np.where(
            np.abs(adaptive_weights) <= self.tol,
            self.tol,
            adaptive_weights,
        )
        return 1 / np.power(
            np.abs(adaptive_weights),
            self.adaptive_weights_power,
        )
