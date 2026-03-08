# Base Linear Model (`nlrs.linear_model.base`)

This module defines the abstract base convex linear model from which all regression models in the `nlrs` package inherit. It integrates with `cvxpy` to solve convex optimization problems and mimics the scikit-learn estimator interface.

## Classes

### `BaseConvexLinearModel(BaseEstimator, RegressorMixin, ABC)`

Abstract base class for all convex linear models. Provides the core machinery for setting up the solver, extracting the fitted coefficients, and predicting, while leaving the explicit problem formulation (`fit`) to the child classes.

#### Initialization Parameters

- `fit_intercept` (bool, default `True`): Whether to compute and include an intercept term in the model.
- `solver` (str, default `"CLARABEL"`): The CVXPY-supported solver used for optimization (e.g., `'CLARABEL'`, `'OSQP'`, `'SCS'`, `'GUROBI'`).
- `tol` (float, default `1e-5`): Tolerance below which coefficients are effectively thresholded to `0.0`.
- `warm_start` (bool, default `False`): Whether to reuse the solution of the previous call to `fit` as initialization.
- `time_limit` (Optional[float], default `None`): Maximum time in seconds given to the solver.
- `verbose` (bool, default `False`): Allows the solver to print its progress if set to `True`.

#### Methods

**`fit(X, y)`**
*(Abstract Method)*
Must be implemented by subclasses to build the specific `cvxpy` objective and constraints.
- `X` (`np.ndarray`): Training features of shape `(n_samples, n_features)`.
- `y` (`np.ndarray`): Target vector or matrix of shape `(n_samples,)` or `(n_samples, n_targets)`.
- **Returns**: Self.

**`predict(X) -> np.ndarray`**
Predicts target values based on the fitted coefficients (`coef_`) and, if included, the `intercept_`.
- `X` (`np.ndarray`): Features of shape `(n_samples, n_features)`.
- **Returns**: `np.ndarray` vector (or matrix, for multitask) of predictions. Requires the model to be fitted beforehand.

**`_solve_and_extract(obj, constraints, coef, intercept=None)`**
Internal helper that delegates the CVXPY problem formulation to `nlrs.solvers.base.solve_convex_problem()`. Upon successful optimization, it extracts the coefficient values, enforces sparsity based on `tol`, and enforces positivity constraints if the model was initialized with `positive=True`. Sets attributes such as `coef_`, `intercept_`, `solver_status_`, and `solver_stats_`.
- `obj` (`cp.Expression`): The convex objective to minimize.
- `constraints` (list): Any constraints for the coefficients or variables.
- `coef` (`cp.Variable`): The coefficient CVXPY variable.
- `intercept` (`Optional[cp.Variable]`): The intercept CVXPY variable.
- **Returns**: Self.
