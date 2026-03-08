# Loss Functions (`nlrs.objectives.losses`)

This module exposes the primitive cost functions driving the convex formulations in `nlrs`. All functions are built upon the `cvxpy` symbolic language, returning `cp.Expression` objects for solvers.

## Functions

### `squared_error`

Standard empirical risk minimization given by OLS.
- **Args**:
    - `X` (`np.ndarray`): Features.
    - `y` (`np.ndarray`): Targets.
    - `beta` (`cp.Variable`): Regressor coefficients.
    - `intercept` (`Optional[cp.Variable]`): Model intercept.
- **Returns**: `(1 / (2 * n)) * cp.sum_squares(preds - y)`

### `epsilon_insensitive`

Minimizes the absolute difference between predictions and targets, ignoring errors smaller than `epsilon`. Central to standard SVR models.
- **Args**:
    - `X`, `y`, `beta`, `intercept`: General parameters.
    - `epsilon` (float): Loss margin parameter.
- **Returns**: `(1 / n) * cp.sum(cp.pos(cp.abs(preds - y) - epsilon))`

### `squared_epsilon_insensitive`

Similar to `epsilon_insensitive`, but squares the non-zero errors. Promotes smoother differentiability properties beyond the boundary.
- **Returns**: `(1 / n) * cp.sum(cp.power(cp.pos(cp.abs(preds - y) - epsilon), 2))`

### `huber`

Calculates the robust Huber loss, bridging squared error for small errors and absolute error for large ones.
- **Args**:
    - `X`, `y`, `beta`, `intercept`: General parameters.
    - `M` (float, default `1.0`): Transition threshold where loss changes from quadratic to linear.
- **Returns**: `(1 / (2 * n)) * cp.sum(cp.huber(preds - y, M))`

### `quantile`

Calculates the generic, asymmetric quantile loss (also known as pinball loss).
- **Args**:
    - `X`, `y`, `beta`, `intercept`: General parameters.
    - `q` (float, default `0.5`): Quantile level to predict, must be strictly in `(0, 1)`.
- **Returns**: `(1 / n) * cp.sum(q * cp.pos(y - preds) + (1 - q) * cp.neg(y - preds))`

### `median_absolute_error` (`mae`)

Calculates the Median Absolute Error loss. Equivalent mathematically to `2 * quantile(X, y, beta, q=0.5)`. This is the basic symmetric $L_1$ loss on residuals.
- **Args**:
    - `X`, `y`, `beta`, `intercept`: General parameters.
- **Returns**: `2 * quantile(X, y, beta, q=0.5, intercept=intercept)`

### `get_loss_expr`

A factory wrapper returning the exact constructed expression based on string handles (`"squared_error"`, `"sum_squares"`, `"epsilon_insensitive"`, `"squared_epsilon_insensitive"`, `"huber"`, `"quantile"`, `"mae"`, `"median_absolute_error"`).
- **Args**: Same standard parameters plus:
    - `epsilon` (float): for epsilon-insensitive losses.
    - `M` (float): for Huber loss.
    - `q` (float): for quantile loss.
    - `multiply_by_n` (bool): which dictates whether the empirical mean is multiplied by `n_samples` to reflect un-normalized bounds common in certain SVR objective definitions.
