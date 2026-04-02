# Linear Models (`nlrs.linear_model.linear`)

This module implements standard and penalized linear regression models (Lasso, Ridge, ElasticNet, and MultiTask extensions). It heavily leverages `cvxpy` to flexibly formulate $L_1$, $L_2$, and combined regularization techniques, including advanced capabilities like **Adaptive Penalties** and **Multi-Task** regression.

## Special Capabilities

- **Positivity Constraints**: Setting `positive=True` ensures all returned coefficients are non-negative.
- **Adaptive Regularization**: In single-target models, setting `adaptive=True` triggers the internal `AdaptiveWeights` scheme. Adaptive weights are applied to pure `l1` and pure `l2` penalties. For combined `l1_l2` (Elastic Net), adaptive weights apply only to the `l1` component.
- **Multi-Task Shrinkage**: The `MultiTaskRegressor` jointly applies grouped regularization schemes across multiple targets to identify shared structural sparsity.

## Classes

### `LinearRegression(BaseConvexLinearModel)`

Ordinary least squares implemented via convex formulation.
- **Inherits bounds**: Allows for `positive=True` constraints on coefficients.

### `_PenalizedLinearModel(BaseConvexLinearModel)`

Intermediate class expanding the base model to handle regularized variables.
- **Added Parameters**:
    - `alpha` (float): Penalty scale factor.
    - `adaptive` (bool): Whether to use data-derived adaptive weights on penalties.
    - `adaptive_weights` (`np.ndarray`): Pre-computed weights if `adaptive` is conditionally bypassed.
    - `adaptive_weights_model` (str): Model generating the base weights (e.g., `'lin_reg'`).
    - `adaptive_weights_power` (float): Power term $\gamma$ shaping the adaptive influence.

### `Lasso(_PenalizedLinearModel)`

Lasso model optimizing the squared error with an $L_1$ penalty. Encourages high sparsity in features.
- When `adaptive=True`, acts as the **Adaptive Lasso**, scaling the $L_1$ terms by feature-dependent inverse-magnitudes derived from an unpenalized (or weakly Ridge-penalized) base model.

### `Ridge(_PenalizedLinearModel)`

Ridge model optimizing the squared error with an $L_2$ penalty, preventing large weights and gracefully handling multicollinearity.
- When `adaptive=True`, applies feature-dependent weighting to the squared $L_2$ terms.

### `ElasticNet(_PenalizedLinearModel)`

Model balancing both $L_1$ and $L_2$ penalties for stable feature selection among correlated groups.
- `l1_ratio` (float, default `0.5`): Mix between $L_1$ ($1.0$) and $L_2$ ($0.0$). 
- When `adaptive=True`, adaptive weights apply only to the $L_1$ sub-component.

### `MultiTaskRegressor(_PenalizedLinearModel)`

Advanced framework for multiple continuous dependent targets (shape `(n_samples, n_tasks)`). Implements "Group Lasso" / "Group Elastic Net" styles by applying $L_{2,1}$ or $L_{2,2}$ norms to the coefficient matrix.
- `alpha` (float, default `1.0`): Constant that multiplies the penalty terms.
- `l1_ratio` (float, default `0.5`): The ElasticNet mixing parameter.
- `loss` (str): Allowed variations are `"squared_error"`, `"epsilon_insensitive"`, `"squared_epsilon_insensitive"`, `"huber"`, `"quantile"`, `"mae"`, and `"mean_absolute_error"`.
- `penalty` (str): `"l1"`, `"l2"`, or `"l1_l2"`. Note that in MT settings, `"l1"` signifies mixed group norm $\sum_j ||\beta_j||_2$, encouraging row-wise (feature-level) sparsity globally across all tasks simultaneously.
- `epsilon` (float, default `0.0`): Epsilon for epsilon-insensitive loss.
- Adaptive weights are not supported for `MultiTaskRegressor`.
