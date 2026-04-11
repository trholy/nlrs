# Support Vector Regression (`nlrs.linear_model.svm`)

This module bridges penalized linear frameworks with support vector formulations to offer robust regressive capabilities. 

## Classes

### `LinearSVR(_PenalizedLinearModel)`

A linear convex regressor formulation leveraging CVXPY for direct primal-domain optimization. By default it uses epsilon-insensitive loss, but it can also be used with any loss exposed in `nlrs.objectives.losses.get_loss_expr`.

#### Initialization Parameters

- `alpha` (float, default `1.0`): The regularization scaling factor.
- `epsilon` (float, default `0.0`): The margin of tolerance wherein errors are penalized exactly zero.
- `l1_ratio` (float, default `0.5`): For ElasticNet-like regularizations if chosen.
- `loss` (str, default `"epsilon_insensitive"`): The base cost function. Supports all losses available in `nlrs.objectives.losses.get_loss_expr` (including `squared_error`, `sum_squares`, `epsilon_insensitive`, `squared_epsilon_insensitive`, `huber`, `quantile`, `mae`, and `mean_absolute_error`).
- `penalty` (str, default `"l2"`): Type of shrinkage, chosen from `"l1"`, `"l2"`, or `"l1_l2"`.
- `positive` (bool, default `False`): Confines the model strictly to non-negative features.
- `adaptive` (bool, default `False`): When `True`, applies adaptive weights to pure `l1` and pure `l2` penalties. For `l1_l2`, adaptive weights are applied only to the `l1` component.

By utilizing `epsilon_insensitive` loss alongside an `'l1_l2'` penalty and `adaptive=True`, users can formulate robust, sparsity-inducing target optimizations resistant to volatile target distributions.

#### Formulation Note vs scikit-learn

`nlrs` does **not** regularize the intercept term in `LinearSVR`.  
`scikit-learn`'s `LinearSVR` (liblinear backend) regularizes the synthetic intercept feature controlled by `intercept_scaling`, so objectives can differ slightly even with matched hyperparameters.
