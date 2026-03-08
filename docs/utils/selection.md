# Selection Utilities (`nlrs.utils.selection`)

This module provides dynamic feature weighting classes crucial for unlocking properties like the **Adaptive Lasso**.

## Classes

### `AdaptiveWeights`

Derives initial importance estimators from a standard or cross-validated base model, scaling penalties disproportionately inversely.

#### Initialization Architecture

- `model` (str): Determines the estimator type. Allowed modes: `"lin_reg"`, `"cv_ridge"`.
- `adaptive_weights_power` (float, default `1.0`): Exponent $\gamma$ controlling adaptation severity.
- `tol` (float, default `1e-6`): Stability denominator bounding to prevent undefined divisions.
- Configuration for `cv_ridge`: Allows `scoring`, `alphas`, and cross-validation control properties (`random_state`, `n_splits`, `shuffle`).

#### Methods

**`get_weights(X, y) -> np.ndarray`**
Generates $\omega_j = 1 / |\hat{\beta}_j|^\gamma$.
- `X`, `y`: Source matrices. Let $\hat{\beta}_j$ be the resulting linear or ridge coefficient magnitude.
- Ensures absolute magnitudes heavily penalize low relevance factors and minimally bound strong baseline features. Returns 1D vectors seamlessly integrated into `BaseConvexLinearModel` structures via `_PenalizedLinearModel._get_adaptive_weights()`.
