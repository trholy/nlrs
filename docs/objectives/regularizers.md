# Regularizers (`nlrs.objectives.regularizers`)

This module defines standard sparsifying constraints as `cp.Expression` instances. It distinguishes between single-target 1D variables and multi-task 2D matrices automatically.

## Adaptive Lasso and Multi-Task Penalties

This library supports adaptive scaling (`adaptive_weights`) for single-target 1D coefficients and structured grouped penalties for multi-target variables. For multi-target variables, the $L_{2,p}$ mixed-norm formulation groups coefficients by feature index and adaptive weights are not supported.

## Functions

### `l1_penalty`

Constructs an overarching $L_{1}$ or $L_{2,1}$ formulation.
- **Variables**: `beta` (`cp.Variable`), `alpha` (float).
- Multi-target variables observe Group Lasso formulations via $L_{2,1}$: `cp.sum(cp.norm(beta, p=2, axis=1))`

### `l2_penalty`

Constructs Ridge regression formulations $L_2^2$ or Group Ridge representations.
- Single target: Uses `0.5 * cp.sum_squares(beta)`.
- Multi-target mode: Uses `$0.5 \sum_j ||\beta_j||_2^2$`.

### `l1_l2_penalty`

Elastic Net penalty cleanly merging both $L_1$ and $L_2$ constraints proportionally defined by `l1_ratio`.
- **Methodology**: Distributes `alpha * l1_ratio` to the `l1_penalty` and `alpha * (1 - l1_ratio)` to the `l2_penalty`.
- Works with `adaptive_weights` directed only into the $L_1$ sub-component for 1D `beta`, i.e. Adaptive Elastic Net keeps the $\ell_2$ part unweighted.
