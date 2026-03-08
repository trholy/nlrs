# Validation Utilities (`nlrs.utils.validation`)

This module controls the safe formatting and boundary checking for underlying inputs passing into `nlrs` classes and optimization constraints.

## Functions

### `check_array`

Statically enforces strictly bound matrix sizes and formats (default to explicitly 2D configurations).
- **Args**:
    - `X` (`np.ndarray`): Data arrays.
    - `allow_nd` (bool): Defaults to `False`. Bypass for 2D restrictions.

### `check_X_y`

Rigorous input matching routine preventing mismatched matrix alignments upstream.
- **Mechanisms**: 
    - Verifies $X$ rows identically match $y$ sample scales.
    - Reshapes trailing dimension variants if `allow_multi_output=False` or standardizes into strictly columnar structure if `allow_multi_output=True`.
    - Forces floating point numeric casts dynamically via `np.issubdtype(X.dtype, np.number)`.

### `check_numeric_params`

Validates hyperparameter restrictions to mathematically bounds (e.g., verifying tolerance $\geq 0.0$ and ensuring string literals are rejected dynamically).
- **Args**: Takes variable arguments as dictionaries mapped to expected `(value, types, min, max)` bounds.
