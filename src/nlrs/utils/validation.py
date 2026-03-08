import numpy as np


def check_array(
        X: np.ndarray,
        allow_nd: bool = False
) -> np.ndarray:
    """
    Check if the input array is valid.
    
    Args:
        X (np.ndarray): Input array to check.
        allow_nd (bool): Whether to allow n-dimensional arrays.
        
    Returns:
        np.ndarray: The validated array.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    if allow_nd:
        return X
    if X.ndim != 2:
        raise ValueError(
            f"Input feature matrix must be 2-dimensional. Got shape {X.shape}."
        )
    return X


def check_X_y(
        X: np.ndarray,
        y: np.ndarray,
        allow_multi_output: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Check if X and y are valid input combinations.
    
    Args:
        X (np.ndarray): Training data.
        y (np.ndarray): Target values.
        allow_multi_output (bool): Whether to allow 2D y array.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: Validated (X, y) tuple.
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("Input data must be NumPy arrays.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Number of samples in X and y must be equal."
            f" Got {X.shape[0]} and {y.shape[0]}."
        )

    if X.ndim != 2:
        raise ValueError(
            f"Input feature matrix must be 2-dimensional. Got shape {X.shape}."
        )

    if not allow_multi_output:
        if y.ndim > 1 and y.shape[1] > 1:
            raise ValueError(
                f"Target must be 1-dimensional or column vector."
                f" Got shape {y.shape}."
            )
        if y.ndim == 2:
            y = y.ravel()
    else:
        if y.ndim == 1:
            y = y.reshape(-1, 1)

    # Ensure float64 or common numeric type
    if not np.issubdtype(X.dtype, np.number):
        X = X.astype(float)
    if not np.issubdtype(y.dtype, np.number):
        y = y.astype(float)

    return X, y


def check_numeric_params(**kwargs):
    """
    Check if numeric parameters are within expected bounds.
    
    Args:
        **kwargs: Dictionary containing parameter names as keys and 
                  tuples of (value, allowed_types, min_value, max_value) as values.
                  
    Raises:
        TypeError: If parameter is not of the allowed types.
        ValueError: If parameter is out of bounds.
    """
    for name, (val, types, vmin, vmax) in kwargs.items():
        if val is None:
            continue
        if not isinstance(val, types):
            raise TypeError(f"'{name}' must be of type {types}.")
        if vmin is not None and val < vmin:
            raise ValueError(f"'{name}' must be >= {vmin}.")
        if vmax is not None and val > vmax:
            raise ValueError(f"'{name}' must be <= {vmax}.")
