# NLRS: Non-Standard Linear Regression Methods with Sparsity

A Python Library for **N**on-Standard **L**inear **Re**gression Methods with **S**parsity.

## Description

The `nlrs` library provides robust, convex implementations of linear regression models equipped with varied sparsity-inducing regularizations (`Lasso`, `Ridge`, `Elastic Net`) and `Support Vector Regression` (SVR). 

Built on top of the open-source Python-embedded modeling language `CVXPY`, `nlrs` extends conventional libraries like `scikit-learn` by enabling advanced modeling constraints and adaptive penalties. By solving problems via CVXPY, `nlrs` natively interfaces with open-source solvers (e.g., `ECOS`, `OSQP`, `CLARABEL`, `SCS`) and commercial solvers (e.g., `CPLEX`, `Gurobi`).

### Key Capabilities and Properties
- **Adaptive Penalties**: Most models support `adaptive=True`, scaling $L_1$ and $L_2$ penalties by feature-specific inverse magnitude weights derived from a generic fit (e.g. Adaptive Lasso, Adaptive Elastic Net). This heavily suppresses irrelevant features while allowing dominant signals to bypass extreme penalization.
- **Constraints**: Enforces physical boundaries directly during optimization, offering `positive=True` attributes to restrict all estimated coefficients to $\geq 0$.
- **Multi-Task Regression**: Features like `MultiTaskRegressor` natively support grouped structure shrinkage (Group Lasso, Group Elastic Net), applying mixed $L_{2,1}$ norms to identify feature relevance globally across multiple target variables simultaneously.
- **Scikit-Learn Compatibility**: All models inherit from `BaseEstimator` and `RegressorMixin`, making them plug-and-play ready for scikit-learn cross-validation, pipelines, and search grids.

---

## Documentation

Read the documentation on [GitLab Pages](https://to82lod.gitpages.uni-jena.de/nlrs/).

## Installation

You can install `nlrs` locally by cloning the repository and installing it via `pip`:

```bash
git clone https://github.com/trholy/nlrs
cd nlrs
pip install .
```

---

## Quick Start Example

Below is a short example demonstrating how to train an **Adaptive Elastic Net** and a **Multi-Task Regressor**.

```python
import numpy as np

from nlrs.linear_model.linear import ElasticNet, MultiTaskRegressor
from nlrs.linear_model.svm import LinearSVR

# 1. Generate dummy data
np.random.seed(42)
X = np.random.randn(100, 10)
# A single target
y = X @ np.random.randn(10) + np.random.randn(10) * 0.1
# A multi-output target
y_multi = np.column_stack([y, X @ np.random.randn(10) + np.random.randn(10)])

# 2. Fit an Adaptive Elastic Net
# 'adaptive=True' derives weights 1/|beta|^gamma to refine the penalty
model = ElasticNet(alpha=0.5, l1_ratio=0.5, adaptive=True, positive=True)
model.fit(X, y)
print("Adaptive ElasticNet Coefficients:\n", model.coef_)

# 3. Fit an Adaptive Multi-Task Regressor
# Penalty "l1_l2" applies a Group Elastic Net formulation across tasks
mt_model = MultiTaskRegressor(penalty="l1_l2", alpha=0.5, l1_ratio=0.5, adaptive=True)
mt_model.fit(X, y_multi)
print("\nMultiTask Coefficients Shape:", mt_model.coef_.shape) 

# 4. Fit a Support Vector Regressor (Linear, Epsilon-insensitive)
svr_model = LinearSVR(epsilon=0.1, penalty="l2", alpha=1.0)
svr_model.fit(X, y)
print("\nLinearSVR Coefficients:\n", svr_model.coef_)
```

---

## Methods Matrix

The formulations below represent the mathematical objectives minimized via CVXPY.

### Standard Linear Models (`nlrs.linear_model.linear`)

#### `LinearRegression`
- **Loss**: $\vert\vert X \beta - y \vert\vert_2^2$
- **Constraints**: `fit_intercept`, `positive=True`

#### `Lasso`
- **Loss**: $\frac{1}{2n} \vert\vert X \beta - y \vert\vert_2^2$
- **Penalties**:
    - $\ell_1$-penalty: $\lambda \cdot \vert\vert \beta \vert\vert_1$
    - Adaptive $\ell_1$-penalty:  $\lambda \cdot \sum_{i}^n \omega_i | \beta_i |$

#### `Ridge`
- **Loss**: $\vert\vert X \beta - y \vert\vert_2^2$
- **Penalties**:
    - $\ell_2$-penalty: $\lambda \cdot \vert\vert \beta \vert\vert_2^2$
    - Adaptive $\ell_2$-penalty: $\lambda \cdot \sum_{i}^n \omega_i \beta_i^2$

#### `ElasticNet`
- **Loss**: $\frac{1}{2n} \vert\vert X \beta - y \vert\vert_2^2$
- **Penalties**: Combines $\ell_1$ (scaled by $\alpha \cdot l1\_ratio$) and $\ell_2$ (scaled by $\alpha \cdot (1 - l1\_ratio)$). Automatically accepts adaptive weights directed to the absolute scale expressions.

#### `MultiTaskRegressor`
- **Properties**: Shrinks the coefficients associated with specific features uniformly across $m$ targets.
- **Loss Functions**: `squared_error`, `epsilon_insensitive`, `squared_epsilon_insensitive`, `huber`, `quantile`, `mae`.
- **Penalties**: Generalizes $\ell_1$ norm to sum over the $L_2$ norms of the feature vectors $\sum_j ||\beta_{j}||_2$. Extends robustly to `adaptive` arrays bounding cross-target structures.

### Support Vector Machines (`nlrs.linear_model.svm`)

#### `LinearSVR`
Optimizes via epsilon-insensitive margins allowing strict disregard for noise below the margin threshold $\epsilon$.
- **Loss Functions**:
    - `epsilon_insensitive`: $\sum_{j}^{m} \max\left(\left| x_{j} \beta - y_{j} \right| - \epsilon, 0\right)$
    - `squared_epsilon_insensitive`: $\sum_{j}^{m} \left( \max\left(\left| x_{j} \beta - y_{j} \right| - \epsilon, 0\right) \right)^2$
- **Penalties**: Natively integrates $\ell_1$, $\ell_2$, and combined Elastic Net constraints identically to linear models, including Adaptive weights support.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
