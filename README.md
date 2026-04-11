# NLRS: Non-Standard Linear Regression Methods with Sparsity

A Python Library for **N**on-Standard **L**inear **Re**gression Methods with **S**parsity.

## Description

The `nlrs` library provides convex implementations of linear regression models equipped with varied sparsity-inducing regularizations (`Lasso`, `Ridge`, `Elastic Net`) and `Support Vector Regression` (SVR). 

Built on top of the Python-embedded modeling language `CVXPY`, `nlrs` extends libraries like `scikit-learn` by enabling advanced modeling constraints, loss functions and adaptive penalties. By solving problems via CVXPY, `nlrs` natively interfaces with open-source solvers (e.g., `ECOS`, `OSQP`, `CLARABEL`, `SCS`) and commercial solvers (e.g., `CPLEX`, `Gurobi`).

### Key Capabilities and Properties
- **Adaptive Penalties**: Models support `adaptive=True`, scaling $\ell_1$ or $\ell_2$ penalties by inverse-magnitude weights derived from a previous model fit (e.g. Linear Regression or RidgeCV). For combined `l1_l2` penalties, adaptive weights are applied only to the $\ell_1$ component.
- **Constraints**: Models support `positive=True` to restrict all estimated coefficients to $\geq 0$.
- **Multi-Task Regression**: `MultiTaskRegressor` natively support grouped shrinkage (Lasso, Elastic Net, Support Vector Regression), applying mixed $\ell_{2,1}$ norm to identify feature relevance globally across multiple target variables simultaneously.
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

## Methods

The formulations below represent the objectives minimized via CVXPY.

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
- **Penalties**: Combines $\ell_1$ (scaled by $\alpha \cdot \ell_1\text{-ratio}$) and $\ell_2$ (scaled by $\alpha \cdot (1 - \ell_1\text{-ratio})$).

#### `MultiTaskRegressor`
- **Loss Functions**: `squared_error`, `epsilon_insensitive`, `squared_epsilon_insensitive`, `huber`, `quantile`, `mae` (alias: `mean_absolute_error`).
- **Penalties**: Generalizes $\ell_1$ norm to sum over the $\ell_2$ norms of the feature vectors $\sum_j ||\beta_{j}||_2$.
- **Adaptive Weights**: Not supported for multitask regression.

### Support Vector Machines (`nlrs.linear_model.svm`)

#### `LinearSVR`
Optimizes via epsilon-insensitive margins allowing strict disregard for noise below the margin threshold $\epsilon$.
- **Loss Functions**:
    - Supports all losses from `nlrs.objectives.losses.get_loss_expr`: `squared_error`, `sum_squares`, `epsilon_insensitive`, `squared_epsilon_insensitive`, `huber`, `quantile`, `mae` (alias `mean_absolute_error`).
- **Penalties**: Natively integrates $\ell_1$, $\ell_2$, and Elastic Net penalty. Adaptive weights apply to pure $\ell_1$- or $\ell_2$-penalty; for $\ell_1\ell_2$-penalty, only the $\ell_1$ part is weighted.
- **Intercept Regularization Note**: `nlrs.linear_model.svm.LinearSVR` does not regularize the intercept term, while scikit-learn/liblinear regularizes the intercept feature (`intercept_scaling`).

---

## Project Structure

```
./
├── .gitignore
├── .gitlab-ci.yml
├── LICENSE
├── README.md
├── THIRD_PARTY_LICENSES.txt
├── docs
│   ├── linear_model
│   │   ├── base.md
│   │   ├── linear.md
│   │   └── svm.md
│   ├── objectives
│   │   ├── losses.md
│   │   └── regularizers.md
│   ├── solvers
│   │   └── base.md
│   └── utils
│       ├── selection.md
│       └── validation.md
├── examples
│   ├── adaptive_lasso.py
│   ├── multitask_lasso.py
│   └── multitask_svr.py
├── mkdocs.yml
├── pyproject.toml
├── setup.py
├── src
│   ├── nlrs
│   │   ├── __init__.py
│   │   ├── linear_model
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── linear.py
│   │   │   └── svm.py
│   │   ├── objectives
│   │   │   ├── __init__.py
│   │   │   ├── losses.py
│   │   │   └── regularizers.py
│   │   ├── solvers
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── selection.py
│   │       └── validation.py
└── tests
    ├── test_AdaptiveWeights.py
    ├── test_adaptive.py
    ├── test_adaptive_advanced.py
    ├── test_collinear.py
    ├── test_edge_cases.py
    ├── test_linear.py
    ├── test_losses.py
    ├── test_multitask.py
    ├── test_multitask_advanced.py
    ├── test_sklearn_compat.py
    ├── test_svm.py
    └── test_validation.py
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
