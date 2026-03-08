# Solvers (`nlrs.solvers.base`)

This module functions as the dispatching mechanism linking internal `cp.Expression` optimization structures to `cvxpy.Problem.solve()`.

## Classes

### `Solution`
A dataclass-like structure capturing the returned outputs of the `problem.solve()` pipeline.
- Integrates properties: `status` (string completion code), `stats` (`SolverStats` time tracking), `solver_kwargs` (solver parameters).

## Functions

### `get_solver_kwargs`
Inspects installed CVXPY properties to validate dependencies and construct time-limit dictionary variations. (e.g., `"CPLEX"` maps to `"cplex_params"`, whereas `"GUROBI"` maps to `"TimeLimit"` keys).
- **Args**: `solver_name` (str), `verbose` (bool), `warm_start` (bool), `time_limit` (float).

### `solve_convex_problem`
Main runner. Attempts evaluating `objective` and `constraints` through CVXPY configurations. Identifies failures and explicitly checks if `status` is among the successful constraints (`"optimal"`, `"optimal_inaccurate"`).
- **Args**: `objective` (`cp.Expression`), `constraints` (list), `solver` (str), etc.
- **Returns**: Valid `Solution` instance or `None` on catastrophic failure.
