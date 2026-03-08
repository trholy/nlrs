import logging
import cvxpy as cp
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_solver_kwargs(
        solver_name: str,
        verbose: bool,
        warm_start: bool,
        time_limit: Optional[float] = None
) -> Dict[str, Any]:
    """
    Get the keyword arguments for the given solver.
    
    Args:
        solver_name (str): Name of the solver.
        verbose (bool): Whether to enable verbose output.
        warm_start (bool): Whether to use warm start.
        time_limit (Optional[float]): Time limit for the solver in seconds.
        
    Returns:
        Dict[str, Any]: Dictionary of solver keyword arguments.
    """
    solver_upper = solver_name.upper()
    solvers = cp.installed_solvers()
    
    if solver_upper not in solvers:
        raise ValueError(
            f"Unknown solver. Known solvers are {solvers}. Got {solver_name}."
        )

    kwargs = {
        "solver": solver_upper,
        "verbose": verbose,
        "warm_start": warm_start,
    }

    if time_limit is not None:
        if solver_upper == "CPLEX":
            kwargs["cplex_params"] = {"timelimit": time_limit}
        elif solver_upper == "MOSEK":
            kwargs["mosek_params"] = {"MSK_DPAR_OPTIMIZER_MAX_TIME": time_limit}
        elif solver_upper == "GUROBI":
            kwargs["TimeLimit"] = time_limit
        elif solver_upper in ["CLARABEL", "OSQP", "PIQP", "CUOPT", "HIGHS"]:
            kwargs["time_limit"] = time_limit
        elif solver_upper == "SCS":
            kwargs["time_limit_secs"] = time_limit
        elif solver_upper in ["GLOP", "PDLP"]:
            kwargs["time_limit_sec"] = time_limit
        elif solver_upper == "CBC":
            kwargs["maximumSeconds"] = time_limit
        elif solver_upper == "XPRESS":
            kwargs["maxtime"] = int(time_limit)
        elif solver_upper == "SCIPY":
            kwargs["scipy_options"] = {"time_limit": time_limit}
        elif solver_upper == "SCIP":
            kwargs["scip_params"] = {"limits/time": time_limit}
        else:
            raise ValueError(
                f"Setting a time limit is not supported or explicitly handled for solver {solver_upper}. "
                "Consider removing the time_limit parameter for this solver."
            )
            
    return kwargs


class Solution:
    def __init__(self, status, stats, solver_kwargs):
        """
        Initialize the Solution object.
        
        Args:
            status: Status of the solver.
            stats: Solver statistics.
            solver_kwargs: Keyword arguments used by the solver.
        """
        self.status = status
        self.stats = stats
        self.solver_kwargs = solver_kwargs

def solve_convex_problem(
    objective: cp.Expression,
    constraints: list[cp.Constraint],
    solver: str,
    verbose: bool = False,
    warm_start: bool = False,
    time_limit: Optional[float] = None
) -> Optional[Solution]:
    """
    Solve a convex optimization problem using CVXPY.
    
    Args:
        objective (cp.Expression): Objective function to minimize.
        constraints (list[cp.Constraint]): List of CVXPY constraints.
        solver (str): Solver name to use.
        verbose (bool): Whether to print verbose output.
        warm_start (bool): Whether to use warm start.
        time_limit (Optional[float]): Time limit in seconds.
        
    Returns:
        Optional[Solution]: The solution object containing the status, stats, etc.
    """
    solver_kwargs = get_solver_kwargs(solver, verbose, warm_start, time_limit)
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        problem.solve(**solver_kwargs)
    except cp.error.SolverError as e:
        logger.warning(
            f"Solving the optimization problem failed. Error: {e}."
        )
        return None
        
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        logger.warning(
            f"Optimization did not find an optimal solution! Status: {problem.status}"
        )
        return None
        
    return Solution(
        problem.status, problem.solver_stats, solver_kwargs
    )
