"""DSB: differential spectra balance, the symbolic analytical reference.

DSB recovers the parameters of a model that is nonlinear in them by equating
the model's differential spectrum to the data's, order by order. The data
spectrum comes from a prior polynomial fit.

Each equation in the balance sets the model discrete equal to the data
discrete at the same order ``k``. The differential-transform factor ``H**k``
therefore cancels on both sides and the balance reduces to matching plain
Maclaurin coefficients ``f^(k)(0)/k!`` (see :mod:`dtfit._symbolic`).
For a polynomial those coefficients are just its ascending coefficients
``c_k``. Any differentiable model expression works, with no per-function
discrete rules.

The square balance is solved symbolically (``sympy.nonlinsolve``) to keep
DSB's analytical character; an overdetermined balance, more polynomial
coefficients than parameters, is refined with nonlinear least squares.
"""

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from dtfit._stats import information_criteria
from dtfit.log import echo
from dtfit.types import FittingResult, InitialGuess
from dtfit._input import _validate_p0
from dtfit._symbolic import model_params, taylor_coeffs


def fit_dsb(
    coeffs_poly: np.ndarray,
    expr: str,
    var: str,
    *,
    rank: int | None = None,
    p0: InitialGuess = None,
) -> FittingResult:
    """Fit ``expr`` by balancing its Maclaurin spectrum against a polynomial's.

    Args:
        coeffs_poly: Polynomial coefficients in ascending order:
            ``coeffs_poly[k]`` is the coefficient of ``var**k`` and equals the
            data's order-``k`` Maclaurin coefficient.
        expr: Model expression, e.g. ``"a0 + a1*x + a2*exp(a3*x)"``.
        var: Main variable name in ``expr``.
        rank: Number of balance equations (Maclaurin orders) to use. Defaults to
            all available polynomial coefficients, giving a square system when it
            equals the parameter count and an overdetermined one otherwise.
        p0: Optional initial guess used by the numeric refinement/fallback.

    Returns:
        FittingResult with the fitted coefficients, callable model and (for an
        overdetermined balance) a parameter covariance estimate.

    Raises:
        ValueError: The model expression has no free parameters; fewer
            polynomial coefficients than parameters; ``rank`` below the
            parameter count; or fewer informative Maclaurin orders (ones
            that carry a parameter) than parameters.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    n = len(params)
    if n == 0:
        raise ValueError("Model expression has no free parameters to fit.")

    p_coeffs = np.asarray(coeffs_poly, dtype=float)
    n_poly = p_coeffs.size
    if n_poly < n:
        raise ValueError(
            f"Polynomial has {n_poly} coefficients but the model has {n} "
            "parameters; the balance would be underdefined. Fit a "
            "higher-degree polynomial first."
        )

    max_rank = n_poly
    rank = max_rank if rank is None else min(int(rank), max_rank)
    if rank < n:
        raise ValueError(
            f"rank={rank} is below the {n} parameters; balance underdefined."
        )

    # H-free model spectrum: Maclaurin coefficients f^(k)(0)/k!.
    model_spec = taylor_coeffs(f_sym, t, rank - 1)
    # Balance: model discrete minus data discrete (the polynomial coefficient).
    balance = [model_spec[k] - sp.Float(p_coeffs[k]) for k in range(rank)]

    # Drop balance equations that carry no parameter (e.g. an identically zero
    # Maclaurin order such as atan's even terms): they only express model misfit
    # and would make the symbolic square system inconsistent.
    param_set = set(params)
    informative = [eq for eq in balance if eq.free_symbols & param_set]
    if len(informative) < n:
        raise ValueError(
            f"Only {len(informative)} of the first {rank} Maclaurin orders "
            f"constrain the {n} parameters; increase the polynomial degree "
            "(rank) so the balance can identify them."
        )

    coeffs, converged, message = _solve_balance(informative, params, n, p0)
    echo("DSB fitted coefficients:", coeffs)

    cov = (
        _balance_covariance(informative, params, coeffs)
        if len(informative) > n
        else None
    )

    # FittingResult lambdifies the fitted model lazily from expr+coeffs; skip
    # the eager compile here.
    return FittingResult(coeffs=coeffs, cov=cov,
                         expr=expr, var=var, names=tuple(str(p) for p in params),
                         converged=converged, message=message)


def _solve_balance(
    balance: list[sp.Expr],
    params: list[sp.Symbol],
    n: int,
    p0: InitialGuess,
) -> tuple[np.ndarray, bool, str]:
    """Solve the balance symbolically on the square subsystem, keeping DSB's
    analytical nature, then refine numerically when overdetermined. Falls back
    to pure numeric least squares if the symbolic solver finds nothing.

    Returns ``(coeffs, converged, message)``. ``converged`` reflects the
    numeric refinement or fallback when one runs, otherwise it records that
    the symbolic solve found a real root.
    """
    square = balance[:n]
    guess = _validate_p0(p0, params)

    try:
        solutions = _solve_symbolic(square, params)
        # Pick the real candidate with the smallest residual over the *full*
        # (possibly overdetermined) balance.
        best = min(
            (np.array([float(sp.re(v)) for v in sol]) for sol in solutions),
            key=lambda c: _residual_norm(balance, params, c),
        )
    except Exception as exc:  # noqa: BLE001 - symbolic solvers raise many types
        echo(f"Symbolic balance solve failed ({exc}); using numeric least squares.")
        return _solve_numeric(balance, params, guess)

    if len(balance) > n:
        echo("Overdetermined balance; refining numerically.")
        return _solve_numeric(balance, params, best)
    return best, True, "symbolic balance solved"


def _solve_symbolic(
    system: Sequence[sp.Basic] | sp.Expr,
    coeffs: list[sp.Symbol],
) -> list[list[sp.Expr]]:
    """Solve the square symbolic balance, keeping only real, non-degenerate
    candidates.

    Drops complex and incomplete roots along with the all-zero root. A root
    where only some parameters are exactly 0 is legitimate (an offset that is
    truly 0, for instance) and is kept. Raises if none remain.
    """
    raw = sp.nonlinsolve(system, coeffs)
    solutions: list[list[sp.Expr]] = []
    for sol in raw.args:
        numeric = [
            sp.N(v, chop=True) for v in cast("Iterable[sp.Basic]", sol)
        ]
        if any(sp.im(v) != 0 for v in numeric):
            continue  # complex (or symbolic/incomplete) root
        vals = [sp.re(v) for v in numeric]
        if all(v == 0 for v in vals):
            continue  # degenerate all-zero root
        solutions.append(vals)
    if not solutions:
        raise RuntimeError("No suitable real solutions found in the symbolic balance.")
    return solutions


def _solve_numeric(
    system: Sequence[sp.Basic] | sp.Expr,
    coeffs: list[sp.Symbol],
    guess: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    """Refine or solve the balance by nonlinear least squares from ``guess``.

    Returns ``(coeffs, converged, message)`` from the least-squares solver."""
    func = sp.lambdify(coeffs, system, "scipy")
    sol = least_squares(lambda c: func(*c), guess.astype(np.float64))
    return np.asarray(sol.x, dtype=np.float64), bool(sol.success), str(sol.message)


def _balance_funcs(balance: list[sp.Expr], params: list[sp.Symbol]):
    return [sp.lambdify(params, eq, "numpy") for eq in balance]


def _residual_norm(
    balance: list[sp.Expr], params: list[sp.Symbol], c: np.ndarray
) -> float:
    funcs = _balance_funcs(balance, params)
    try:
        r = np.array([float(f(*c)) for f in funcs])
    except (TypeError, ValueError, OverflowError):
        return np.inf
    return float(r @ r) if np.all(np.isfinite(r)) else np.inf


def _balance_covariance(
    balance: list[sp.Expr], params: list[sp.Symbol], c: np.ndarray
) -> np.ndarray | None:
    """Gauss-Newton parameter covariance ``sigma^2 (J^T J)^-1`` from the
    residual balance Jacobian at the solution (overdetermined case only)."""
    n = len(params)
    m = len(balance)
    jac_syms = [[sp.diff(eq, p) for p in params] for eq in balance]
    jac_funcs = [[sp.lambdify(params, d, "numpy") for d in row] for row in jac_syms]
    try:
        jac = np.array(
            [[float(jac_funcs[i][j](*c)) for j in range(n)] for i in range(m)]
        )
        funcs = _balance_funcs(balance, params)
        res = np.array([float(f(*c)) for f in funcs])
    except (TypeError, ValueError, OverflowError):
        return None
    jtj = jac.T @ jac
    try:
        jtj_inv = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        return None
    sigma2 = float(res @ res) / max(m - n, 1)
    return sigma2 * jtj_inv


def find_degree(
    data_x: np.ndarray,
    data_y: np.ndarray,
    method: str = "bic",
    max_degree: int = 12,
) -> int:
    """Select a polynomial degree for ``(data_x, data_y)`` by ``"bic"`` or
    ``"aic"``: the degree minimizing that criterion over ``0..max_degree``.

    Args:
        data_x: Sample locations.
        data_y: Sample values, same length as ``data_x``.
        method: ``"bic"`` or ``"aic"``.
        max_degree: Highest degree to try. Silently clamped to
            ``data_y.size - 1`` when larger.

    Returns:
        The selected degree.

    Raises:
        ValueError: ``method`` is neither ``"bic"`` nor ``"aic"``, or
            ``data_y`` is empty.

    Logs a debug-level message (see :func:`dtfit.log.echo`) when the
    selected degree is ``max_degree`` (after clamping), which means the
    search may not have found an interior minimum.
    """
    if method not in ("bic", "aic"):
        raise ValueError(
            f"Unsupported degree-selection method {method!r}; use 'bic' or 'aic'."
        )
    degree = _find_degree_direct(data_x, data_y, max_degree, method)
    if degree == max_degree:
        echo(f"Warning: maximum degree {max_degree} reached.")
    echo(f"Best polynomial degree selected by {method}: {degree}")
    return degree


def _find_degree_direct(
    data_x: np.ndarray,
    data_y: np.ndarray,
    max_degree: int,
    criterion: str,
) -> int:
    """Degree minimizing AIC/BIC computed from the residual sum of squares."""
    if data_y is None or data_y.size == 0:
        raise ValueError("data must be a non-empty 1-D array")

    n = data_y.size
    max_degree = int(min(max_degree, max(0, n - 1)))
    best_degree, best_score = 0, np.inf

    for deg in range(0, max_degree + 1):
        try:
            coeffs = np.polyfit(data_x, data_y, deg)
        except Exception:
            continue
        model = np.poly1d(coeffs)(data_x)
        rss = float(np.sum((data_y - model) ** 2))
        if rss <= 0:
            return deg
        aic, bic = information_criteria(rss, n, deg + 1)
        score = aic if criterion == "aic" else bic
        if score < best_score:
            best_degree, best_score = deg, score

    return int(best_degree)
