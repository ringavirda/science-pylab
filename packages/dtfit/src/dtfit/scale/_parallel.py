"""Parallel batch fitting: fan many independent fits across CPU cores.

The batch methods (:func:`dtfit.fit_lsi`, :func:`dtfit.fit_eac`) are pure per
problem, since fitting one signal never touches another. Real workloads
already arrive in that shape (the channels of a multivariate series, the
cells of a noise/size sweep, the chunks of a large stream, the axes of a
trajectory), making them embarrassingly parallel. :func:`fit_many` maps the
chosen method over a list of independent problems with :mod:`joblib`, so an
N-core machine fits ~N signals at once.

Two backends, both useful:

* ``backend="loky"`` (default) gives each worker its own process and true
  parallelism unaffected by the GIL. A problem carries its model as a
  picklable SymPy expression string; the worker rebuilds and lambdifies it,
  then returns a :class:`dtfit.FittingResult` that drops its lambdified
  callable on pickling and rebuilds it lazily on the caller side. Nothing
  unpicklable crosses the process boundary. That matters on Windows, where
  workers are spawned rather than forked.
* ``backend="threading"`` keeps the workers in one process sharing memory.
  The compiled numeric kernels (``dtfit._core._native``) release the GIL on their
  hot loops; the integral and projection work therefore runs concurrently,
  with no process or pickling overhead. Best when the per-problem arrays are
  large.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, cast

import numpy as np
from joblib import Parallel, delayed

from dtfit.methods import fit_lsi, fit_eac
from dtfit.types import FittingResult

__all__ = ["FittingProblem", "fit_many"]

_FITTERS: dict[str, Callable[..., FittingResult]] = {
    "lsi": fit_lsi,
    "eac": fit_eac,
}


@dataclass
class FittingProblem:
    """One independent fit, fully described by picklable data.

    Attributes:
        x, y: Observed samples for this problem.
        expr: Model expression string, e.g. ``"a*exp(b*t)"``.
        var: Main variable name in ``expr``.
        method: ``"lsi"`` or ``"eac"``.
        kwargs: Method-specific keyword arguments (e.g. ``p0``, ``bounds``).
        label: Optional tag carried through to the result (channel name, etc.).
    """

    x: np.ndarray
    y: np.ndarray
    expr: str
    var: str
    method: str = "lsi"
    kwargs: dict[str, Any] = field(default_factory=dict)
    label: Any = None


def _fit_one(problem: FittingProblem) -> FittingResult:
    """Worker entry point: fit a single problem, returning a picklable result.

    Kept at module level because a spawned worker process has to import it by
    name. A failed fit is recorded in ``error`` instead of killing the batch,
    and the returned :class:`FittingResult` carries the problem's ``label``.
    """
    fitter = _FITTERS.get(problem.method)
    if fitter is None:
        return FittingResult(
            coeffs=np.array([]), expr=problem.expr, var=problem.var,
            label=problem.label,
            error=f"unknown method {problem.method!r} (use 'lsi' or 'eac')",
        )
    try:
        res = fitter(
            np.asarray(problem.x, dtype=float),
            np.asarray(problem.y, dtype=float),
            problem.expr,
            problem.var,
            **problem.kwargs,
        )
        res.label = problem.label
        return res
    except Exception as exc:  # keep the batch alive; report per-problem
        return FittingResult(
            coeffs=np.array([]), expr=problem.expr, var=problem.var,
            label=problem.label, error=f"{type(exc).__name__}: {exc}",
        )


def fit_many(
    problems: Sequence[FittingProblem],
    *,
    n_jobs: int = -1,
    backend: str = "loky",
    verbose: int = 0,
) -> list[FittingResult]:
    """Fit many independent problems in parallel.

    Args:
        problems: Independent :class:`FittingProblem` specs.
        n_jobs: Worker count (``-1`` = all cores; ``1`` = serial, no pool).
        backend: ``"loky"`` (processes, default), ``"threading"`` (threads,
            riding the GIL-released native kernels), or
            ``"multiprocessing"``.
        verbose: Forwarded to :class:`joblib.Parallel` for progress
            reporting.

    Returns:
        One :class:`dtfit.FittingResult` per problem, in input order, each
        tagged with the problem's ``label``. A problem that failed has
        ``error`` set and empty ``coeffs``; the batch itself never aborts.
    """
    problems = list(problems)
    if not problems:
        return []
    if n_jobs == 1:  # avoid pool setup/serialization overhead for serial runs
        return [_fit_one(p) for p in problems]
    results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(_fit_one)(p) for p in problems
    )
    return cast("list[FittingResult]", list(results))
