"""Projected nonlinear least squares on an image, and the presets built on
it."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import differential_evolution, least_squares, minimize

from dtfit.methods._common import _validate_p0, normalize_bounds, normalize_p0
from dtfit.methods._modelinput import ModelSpec, resolve_model, result_kwargs
from dtfit.types import FittingResult
from .image import Image
from .original import Original


def _whitener(G: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of ``G`` with a relative jitter for near-
    singular Grams."""
    k = G.shape[0]
    jitter = 1e-14 * float(np.trace(G)) / k
    return cholesky(G + jitter * np.eye(k), lower=True)


def _solve(
    residual: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], np.ndarray],
    guess: np.ndarray,
    bounds: list[tuple[float, float]] | None,
    solver_options: dict[str, Any] | None,
    random_state: int | None,
    poor_cost: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, bool, str, int]:
    """Levenberg-Marquardt without bounds; trust-region with bounds.

    With bounds, a second, global stage runs when every bound is finite and
    the local trust-region solve is poor: it did not report success, its
    whitened residual energy ``sol.fun @ sol.fun`` is non-finite, or that
    energy exceeds ``poor_cost``. The global stage is a differential-
    evolution search polished by L-BFGS-B; its candidate replaces the local
    one only if its cost is strictly lower, so the local solution wins ties.
    Returns ``(coeffs, jacobian, converged, message, nfev)``, with ``nfev``
    the sum over every stage run.
    """
    opts = dict(solver_options or {})
    ls_opts = {
        k: opts[k] for k in ("xtol", "ftol", "gtol", "max_nfev") if k in opts
    }
    if bounds is None:
        sol = least_squares(residual, guess, jac=jac, method="lm", **ls_opts)
        return (
            sol.x, sol.jac, bool(sol.success), str(sol.message),
            int(sol.nfev),
        )
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    sol = least_squares(
        residual, np.clip(guess, lo, hi), jac=jac, bounds=(lo, hi),
        method="trf", **ls_opts,
    )
    local_cost = float(sol.fun @ sol.fun)
    finite = bool(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))
    poor = (
        (not sol.success)
        or (not np.isfinite(local_cost))
        or local_cost > poor_cost
    )
    if not (poor and finite):
        return (
            sol.x, sol.jac, bool(sol.success), str(sol.message),
            int(sol.nfev),
        )

    def cost(c: np.ndarray) -> float:
        r = residual(c)
        return float(r @ r)

    res_g = differential_evolution(
        cost, list(zip(lo, hi)), strategy="best1bin", popsize=15,
        seed=random_state,
    )
    min_opts = {k: opts[k] for k in ("ftol", "gtol") if k in opts}
    if "max_nfev" in opts:
        min_opts["maxfun"] = opts["max_nfev"]
    res = minimize(
        cost, res_g.x, method="L-BFGS-B", bounds=list(zip(lo, hi)),
        options=min_opts or None,
    )
    xg = np.asarray(res.x, dtype=float)
    global_cost = cost(xg)
    nfev = (
        int(sol.nfev) + int(getattr(res_g, "nfev", 0))
        + int(getattr(res, "nfev", 0))
    )
    if not global_cost < local_cost:
        return sol.x, sol.jac, bool(sol.success), str(sol.message), nfev
    return xg, jac(xg), bool(res.success), str(res.message), nfev


def _covariance(jac: np.ndarray, s2: float) -> np.ndarray | None:
    """``s2 (J^T J)^-1`` from the SVD of ``J``. A parameter with a
    component in a null direction of ``J`` is not identified: its diagonal
    entry is ``inf`` and its off-diagonal entries are ``nan``."""
    try:
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s[0] == 0.0:
        return None
    tol = s[0] * max(jac.shape) * float(np.finfo(float).eps)
    keep = s > tol
    inv_s2 = np.where(keep, 1.0 / np.where(keep, s, 1.0) ** 2, 0.0)
    cov = s2 * ((vt.T * inv_s2) @ vt)
    if not keep.all():
        null = np.abs(vt[~keep]).max(axis=0) > 1e-8
        idx = np.flatnonzero(null)
        cov[idx, :] = np.nan
        cov[:, idx] = np.nan
        cov[idx, idx] = np.inf
    return cov


def _rss_from_image(image: Image, S_f: np.ndarray) -> float:
    """``sum w (y - f)^2`` from the image alone; exact when ``f`` lies in
    the span."""
    d = image.S - S_f
    Ginv = np.linalg.pinv(image.G, hermitian=True)
    return float(image.sumsq - image.S @ image.beta + d @ (Ginv @ d))


def fit(
    model: Any,
    data: Original | Image,
    var: str | None = None,
    *,
    basis: Any = "legendre",
    order: int | None = None,
    p0: Any = None,
    bounds: Any = None,
    sigma: Any = None,
    absolute_sigma: bool = False,
    robust: bool = False,
    oscillatory: bool = False,
    freq_param: str | None = None,
    param_names: Any = None,
    solver_options: dict[str, Any] | None = None,
    random_state: int | None = 0,
) -> FittingResult:
    """Fit ``model`` to ``data`` by nonlinear least squares restricted to
    the span of a basis, on the data's image.

    ``data`` is an :class:`Original` or an :class:`Image`. An Original is
    imaged first in ``basis`` at ``order``; ``sigma`` (per-sample standard
    deviations) and ``robust`` apply to that construction only, so both
    raise ``TypeError`` with an Image. ``model`` is a SymPy expression
    string, a ``sympy.Expr`` or a callable ``f(x, *params)``; ``var`` names
    the variable of a symbolic model. ``p0`` and ``bounds`` follow the
    canonical parameter order (sorted names for a symbolic model, signature
    order for a callable) and accept name-keyed mappings. With
    ``absolute_sigma=False`` the covariance is scaled by the residual
    variance ``rss / (n - p)``; with ``True`` it is not, as in
    ``scipy.optimize.curve_fit``.

    Parameters:
        model: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``.
        data: An :class:`Original` (imaged here) or an :class:`Image`
            (used as given).
        var: The main variable name, required for a symbolic model; a
            label only for a callable.
        basis: The basis to image an Original in, ``"legendre"``,
            ``"block"``, or a :class:`~dtfit.image.Basis` instance.
            ``"auto"`` is rejected here (``TypeError``); it needs the
            samples an Image no longer carries.
        order: The basis order (polynomial degree for Legendre, window
            count for block). Required when ``data`` is an Original.
        p0: Initial guess, a positional sequence in canonical parameter
            order or a ``{name: value}`` mapping; ``None`` defaults to
            ones.
        bounds: Per-parameter bounds, a sequence of ``(lo, hi)`` pairs, a
            ``{name: (lo, hi)}`` mapping, or the ``(lo, hi)`` scipy
            convention; see :func:`dtfit.methods._common.normalize_bounds`.
        sigma: Per-sample standard deviations for an Original; builds the
            weights ``w = 1/sigma**2``. Raises ``TypeError`` with an Image
            and ``ValueError`` if the Original already carries weights.
        absolute_sigma: If ``False`` (default) the covariance is scaled by
            the residual variance ``rss / (n - p)``, as when ``sigma`` is
            only relative; if ``True`` it is not, as in
            ``scipy.optimize.curve_fit`` with true absolute uncertainties.
        robust: Huber-reweight the image when building it from an
            Original; raises ``TypeError`` with an Image (already built).
        oscillatory: Declares the model has a dominant frequency;
            validated here and used by the order rule and the frequency
            seed added in a later task. No effect yet.
        freq_param: The name of the frequency parameter when
            ``oscillatory=True``; likewise validated here for the later
            task.
        param_names: Parameter names for a callable model, in signature
            order after ``x``; introspected from the signature when
            omitted. Optional and cross-checked for a symbolic model.
        solver_options: Forwarded to the least-squares stage: ``xtol``,
            ``ftol``, ``gtol`` and ``max_nfev`` go to
            ``scipy.optimize.least_squares``; ``ftol`` and ``gtol`` also
            go to the polishing ``scipy.optimize.minimize`` call
            (``max_nfev`` becomes its ``maxfun``) when the bounded global
            stage runs.
        random_state: Seed for the bounded global stage's differential
            evolution. ``None`` makes that stage nondeterministic between
            calls; unused when the fit is unbounded or the local solve is
            not poor.

    Returns:
        A :class:`~dtfit.types.FittingResult` with ``coeffs``, ``cov``,
        ``converged``, ``message``, ``x_range`` (the domain), ``n_obs``,
        ``nfev`` and:

        - ``rss``: the weighted residual sum of squares over the samples
          when an Original was given, and the image identity
          ``sumsq - S^T G^-1 S + d^T G^-1 d`` when an Image was given;
          ``rss_source`` records which (``"samples"`` or ``"image"``).
        - ``tss``: the image's total weighted sum of squares about its
          weighted mean, ``sumsq - sumy**2 / wsum``.
        - ``cost``: the final optimizer cost ``0.5 * ||r||^2`` on the
          whitened image residual, not ``0.5 * rss``.
        - ``cov``: ``None`` when the degrees of freedom are exhausted
          (``n_obs <= len(names)``); otherwise a parameter with a
          component in a null direction of the Jacobian is unidentified,
          reported with an ``inf`` diagonal entry and ``nan``
          off-diagonal entries rather than a spuriously small variance.

    Raises:
        TypeError: ``robust=True``, ``sigma`` given, or ``basis="auto"``
            with an Image; ``data`` is neither an Original nor an Image.
        ValueError: ``order`` missing with an Original; the image has
            fewer coefficients than parameters; the model is not finite
            at ``p0`` on the data's grid; ``sigma`` given for an Original
            that already carries weights; a malformed ``p0`` or
            ``bounds`` (from the normalizers).
        RuntimeError: the model has no free parameters.
    """
    spec: ModelSpec = resolve_model(model, var, param_names=param_names)
    names = list(spec.names)
    if not names:
        raise RuntimeError("Model has no free parameters to fit.")
    p0_arr = normalize_p0(p0, names)
    bounds_list = normalize_bounds(bounds, names)
    guess = _validate_p0(p0_arr, names)

    if isinstance(data, Image):
        if robust:
            raise TypeError(
                "robust=True applies when the image is built; this image "
                "is already made (use Original.image(robust=True) or pass "
                "the Original)."
            )
        if sigma is not None:
            raise TypeError(
                "sigma applies when the image is built; pass the Original."
            )
        if basis == "auto":
            raise TypeError(
                "basis='auto' needs the samples; pass the Original."
            )
        image = data
    else:
        if not isinstance(data, Original):
            raise TypeError(
                f"data must be an Original or an Image, got "
                f"{type(data).__name__}"
            )
        if sigma is not None and data.weighted:
            raise ValueError(
                "the Original already carries weights; build it with "
                "sigma instead of passing sigma to fit"
            )
        original = (
            data if sigma is None
            else Original(data.x, data.y, sigma=sigma, domain=data.domain)
        )
        if order is None:
            raise ValueError("order is required")
        image = Image.of(original, basis, order, robust=robust)
    if image.n_coef < len(names):
        raise ValueError(
            f"an image with {image.n_coef} coefficients cannot identify "
            f"{len(names)} parameters; raise the order so the image has "
            f"at least {len(names)} coefficients"
        )

    Phi = image.phi()
    x = image.grid.positions()
    w = image.w if image.w is not None else np.ones(image.n)
    Lc = _whitener(image.G)

    def model_image(theta: np.ndarray) -> np.ndarray:
        return Phi.T @ (w * spec.eval(x, theta))

    def residual_at(theta: np.ndarray) -> np.ndarray | None:
        S_f = model_image(theta)
        if not np.all(np.isfinite(S_f)):
            return None
        return solve_triangular(Lc, image.S - S_f, lower=True)

    def jacobian(theta: np.ndarray) -> np.ndarray:
        cols = [Phi.T @ (w * d) for d in spec.param_derivs(x, theta)]
        J = -np.column_stack(cols)
        J = np.where(np.isfinite(J), J, 0.0)
        return solve_triangular(Lc, J, lower=True)

    tss = float(image.sumsq - image.sumy ** 2 / image.wsum)
    if bounds_list is not None:
        lo = np.array([b[0] for b in bounds_list])
        hi = np.array([b[1] for b in bounds_list])
        guess = np.clip(guess, lo, hi)
    r0 = residual_at(guess)
    if r0 is None:
        raise ValueError(
            "the model is not finite at p0 on the data's grid; "
            "rescale x or change p0"
        )
    sentinel = np.full(
        image.n_coef, max(1e6, 10.0 * float(np.linalg.norm(r0)))
    )

    def residual(theta: np.ndarray) -> np.ndarray:
        r = residual_at(theta)
        return sentinel if r is None else r

    coeffs, jac, converged, message, nfev = _solve(
        residual, jacobian, guess, bounds_list, solver_options,
        random_state, poor_cost=0.5 * tss,
    )

    S_f = model_image(coeffs)
    if isinstance(data, Original):
        f = spec.eval(x, coeffs)
        rss = float(np.sum(w * (original.y - f) ** 2))
        rss_source = "samples"
    else:
        rss = max(_rss_from_image(image, S_f), 0.0)
        rss_source = "image"
    converged = bool(converged and np.isfinite(rss))
    dof = image.n - len(names)
    s2 = 1.0 if absolute_sigma else (rss / dof if dof > 0 else float("nan"))
    cov = _covariance(jac, s2) if dof > 0 else None
    r = residual(coeffs)
    result = FittingResult(
        coeffs=coeffs, cov=cov, converged=converged, message=message,
        x_range=image.domain, n_obs=image.n, rss=rss, tss=tss, nfev=nfev,
        cost=0.5 * float(r @ r), **result_kwargs(spec, coeffs),
    )
    result.rss_source = rss_source
    return result
