"""Projected nonlinear least squares on an image, and the presets built on
it."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import numpy.polynomial.legendre as leg
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import differential_evolution, least_squares, minimize

from dtfit._signal import dominant_period
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
    ls_opts: dict[str, Any] = {
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
    if not sol.success:
        reason = "did not converge"
    elif not np.isfinite(local_cost):
        reason = "reached a non-finite cost"
    else:
        reason = "explained less than half the variance"

    def cost(c: np.ndarray) -> float:
        r = residual(c)
        return float(r @ r)

    res_g = differential_evolution(
        cost, list(zip(lo, hi)), strategy="best1bin", popsize=15,
        seed=random_state,
    )
    min_opts: dict[str, Any] = {
        k: opts[k] for k in ("ftol", "gtol") if k in opts
    }
    if "max_nfev" in opts:
        min_opts["maxfun"] = opts["max_nfev"]
    res = minimize(
        cost, res_g.x, method="L-BFGS-B", bounds=list(zip(lo, hi)),
        options=min_opts or None,
    )
    xg = np.asarray(res.x, dtype=float)
    global_cost = cost(xg)
    de_nfev = int(getattr(res_g, "nfev", 0)) + int(getattr(res, "nfev", 0))
    warnings.warn(
        f"the local solve from p0 {reason}; a differential-evolution "
        f"search ran ({de_nfev} function evaluations)",
        UserWarning, stacklevel=3,
    )
    nfev = int(sol.nfev) + de_nfev
    if not global_cost < local_cost:
        return sol.x, sol.jac, bool(sol.success), str(sol.message), nfev
    return xg, jac(xg), bool(res.success), str(res.message), nfev


# Serves fit() only; methods/_common.py has the whitened-residual variant
# for scale/_partitioned.py and _core/_spectral.py.
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


def fft_frequency_seed(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant angular frequency of ``y`` on the grid ``x``.

    The samples are interpolated onto a uniform grid first (an identity
    when ``x`` already is one); ``2 pi f`` is then the peak of the
    mean-removed real FFT of the resampled signal, with the DC bin
    ignored.

    Args:
        x: Sample positions, non-decreasing.
        y: Sample values, same length as ``x``.

    Returns:
        The angular frequency (radians per unit ``x``) of the strongest
        spectral peak.
    """
    x = np.asarray(x, dtype=float)
    xu = np.linspace(x[0], x[-1], x.size)
    yy = np.interp(xu, x, np.asarray(y, dtype=float))
    yy = yy - float(np.mean(yy))
    spec = np.abs(np.fft.rfft(yy))
    if spec.size:
        spec[0] = 0.0
    freqs = np.fft.rfftfreq(yy.size, d=float(xu[1] - xu[0]))
    return 2.0 * np.pi * float(freqs[int(np.argmax(spec))])


def osc_order(x: np.ndarray, y: np.ndarray, max_order: int = 200) -> int:
    """Legendre order that resolves the dominant cycle of ``y``.

    ``ceil(pi * cycles) + 8``, the polynomial resolution threshold with
    headroom, where ``cycles`` is the number of periods of the FFT-peak
    frequency spanned by ``x``.

    Args:
        x: Sample positions, non-decreasing.
        y: Sample values, same length as ``x``.
        max_order: Upper cap on the returned order, >= 1.

    Returns:
        The suggested Legendre order, at most ``max_order``.
    """
    w0 = fft_frequency_seed(x, y)
    cycles = w0 * float(x[-1] - x[0]) / (2.0 * np.pi)
    return min(int(np.ceil(np.pi * cycles)) + 8, max_order)


def _sensitivity_truncation(
    spec: ModelSpec, params: np.ndarray, domain: tuple[float, float],
    max_order: int,
) -> np.ndarray:
    """Relative L2 truncation error of every parameter sensitivity after
    each Legendre order ``0..max_order-1``, as an array of shape
    ``(p, max_order)``."""
    xi, q = leg.leggauss(4 * max_order)
    x0, x1 = domain
    xq = x0 + (x1 - x0) * (xi + 1.0) / 2.0
    V = leg.legvander(xi, max_order)
    j = np.arange(max_order + 1)
    out = []
    for s in spec.param_derivs(xq, params):
        beta = ((2 * j + 1) / 2.0) * (V.T @ (q * s))
        energy = beta ** 2 * 2.0 / (2 * j + 1)
        tot = float(energy.sum()) + 1e-300
        tail = np.cumsum(energy[::-1])[::-1]
        out.append(np.sqrt(tail[1:] / tot))
    return np.array(out)


def order_for(
    model: Any, params: Any, domain: tuple[float, float], *,
    var: str | None = None, tol: float = 0.02, max_order: int = 64,
    param_names: Any = None,
) -> int:
    """Smallest Legendre order that represents the model's sensitivities.

    The order at which every parameter sensitivity of the model at
    ``params`` is represented on ``domain`` to relative L2 error ``tol``,
    floored at ``n_params - 1`` and capped at ``max_order``. Measured
    against the model catalog, ``tol=0.02`` is at or above the order at
    which the projected fit reaches NLLS efficiency for every family.

    Args:
        model: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``.
        params: Parameter values, in canonical order, at which the
            sensitivities are measured.
        domain: ``(x0, x1)``, the interval the order is chosen for.
        var: The main variable name, required for a symbolic model.
        tol: Relative L2 truncation tolerance, in ``(0, 1)``.
        max_order: Upper cap on the returned order, >= 1.
        param_names: Parameter names for a callable model; see
            :func:`~dtfit.methods._modelinput.resolve_model`.

    Returns:
        The smallest order meeting ``tol`` for every parameter, or
        ``max_order`` if none does; never less than ``n_params - 1`` or 1.

    Raises:
        RuntimeError: the model has no free parameters.
    """
    spec = resolve_model(model, var, param_names=param_names)
    if not spec.names:
        raise RuntimeError("Model has no free parameters to fit.")
    p = np.asarray(params, dtype=float)
    err = _sensitivity_truncation(
        spec, p, (float(domain[0]), float(domain[1])), max_order,
    )
    ok = np.all(err < tol, axis=0)
    k = int(np.argmax(ok)) if ok.any() else max_order
    return max(k, len(spec.names) - 1, 1)


def coverage(
    model: Any, params: Any, image: Image, *, var: str | None = None,
    param_names: Any = None,
) -> float:
    """Largest relative truncation error of the model's sensitivities at
    the image's order.

    Above ``0.02`` the image loses information about ``params``: the
    order is too low to identify the model's parameter sensitivities from
    this image. ``0.0`` for a non-Legendre basis, which this measure does
    not apply to. The truncation error is taken against a reference
    expansion of at least 64 modes, so the tail beyond the image order is
    measured against a full expansion rather than a truncated one.

    Args:
        model: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``.
        params: Parameter values, in canonical order, at which the
            sensitivities are measured.
        image: The image the fit would run on.
        var: The main variable name, required for a symbolic model.
        param_names: Parameter names for a callable model; see
            :func:`~dtfit.methods._modelinput.resolve_model`.

    Returns:
        The largest relative L2 truncation error, over every parameter,
        of the model's sensitivities at ``image.order``; ``inf`` if that
        error is not finite.

    Raises:
        RuntimeError: the model has no free parameters.
    """
    if image.basis.name != "legendre":
        return 0.0
    spec = resolve_model(model, var, param_names=param_names)
    if not spec.names:
        raise RuntimeError("Model has no free parameters to fit.")
    err = _sensitivity_truncation(
        spec, np.asarray(params, dtype=float), image.domain,
        max(2 * image.order, 64),
    )
    worst = float(np.max(err[:, image.order]))
    return worst if np.isfinite(worst) else float("inf")


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
            ``"auto"`` fits the candidates: the Legendre basis with the
            oscillatory recipe when ``oscillatory`` or ``freq_param`` is
            given or the detrended spectrum has a peak share above 0.3,
            the Legendre basis at its default order, and the block basis
            at its default order; the candidate with the lowest unweighted
            sample RSS (over the Original's own samples, so a robust fit is
            compared like with like) is returned, a later candidate winning
            only when its RSS is lower by more than 0.1 percent. Rejected
            with an Image (``TypeError``), which no longer carries the
            samples the routing needs.
        order: The basis order (polynomial degree for Legendre, window
            count for block). When omitted with an Original, defaults to
            :func:`order_for` at ``p0`` (:func:`osc_order` also, and the
            larger taken, when oscillatory), floored at ``n_params - 1``
            and capped at ``n_obs - 2``; for the block basis, ``4 *
            n_params``. A :class:`~dtfit.image.Basis` instance sets its
            own order; passing ``order`` with one that disagrees raises
            (:func:`~dtfit.image.make_basis`).
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
        oscillatory: Declares the model has a dominant frequency. Raises
            the default order to :func:`osc_order` when that exceeds
            :func:`order_for`; implied by ``freq_param`` and by
            ``basis="auto"`` routing to a cyclic signal.
        freq_param: The name of the frequency parameter to seed from the
            FFT peak of the data (:func:`fft_frequency_seed`) before the
            solve; also sets ``oscillatory=True``. Overwrites ``p0`` for
            that parameter. With an Image the peak is read from
            :meth:`~dtfit.image.Image.reconstruct` on its own grid.
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
        - ``image_order`` and ``basis_name``: the order and basis of the
          image the fit ran on.

    Raises:
        TypeError: ``robust=True``, ``sigma`` given, or ``basis="auto"``
            with an Image; ``data`` is neither an Original nor an Image.
        ValueError: ``freq_param`` names no parameter of the model; the
            image has fewer coefficients than parameters; the model is
            not finite at ``p0`` on the data's grid; ``sigma`` given for
            an Original that already carries weights; a malformed ``p0``
            or ``bounds`` (from the normalizers).
        RuntimeError: the model has no free parameters; every candidate
            basis fails when ``basis="auto"``.

    Warns:
        UserWarning: the image's coverage of the model's sensitivities at
            ``p0`` exceeds ``0.02`` (:func:`coverage`); the order is too
            low to identify the model from this image. Also raised when
            the bounded local solve is poor and a differential-evolution
            search runs to recover it (see :func:`_solve`).

    A model whose value is not finite at a sample raises ``ValueError`` at
    ``p0`` and, once the solve is under way, is instead scored with an
    overflow cost so the optimizer can step away from it; a sensitivity
    that is not finite at isolated samples (an exponent's derivative at
    ``x = 0``, say) is taken as zero there, its analytic limit.
    """
    spec: ModelSpec = resolve_model(model, var, param_names=param_names)
    names = list(spec.names)
    if not names:
        raise RuntimeError("Model has no free parameters to fit.")
    p0_arr = normalize_p0(p0, names)
    bounds_list = normalize_bounds(bounds, names)
    guess = _validate_p0(p0_arr, names)

    oscillatory = bool(oscillatory or freq_param is not None)
    if freq_param is not None and freq_param not in names:
        raise ValueError(
            f"freq_param {freq_param!r} is not a parameter of the model "
            f"(have {names})."
        )
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
        original = None
        if freq_param is not None:
            xg = image.grid.positions()
            seed = fft_frequency_seed(xg, image.reconstruct(xg))
            if seed > 0:
                guess[names.index(freq_param)] = seed
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
        if basis == "auto":
            strength = dominant_period(original.y)[1]
            cands: list[tuple[str, bool]] = []
            if oscillatory or strength > 0.3:
                cands.append(("legendre", True))
            cands.append(("legendre", False))
            cands.append(("block", False))
            best: FittingResult | None = None
            best_rss = float("inf")
            last: Exception | None = None
            for b_name, osc in cands:
                try:
                    cand = fit(
                        model, original, var, basis=b_name, order=order,
                        p0=p0, bounds=bounds,
                        absolute_sigma=absolute_sigma, robust=robust,
                        oscillatory=osc, freq_param=freq_param,
                        param_names=param_names,
                        solver_options=solver_options,
                        random_state=random_state,
                    )
                except (ValueError, RuntimeError, FloatingPointError,
                        np.linalg.LinAlgError) as exc:
                    last = exc
                    continue
                cand_rss = float(
                    np.sum((original.y - cand.predict(original.x)) ** 2)
                )
                if not np.isfinite(cand_rss):
                    continue
                if best is None or cand_rss < best_rss * (1.0 - 1e-3):
                    best, best_rss = cand, cand_rss
            if best is None:
                raise RuntimeError(
                    "basis='auto': every candidate basis failed"
                ) from last
            return best
        if freq_param is not None:
            seed = fft_frequency_seed(original.x, original.y)
            if seed > 0:
                guess[names.index(freq_param)] = seed
        if order is None:
            if not isinstance(basis, str):
                order = basis.order
            else:
                if basis == "block":
                    order = 4 * len(names)
                else:
                    order = order_for(
                        model, guess, original.domain, var=var,
                        param_names=param_names,
                    )
                    if oscillatory:
                        order = max(
                            order, osc_order(original.x, original.y)
                        )
                order = max(order, len(names) - 1, 1)
                order = min(order, original.n - 2)
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
        # A transcendental sensitivity can be singular at an isolated sample
        # while its projection is finite elsewhere: d/dn of x**n is
        # x**n*log(x), NaN at x=0 with limit 0 there. Zeroing that sample
        # before the projection keeps it from poisoning every basis
        # coefficient through 0*nan = nan in the matrix product below.
        cols = [
            Phi.T @ (w * np.where(np.isfinite(d), d, 0.0))
            for d in spec.param_derivs(x, theta)
        ]
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
    if image.basis.name == "legendre":
        cov_err = coverage(model, guess, image, var=var,
                            param_names=param_names)
        if cov_err > 0.02:
            warnings.warn(
                f"image coverage {cov_err:.3f} at order {image.order}: "
                "the model's sensitivities are not represented at this "
                "order; build the image at order_for(model, p0, domain)",
                UserWarning, stacklevel=2,
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
    if original is not None:
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
    result.image_order = image.order
    result.basis_name = image.basis.name
    return result


_LEGACY_LSI = ("filter_data", "alpha", "huber_c")
_LEGACY_EAC = ("active_ratio", "window_mode", "f_scale", "huber_c")


def _drop_legacy(
    legacy: dict[str, Any], allowed: tuple[str, ...], preset: str
) -> None:
    """Reject an unknown keyword and warn on every recognized-but-dead one
    in ``legacy``, the ``**legacy`` catch-all of a preset."""
    for key in legacy:
        if key not in allowed:
            raise TypeError(
                f"{preset}() got an unexpected keyword argument {key!r}"
            )
        warnings.warn(
            f"{preset}(): {key} is no longer used and is ignored; the "
            "image core has no equivalent",
            DeprecationWarning, stacklevel=3,
        )


def fit_lsi(
    data_x: Any, data_y: Any, expr: Any, var: str | None = None, *,
    k_star: Any = None, p0: Any = None, bounds: Any = None, sigma: Any = None,
    absolute_sigma: bool = False, oscillatory: bool = False,
    freq_param: str | None = None, random_state: int | None = 0,
    robust: bool = False, solver_options: dict[str, Any] | None = None,
    nan_policy: str = "raise", param_names: Any = None, **legacy: Any,
) -> FittingResult:
    """LSI: :func:`fit` in the Legendre basis. ``k_star`` is the order;
    ``None`` or ``"auto"`` takes the default from :func:`order_for`.

    Builds an :class:`Original` from ``(data_x, data_y, sigma, nan_policy)``
    and calls :func:`fit` on it with ``basis="legendre"``; every parameter
    below not listed here (``model``, canonical parameter order, the
    covariance and coverage rules) behaves exactly as documented there.

    Parameters:
        data_x, data_y: Observed samples. A pandas ``Series`` or single-
            column ``DataFrame`` is accepted (:class:`Original`).
        expr: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``; see :func:`fit`.
        var: The main variable name, required for a symbolic model.
        k_star: The Legendre order. ``None`` or ``"auto"`` both take the
            order-rule default (:func:`order_for`, raised for
            ``oscillatory`` per :func:`osc_order`); an int sets it
            explicitly and must leave at least as many coefficients as
            parameters.
        p0: Initial guess, positional in canonical parameter order or a
            ``{name: value}`` mapping; ``None`` defaults to ones.
        bounds: Per-parameter bounds; see :func:`fit`.
        sigma: Per-sample standard deviations, the same length as
            ``data_y``; builds the Original's weights. ``None`` weights
            every sample equally.
        absolute_sigma: If ``False`` (default) the covariance is scaled by
            the residual variance; if ``True``, not, as in
            ``scipy.optimize.curve_fit``.
        oscillatory: Raises the default order to :func:`osc_order` when
            that exceeds :func:`order_for`; implied by ``freq_param``.
        freq_param: Name of the frequency parameter to seed from the data's
            FFT peak (:func:`fft_frequency_seed`) before the solve; implies
            ``oscillatory=True``.
        random_state: Seed for the bounded global stage's differential
            evolution; ``None`` is nondeterministic between calls.
        robust: Huber-reweight the image when it is built from the
            samples.
        solver_options: Forwarded to the least-squares stage; see
            :func:`fit`.
        nan_policy: ``"raise"`` (default) rejects a non-finite sample;
            ``"omit"`` drops it before fitting (:class:`Original`).
        param_names: Parameter names for a callable model, in signature
            order; cross-checked for a symbolic model.
        **legacy: Retired keywords accepted for source compatibility and
            ignored: ``filter_data``, ``alpha``, ``huber_c``. Any other
            keyword raises ``TypeError``.

    Returns:
        A :class:`~dtfit.types.FittingResult`; see :func:`fit`.

    Raises:
        TypeError: an unrecognized keyword in ``**legacy``.
        ValueError: ``freq_param`` names no parameter of the model; the
            image has fewer coefficients than parameters; the model is not
            finite at ``p0``; a malformed ``p0``, ``bounds`` or ``sigma``;
            multivariate ``data_x`` or ``data_y`` (:class:`Original`).
        RuntimeError: the model has no free parameters.

    Warns:
        DeprecationWarning: a recognized legacy keyword was passed (see
            ``**legacy``).
        UserWarning: the image's coverage of the model's sensitivities at
            ``p0`` is poor, or the order is too low to identify the model;
            see :func:`fit`.
    """
    _drop_legacy(legacy, _LEGACY_LSI, "fit_lsi")
    order = None if k_star is None or k_star == "auto" else int(k_star)
    original = Original(data_x, data_y, sigma=sigma, nan_policy=nan_policy)
    return fit(
        expr, original, var, basis="legendre", order=order, p0=p0,
        bounds=bounds, absolute_sigma=absolute_sigma, robust=robust,
        oscillatory=oscillatory, freq_param=freq_param,
        param_names=param_names, solver_options=solver_options,
        random_state=random_state,
    )


def fit_eac(
    data_x: Any, data_y: Any, expr: Any, var: str | None = None, *,
    n_windows: int | None = None, p0: Any = None, bounds: Any = None,
    sigma: Any = None, absolute_sigma: bool = False, robust: bool = False,
    loss: str = "linear", solver_options: dict[str, Any] | None = None,
    nan_policy: str = "raise", param_names: Any = None, **legacy: Any,
) -> FittingResult:
    """EAC: :func:`fit` in the block basis with ``n_windows`` windows
    (default four per parameter). A ``loss`` other than ``"linear"``
    selects the robust image, the one outlier defence that remains.

    Builds an :class:`Original` from ``(data_x, data_y, sigma, nan_policy)``
    and calls :func:`fit` on it with ``basis="block"``; every parameter
    below not listed here behaves exactly as documented there.

    Parameters:
        data_x, data_y: Observed samples. A pandas ``Series`` or single-
            column ``DataFrame`` is accepted (:class:`Original`).
        expr: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``; see :func:`fit`.
        var: The main variable name, required for a symbolic model.
        n_windows: Number of block windows. ``None`` defaults to
            ``4 * n_params``; an int sets it explicitly and must leave at
            least as many coefficients as parameters.
        p0: Initial guess, positional in canonical parameter order or a
            ``{name: value}`` mapping; ``None`` defaults to ones.
        bounds: Per-parameter bounds; see :func:`fit`.
        sigma: Per-sample standard deviations, the same length as
            ``data_y``; builds the Original's weights. ``None`` weights
            every sample equally.
        absolute_sigma: If ``False`` (default) the covariance is scaled by
            the residual variance; if ``True``, not, as in
            ``scipy.optimize.curve_fit``.
        robust: Huber-reweight the image when it is built from the
            samples. Also set to ``True`` when ``loss`` is not
            ``"linear"``.
        loss: ``"linear"`` (default) leaves ``robust`` as given; any other
            value is the retired robust-loss selector and now sets
            ``robust=True`` instead, with a warning.
        solver_options: Forwarded to the least-squares stage; see
            :func:`fit`.
        nan_policy: ``"raise"`` (default) rejects a non-finite sample;
            ``"omit"`` drops it before fitting (:class:`Original`).
        param_names: Parameter names for a callable model, in signature
            order; cross-checked for a symbolic model.
        **legacy: Retired keywords accepted for source compatibility and
            ignored: ``active_ratio``, ``window_mode``, ``f_scale``,
            ``huber_c``. Any other keyword raises ``TypeError``.

    Returns:
        A :class:`~dtfit.types.FittingResult`; see :func:`fit`.

    Raises:
        TypeError: an unrecognized keyword in ``**legacy``.
        ValueError: the image has fewer coefficients than parameters; the
            model is not finite at ``p0``; a malformed ``p0``, ``bounds``
            or ``sigma``; multivariate ``data_x`` or ``data_y``
            (:class:`Original`).
        RuntimeError: the model has no free parameters.

    Warns:
        DeprecationWarning: a recognized legacy keyword was passed, or
            ``loss`` is not ``"linear"`` (see ``loss``).
    """
    _drop_legacy(legacy, _LEGACY_EAC, "fit_eac")
    if loss != "linear":
        warnings.warn(
            "fit_eac(): loss is replaced by robust=True (a robust "
            "image); using it",
            DeprecationWarning, stacklevel=2,
        )
        robust = True
    original = Original(data_x, data_y, sigma=sigma, nan_policy=nan_policy)
    order = None if n_windows is None else int(n_windows)
    return fit(
        expr, original, var, basis="block", order=order, p0=p0,
        bounds=bounds, absolute_sigma=absolute_sigma, robust=robust,
        param_names=param_names, solver_options=solver_options,
    )
