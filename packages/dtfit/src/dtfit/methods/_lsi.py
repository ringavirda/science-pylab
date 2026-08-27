"""LSI: least squares integral.

Numeric successor to the symbolic DSBI method. Fits a model that is nonlinear
in its parameters by minimizing the integral least-squares discrepancy between
the data and the model over the observation interval.

Reconditioned formulation
-------------------------
Matching monomial Maclaurin coefficients under an integral weighting gives
``M[i,j] = w_i w_j H^(i+j+1)/(i+j+1)``, a weighted Hilbert matrix whose
condition number explodes with the order; reading the empirical spectrum off a
raw-Vandermonde ``numpy.polyfit`` is ill-conditioned for the same reason.
Working in an orthogonal (Legendre) basis on the data interval avoids both:

* the empirical spectrum is the Legendre fit of the data, a well-conditioned
  least squares on an orthogonal basis rather than a raw Vandermonde;
* the model spectrum is the model's Legendre projection, evaluated by
  Gauss-Legendre quadrature, which integrates the model exactly instead of
  truncating its Taylor series;
* orthogonality on the interval turns the integral criterion
  ``∫ (data - model)^2 dt`` into a diagonal sum of squared coefficient
  residuals ``Σ_j H/(2j+1) (β_j^data - β_j^model)^2``. The Hilbert matrix
  collapses to a diagonal weight and the problem is perfectly conditioned.

This works directly on raw ``(x, y)`` data and returns a parameter covariance
estimate alongside the coefficients.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import sympy as sp
from numpy.polynomial import Legendre
from scipy.optimize import least_squares

from dtfit.log import echo
from dtfit._core._kernels import legendre_project
from dtfit._core._spectral import solve_weighted_nlls
from dtfit._pandas import is_dataframe, is_series, to_1d_array
from dtfit.types import FittingResult, InitialGuess
from ._modelinput import resolve_model, result_kwargs
from ._common import (
    _covariance, information_criteria, _validate_xy, _validate_p0,
    _resolve_sigma, _savgol_prefilter, normalize_p0, normalize_bounds,
)


def fft_frequency_seed(x: np.ndarray, y: np.ndarray) -> float:
    """Dominant angular frequency of ``y`` over the uniform grid ``x``.

    The peak of the mean-removed real FFT, returned as ``2*pi*f``. This seeds
    the angular-frequency parameter of an oscillatory model, and is the
    ``freq_param`` initial guess behind :func:`fit_lsi`'s oscillatory recipe:
    without it a smoothed low-order spectral fit cannot recover a sinusoid's
    frequency. Assumes near-uniform sampling and reads the spacing from
    ``x[1] - x[0]``.
    """
    yy = np.asarray(y, dtype=float) - float(np.mean(y))
    spec = np.abs(np.fft.rfft(yy))
    if spec.size:
        spec[0] = 0.0  # ignore the DC bin
    freqs = np.fft.rfftfreq(yy.size, d=float(x[1] - x[0]))
    return 2.0 * np.pi * float(freqs[int(np.argmax(spec))])


def _osc_order(x: np.ndarray, y: np.ndarray, max_order: int = 200) -> int:
    """Spectral order high enough to resolve the dominant cycle.

    A degree-``k`` polynomial only starts to represent a sinusoid once ``k``
    exceeds its mapped wavenumber ``pi * cycles``, the classical
    Legendre/Chebyshev resolution threshold. Below it the data and model
    spectra both collapse toward zero and the amplitude is left
    under-constrained; ``pi * cycles`` plus headroom clears the threshold at
    every cycle count.
    """
    w0 = fft_frequency_seed(x, y)
    cycles = w0 * float(x[-1] - x[0]) / (2.0 * np.pi)
    return min(int(np.ceil(np.pi * cycles)) + 8, max_order)


def _auto_order(x: np.ndarray, y: np.ndarray, max_order: int = 8) -> int:
    """Pick the Legendre degree that minimizes BIC of the data fit: enough
    spectral resolution to capture the signal without fitting noise."""
    n = y.size
    best_order, best_bic = 1, np.inf
    for k in range(1, min(max_order, n - 1) + 1):
        resid = y - Legendre.fit(x, y, k)(x)
        rss = float(resid @ resid)
        if rss <= 0:
            return k
        _, bic = information_criteria(rss, n, k + 1)
        if bic < best_bic:
            best_order, best_bic = k, bic
    return best_order


def fit_lsi(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str | sp.Expr | Callable[..., Any],
    var: str | None = None,
    *,
    param_names: Sequence[str] | None = None,
    k_star: int | str | None = None,
    alpha: float = 0.0,
    filter_data: bool = False,
    bounds: (
        Sequence[tuple[float, float]]
        | Mapping[str, tuple[float, float]]
        | tuple[Any, Any]
        | None
    ) = None,
    p0: InitialGuess | Mapping[str, float] = None,
    sigma: np.ndarray | None = None,
    absolute_sigma: bool = False,
    oscillatory: bool = False,
    freq_param: str | None = None,
    random_state: int | None = 0,
    robust: bool = False,
    huber_c: float = 3.0,
    solver_options: dict[str, Any] | None = None,
    nan_policy: str = "raise",
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` with the integral least-squares
    criterion in the reconditioned (Legendre) differential-transformation
    scheme.

    Args:
        data_x, data_y: Observed samples.
        expr: The model, in any of three equivalent forms resolved by
            :func:`dtfit.methods.resolve_model`: a SymPy-expression string
            such as ``"a0 + a1*x + a2*exp(a3*x)"``, a :class:`sympy.Expr`, or
            a plain Python callable ``f(x, *params)``. Symbolic models lay
            their parameters out alphabetically; a callable follows its
            signature order, the parameters after the leading ``x``. A
            callable carries no expression: the returned
            :class:`~dtfit.types.FittingResult` cannot be serialized with
            ``to_dict``, though ``predict`` and its error bands still work
            through the numeric evaluator.
        var: Main variable name in ``expr``. Required for a symbolic model,
            as in :func:`fit_eac`. For a callable it is a label only, naming
            the result's variable and defaulting to ``"x"``.
        param_names: Parameter names. For a callable they set or override the
            names, in signature order, when the signature cannot be
            introspected; for a symbolic model they are optional and
            validated against the names parsed from the expression. See
            :func:`dtfit.methods.resolve_model`.
        k_star: Number of spectral coefficients (Legendre order) to match. An
            int sets the order explicitly; ``"auto"`` selects it by BIC of the
            data fit; ``None`` (the default) uses order 5, or under the
            oscillatory recipe an order high enough to resolve the dominant
            cycle (:func:`_osc_order`).
        alpha: Extra exponential down-weight ``exp(-alpha*j)`` on high-order
            coefficients, on top of the built-in ``1/(2j+1)`` orthonormal
            weight. Defaults to 0, the orthogonal basis already taming high
            orders.
        filter_data: Apply a Savitzky-Golay pre-filter to the data before the
            spectral projection. Off by default, since a fitter must not
            silently modify the caller's data. Worth turning on for very
            noisy telemetry.
        bounds: Optional parameter bounds: a per-parameter ``(min, max)`` pair
            list in sorted-name order, a partial ``{name: (min, max)}`` dict
            where unnamed parameters stay unbounded, or a scipy-style
            ``(lo, hi)`` 2-tuple (see
            :func:`dtfit.methods.normalize_bounds`). When given, a global
            search (differential evolution) runs before local refining if
            every bound is finite and a supplied ``p0`` does not already
            yield a converged bounded local fit that explains the spectrum
            (see :func:`dtfit._core._spectral.solve_weighted_nlls`).
            Otherwise the bounds only constrain the local solve.
        p0: Optional initial guess (defaults to ones): a sequence in
            sorted-name order or a full ``{name: value}`` dict.
        sigma: Optional per-sample standard deviations, one per ``(x, y)``
            sample. The empirical Legendre spectrum then becomes a weighted
            fit with weights ``1/sigma``, which is how to down-weight a
            corrupted region without dropping it. Entries must be finite and
            strictly positive, and the array must carry the same length as
            the raw ``data_y``; under ``nan_policy="omit"`` the dropped
            ``(x, y)`` rows are dropped from ``sigma`` too, exactly as
            :func:`fit_eac` does. ``None`` (the default) weights every sample
            equally.
        absolute_sigma: How ``sigma`` scales the covariance, mirroring
            :func:`scipy.optimize.curve_fit`. ``False`` (the default) reads
            ``sigma`` as relative and rescales the covariance by the reduced
            chi-square, leaving the standard errors unchanged when every
            ``sigma`` is multiplied by a constant. ``True`` reads ``sigma``
            as carrying absolute units: the covariance reflects them
            directly, and scaling every ``sigma`` by ``k`` scales the
            standard errors by ``k``. No effect when ``sigma`` is ``None``.
        oscillatory: Apply the oscillatory recipe. A smoothed, low-order
            spectral fit erases a cycle, so this forces ``filter_data=False``
            and, unless ``k_star`` is given explicitly, raises the spectral
            order to resolve the dominant cycle (:func:`_osc_order`). Across
            the forecasting and parameter-estimation domain studies a
            sinusoid recovers to within 1% under this recipe against roughly
            50% without it. Passing ``freq_param`` implies the recipe; pass
            ``oscillatory=True`` to enable it without naming a frequency
            parameter.
        freq_param: Name of the model's angular-frequency parameter. Its
            initial guess is then seeded from the data's FFT peak
            (:func:`fft_frequency_seed`), which is what the local (no-bounds)
            solve needs to lock onto the right cycle. Implies the oscillatory
            recipe. Symbolic models and callables behave identically here
            because the recipe is purely data-driven: it forces
            ``filter_data=False``, raises the spectral order via
            :func:`_osc_order` and seeds ``freq_param`` from the FFT, with no
            symbolic manipulation of the model's frequency dependence.
        random_state: Seed for the global (differential-evolution) search on
            the bounded path, for reproducibility; ``None`` draws from
            NumPy's global RNG and is non-deterministic. Defaults to ``0``.
            No effect on the unbounded local solve, which is already
            deterministic.
        robust: If True, robustify the empirical spectrum by IRLS: each
            sample's residual to the current model is winsorized within
            ``huber_c`` robust sigmas (MAD) before re-projecting, keeping an
            outlier sample from distorting the Legendre coefficients. Forces
            ``filter_data=False``, a linear pre-smoother otherwise smearing
            outliers first. Costs a handful of re-solves and needs no tuning.
        huber_c: Robust winsorization threshold in residual sigmas for
            ``robust=True`` (~3 leaves clean samples untouched).
        solver_options: Optional dict of solver tolerances forwarded to the
            underlying optimizers (``xtol`` / ``ftol`` / ``gtol`` /
            ``max_nfev`` where applicable; unknown keys are ignored). See
            :func:`dtfit._core._spectral.solve_weighted_nlls`.
        nan_policy: Non-finite handling passed to the ``(x, y)`` validator,
            ``"raise"`` or ``"omit"``; see
            :func:`dtfit.methods._common._validate_xy`.

    Returns:
        FittingResult with the fitted coefficients, callable model and a
        parameter covariance estimate. Fit-quality diagnostics (``rsquared``,
        ``aic``/``bic``, ``nfev``, ``cost``) are recorded when computable.
    """
    spec = resolve_model(expr, var, param_names=param_names)
    names = list(spec.names)
    if not names:
        raise RuntimeError("Model expression has no free parameters to fit.")

    p0_arr = normalize_p0(p0, names)
    bounds_list = normalize_bounds(bounds, names)

    # Coerce pandas Series / single-column DataFrame input to plain 1-D float
    # arrays. Gated on the pandas types: a raw ndarray or list passes through
    # untouched. The coerced arrays feed ``_resolve_sigma`` as well.
    if is_series(data_x) or is_dataframe(data_x):
        data_x = to_1d_array(data_x, "data_x")
    if is_series(data_y) or is_dataframe(data_y):
        data_y = to_1d_array(data_y, "data_y")

    x, y = _validate_xy(data_x, data_y, nan_policy=nan_policy)
    # Keep the unfiltered samples for the fit-quality stats below; ``y`` may
    # be replaced by the Savitzky-Golay pre-filter further down.
    y_orig = y.copy()

    # Optional per-sample weights (1/sigma) for the empirical spectrum. The
    # weighted Legendre fit normalizes out any uniform scaling of the weights,
    # which leaves the relative down-weighting in the empirical spectrum while
    # a single representative scale ``s_rep`` (the RMS sigma) folds into the
    # criterion weight. That is what carries the absolute noise level into the
    # covariance: under ``absolute_sigma=True``, scaling every sigma by k
    # scales the SEs by k.
    w_sig: np.ndarray | None = None
    s_rep = 1.0
    sig = _resolve_sigma(sigma, data_x, data_y, x, nan_policy)
    if sig is not None:
        w_sig = 1.0 / sig
        s_rep = float(np.sqrt(np.mean(sig * sig)))

    oscillatory = oscillatory or freq_param is not None
    if oscillatory:
        # The cycle lives in the high-order spectrum and smoothing destroys
        # it. Raise the order only when the caller left ``k_star`` unset.
        filter_data = False
        if k_star is None:
            k_star = _osc_order(x, y)
    if robust:
        # Winsorization (below) pulls each sample toward the model. A linear
        # pre-smoother would smear an outlier across its neighbours first.
        filter_data = False

    # 1. Optional smoothing to tame noise before the spectral projection.
    if filter_data:
        y = _savgol_prefilter(y)

    if k_star is None:
        order = 5  # the conditioned default Legendre order
    elif k_star == "auto":
        order = _auto_order(x, y)
    else:
        order = int(k_star)
    # The spectral residual has order+1 entries and must carry at least as
    # many equations as parameters, or the least squares is underdetermined
    # and LM raises. Flooring the order at n_params-1 keeps every catalogue
    # model solvable at the default k_star: an 8-parameter Fourier series
    # needs order >= 7.
    n_params = len(names)
    order = max(order, n_params - 1)
    order = max(1, min(order, y.size - 2))
    if order + 1 < n_params:
        # The sample count clamped the order below n_params-1: the spectral
        # residual now carries fewer equations than parameters, and LM would
        # raise a cryptic 'm < n'. Mirrors EAC's 2*n_params guard.
        raise ValueError(
            f"fit_lsi needs at least {n_params + 1} samples to fit "
            f"{n_params} parameters (got {y.size}); the spectral match "
            "would be underdetermined."
        )
    echo(f"LSI Legendre spectral order: {order}")

    x0, xn = float(x[0]), float(x[-1])
    h = xn - x0
    domain = [x0, xn]

    # 2. Empirical spectrum: Legendre coefficients of the data. With ``sigma``
    #    the fit is weighted by ``1/sigma``, and noisy samples then contribute
    #    less to the empirical coefficients.
    beta_data = Legendre.fit(x, y, order, domain=domain, w=w_sig).coef

    # 3. Model spectrum via Gauss-Legendre quadrature (model integrated exactly).
    n_quad = max(2 * (order + 1), 16)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    t_quad = x0 + h * (nodes + 1.0) / 2.0
    # P_j(node) for j = 0..order:  legvander gives columns of Legendre polys.
    legvander = np.polynomial.legendre.legvander(nodes, order)  # (n_quad, order+1)
    norm = (2.0 * np.arange(order + 1) + 1.0) / 2.0  # standard-Legendre scaling

    def model_spectrum(c: np.ndarray) -> np.ndarray:
        fv = spec.eval(t_quad, c)
        return legendre_project(fv, weights, legvander, norm)

    # 4. Diagonal orthonormal weight: ∫(data-model)^2 dt = Σ H/(2j+1) Δβ_j^2,
    #    with an optional extra exponential down-weight. Under ``sigma`` the
    #    whole criterion is divided by the representative noise scale
    #    ``s_rep``. Being a scalar, that changes only the absolute-sigma
    #    covariance scale, not the fit.
    j = np.arange(order + 1)
    sqrt_w = np.sqrt((h / (2.0 * j + 1.0)) * np.exp(-alpha * j)) / s_rep

    def residual(c: np.ndarray) -> np.ndarray:
        mspec = model_spectrum(c)
        if not np.all(np.isfinite(mspec)):
            return np.full(order + 1, 1e6)
        return sqrt_w * (beta_data - mspec)

    guess = _validate_p0(p0_arr, names)

    # Seed the angular-frequency parameter from the data's FFT peak so the
    # local solve locks onto the right cycle. The bounds path runs a global
    # search and ignores the seed; there the bounds bracket the frequency.
    if freq_param is not None:
        if freq_param not in names:
            raise ValueError(
                f"freq_param {freq_param!r} is not a parameter of the model "
                f"(have {names})."
            )
        guess[names.index(freq_param)] = fft_frequency_seed(x, y)

    # 5. Solve through the shared weighted-NLLS driver, the same bounded-local
    #    / global-DE-fallback / unbounded-LM logic the map-reduce fitters use.
    #    Without bounds this is a weighted LM from the seed. With bounds it
    #    tries a bounded local solve first, which from a data-driven seed (an
    #    FFT peak for the frequency, say) costs 10-50x less than the
    #    differential-evolution fallback that guards the multimodal cases.
    coeffs, jac, converged, message, nfev = solve_weighted_nlls(
        residual, sqrt_w, beta_data, guess, p0=p0_arr, bounds=bounds_list,
        seed=random_state, solver_options=solver_options,
    )

    if robust:
        # Robust integral by IRLS: winsorize each sample's residual to the
        # current model within huber_c robust sigmas, then recompute the
        # empirical Legendre spectrum, so an outlier sample cannot distort the
        # projection. ``beta_data`` is reassigned here and the ``residual``
        # closure reads it lazily.
        inner_ok, inner_message = converged, message
        # Give the re-solves the tolerances the main solve honoured, keeping
        # only the keys ``scipy.optimize.least_squares`` accepts.
        opts = solver_options or {}
        ls_opts = {
            k: opts[k]
            for k in ("xtol", "ftol", "gtol", "max_nfev")
            if k in opts
        }
        for _ in range(3):
            mv = spec.eval(x, coeffs)
            r = y - mv
            med = float(np.median(r))
            mad_sigma = 1.4826 * float(np.median(np.abs(r - med)))
            if mad_sigma <= 0.0:
                break
            clip = huber_c * mad_sigma
            y_eff = mv + (med + np.clip(r - med, -clip, clip))
            beta_data = Legendre.fit(x, y_eff, order, domain=domain, w=w_sig).coef
            if bounds_list is not None:
                sol = least_squares(residual, coeffs,
                                    bounds=([b[0] for b in bounds_list],
                                            [b[1] for b in bounds_list]),
                                    method="trf", **ls_opts)
            else:
                sol = least_squares(residual, coeffs, method="lm", **ls_opts)
            coeffs = np.asarray(sol.x, dtype=np.float64)
            jac = sol.jac
            nfev += int(getattr(sol, "nfev", 0))
            inner_ok, inner_message = bool(sol.success), str(sol.message)
        converged = inner_ok
        message = f"robust IRLS ({inner_message})"

    echo("LSI fitted coefficients:", coeffs)
    final_res = residual(coeffs)
    cov = _covariance(jac, final_res, n_params, absolute_sigma=absolute_sigma)

    # Fit-quality stats go against the original samples: smoothing and robust
    # winsorization changed the projected data, not the observations the fit
    # is judged on.
    model_at_x = spec.eval(x, coeffs)
    resid_x = y_orig - model_at_x
    rss = float(resid_x @ resid_x)
    tss = float(np.sum((y_orig - float(np.mean(y_orig))) ** 2))
    cost = 0.5 * float(final_res @ final_res)

    # A symbolic model keeps the lazy-lambdify path (expr/var). A callable has
    # no expression, so ``result_kwargs`` sends a bound f(x) closure and the
    # params-explicit evaluator instead.
    return FittingResult(coeffs=coeffs, cov=cov,
                         converged=converged, message=message,
                         x_range=(float(np.min(x)), float(np.max(x))),
                         n_obs=x.size, rss=rss, tss=tss, nfev=nfev, cost=cost,
                         **result_kwargs(spec, coeffs))
