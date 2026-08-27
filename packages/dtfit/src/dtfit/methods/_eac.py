"""EAC: equal-areas criterion fitting.

Numeric successor to the symbolic DSBE method. Identifies model parameters by
matching integral areas of the model and the data over a set of windows rather
than balancing differential spectra. Integration smooths noise, which makes
this markedly more robust than spectral or derivative-based approaches, and it
works directly on raw ``(x, y)`` data.

The active region is split into ``n_windows >= n`` windows, ``2n`` by default,
giving an overdetermined area-matching system solved by Levenberg-Marquardt or
trust-region least squares with an analytic (integrated) Jacobian. More
equations than unknowns means the random per-window integration errors partly
cancel, and a parameter covariance can be estimated from the residual
Jacobian.

Windows are placed either ``"uniform"``, equal spans over the active region,
or ``"curvature"``, edges carrying roughly equal cumulative absolute curvature:
narrow where the signal bends, wide where it is smooth. Curvature placement
conditions the system better for signals with a localized transient, a step
take-off or a sharp turn or a peak's rise, and the parameter-estimation domain
study ranked it best on concentrated transients and rational-saturating shapes
(Michaelis-Menten / Hill).

Two outlier defences are available and they compose. A robust least-squares
loss (``loss=`` / ``f_scale=``) down-weights contaminated window-area residuals
within a single fit, the mechanism characterised in the EAC paper. Under
heavier contamination :func:`dtfit.ensemble_fit` aggregates fits over
overlapping windows by median, rejecting whole corrupted windows without
per-problem ``f_scale`` tuning.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

import numpy as np
import sympy as sp
from scipy.integrate import simpson
from scipy.optimize import least_squares

from dtfit.log import echo
from dtfit._core._kernels import simpson_windows, simpson_windows_rows
from dtfit._pandas import is_dataframe, is_series, to_1d_array
from dtfit.types import FittingResult, InitialGuess
from ._common import (
    _covariance, _validate_xy, _validate_p0, _resolve_sigma,
    normalize_p0, normalize_bounds,
)
from ._modelinput import resolve_model, result_kwargs


def _dominant_cycles(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate how many full periods of the dominant tone the record spans.

    Reads the FFT peak above DC. Non-oscillatory shapes (a trend, a single
    peak, a sigmoid) concentrate their energy at or near DC and return ~0-1;
    a genuine oscillation returns roughly its cycle count. ``fit_eac`` uses
    this to auto-scale the window count so windows stay sub-period.
    """
    if x.size < 8:
        return 0.0
    yd = np.asarray(y, dtype=float) - float(np.mean(y))
    duration = float(x[-1] - x[0])
    if duration <= 0.0 or not np.any(yd):
        return 0.0
    freqs = np.fft.rfftfreq(x.size, d=duration / (x.size - 1))
    ps = np.abs(np.fft.rfft(yd))
    if ps.size < 2:
        return 0.0
    k_peak = 1 + int(np.argmax(ps[1:]))  # skip the DC bin
    return float(freqs[k_peak] * duration)


def _curvature_edges(x: np.ndarray, y: np.ndarray, m: int) -> np.ndarray:
    """Index edges so each window holds ~equal cumulative |curvature|."""
    d2 = np.abs(np.gradient(np.gradient(y, x), x))
    d2 = d2 + 1e-9  # floor so flat regions still get covered
    cum = np.concatenate([[0.0], np.cumsum(d2)])
    cum /= cum[-1]
    targets = np.linspace(0, 1, m + 1)
    edges = np.searchsorted(cum, targets)
    edges[0], edges[-1] = 0, x.size
    # enforce >= 3 samples per window for Simpson
    for k in range(1, m + 1):
        if edges[k] - edges[k - 1] < 3:
            edges[k] = min(edges[k - 1] + 3, x.size)
    return np.unique(edges)


def _place_windows(
    x: np.ndarray,
    y: np.ndarray,
    n: int,
    *,
    window_mode: str,
    active_ratio: float,
    n_windows: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Place the integration windows and integrate the data areas.

    Returns ``(x_active, y_active, starts, stops, data_areas, m)``.
    ``"curvature"`` placement falls back to ``"uniform"`` when it would leave
    fewer than ``n`` windows, which scipy would otherwise report as a cryptic
    ``m < n``. ``"uniform"`` clamps to ``>= n`` and auto-scales the window
    count on oscillatory data to keep windows sub-period.
    """
    if window_mode == "curvature":
        # Information-adaptive edges over all the data: each window carries
        # roughly equal cumulative |curvature| (>= 3 samples for Simpson).
        x_active = np.ascontiguousarray(x)
        m = max(n, 2 * n if n_windows is None else int(n_windows))
        edges = _curvature_edges(x_active, y, m)
        starts = np.ascontiguousarray(edges[:-1], dtype=np.intp)
        stops = np.ascontiguousarray(edges[1:], dtype=np.intp)
        if starts.size >= n:
            y_active = np.ascontiguousarray(y)
            data_areas = simpson_windows(y_active, x_active, starts, stops)
            return x_active, y_active, starts, stops, data_areas, int(starts.size)
        echo(
            "EAC: curvature placement underdetermined "
            f"({starts.size} windows < {n} params); falling back to uniform"
        )
    # Uniform spans over the leading active region (>= n for solvability, default
    # 2n for redundancy), each with at least 3 samples for Simpson.
    idx_max = max(int(x.size * active_ratio), n + 1)
    requested = 2 * n if n_windows is None else int(n_windows)
    if n_windows is None:
        # Oscillatory data: a window spanning whole periods integrates to ~0, so
        # its area is blind to amplitude/phase and the criterion loses
        # conditioning as the cycle count grows. Auto-scale to keep windows
        # sub-period (~3 per dominant cycle). Non-oscillatory shapes have their
        # FFT peak at/near DC (cycles ~ 0-1) and stay at 2n.
        cycles = _dominant_cycles(x[:idx_max], y[:idx_max])
        if cycles >= 2.0:
            requested = max(requested, int(np.ceil(3.0 * cycles)))
    m = max(n, min(requested, idx_max // 3))
    window = max(idx_max // m, 2)
    x_active = np.ascontiguousarray(x[:idx_max])
    y_active = np.ascontiguousarray(y[:idx_max])
    starts = np.array([i * window for i in range(m)], dtype=np.intp)
    stops = np.array(
        [(i + 1) * window if i < m - 1 else idx_max for i in range(m)],
        dtype=np.intp,
    )
    data_areas = simpson_windows(y_active, x_active, starts, stops)
    return x_active, y_active, starts, stops, data_areas, m


def _area_weights(
    x_active: np.ndarray,
    sigma_active: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
) -> np.ndarray:
    """Per-window least-squares weights from per-sample ``sigma``.

    A window's area is a fixed linear combination of its samples, the
    composite Simpson quadrature weights ``s_i``, obtained here by integrating
    the unit vectors so they match :func:`simpson_windows` exactly. With
    independent per-sample noise of std ``sigma_i`` the area variance is
    ``sum_i (s_i * sigma_i)**2`` and the window weight is the inverse area
    standard deviation ``1 / sqrt(area_var)``. Multiplying each area residual
    by it turns the area-matching system into weighted least squares, which is
    what makes ``absolute_sigma`` read exactly as in
    :func:`scipy.optimize.curve_fit`.
    """
    w = np.empty(starts.size, dtype=float)
    for k in range(starts.size):
        a, b = int(starts[k]), int(stops[k])
        xw = np.ascontiguousarray(x_active[a:b], dtype=float)
        length = xw.size
        quad = np.empty(length, dtype=float)
        unit = np.zeros(length, dtype=float)
        for j in range(length):
            unit[j] = 1.0
            quad[j] = simpson(y=unit, x=xw)
            unit[j] = 0.0
        area_var = float(np.sum((quad * sigma_active[a:b]) ** 2))
        w[k] = 1.0 / np.sqrt(area_var) if area_var > 0.0 else 0.0
    return w


def fit_eac(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str | sp.Expr | Callable[..., Any],
    var: str | None = None,
    *,
    active_ratio: float = 1.0,
    n_windows: int | None = None,
    window_mode: str = "uniform",
    bounds: (
        Sequence[tuple[float, float]]
        | Mapping[str, tuple[float, float]]
        | tuple[Any, Any]
        | None
    ) = None,
    loss: str = "linear",
    f_scale: float | None = None,
    robust: bool = False,
    huber_c: float = 3.0,
    p0: InitialGuess | Mapping[str, float] = None,
    sigma: np.ndarray | Sequence[float] | None = None,
    absolute_sigma: bool = False,
    solver_options: Mapping[str, Any] | None = None,
    param_names: Sequence[str] | None = None,
    nan_policy: str = "raise",
) -> FittingResult:
    """Fit ``expr`` to ``(data_x, data_y)`` with the equal-areas criterion.

    Args:
        data_x, data_y: Observed samples.
        expr: The model, in any of three equivalent forms resolved by
            :func:`dtfit.methods.resolve_model`: a SymPy expression string
            such as ``"a * atan(w * x)"``, a :class:`sympy.Expr`, or a plain
            Python callable ``f(x, *params)``. A symbolic model differentiates
            exactly for its area Jacobian; a callable is forward-differenced.
            The canonical parameter order is sorted-by-name for a symbolic
            model and signature order for a callable, and it is the layout of
            ``coeffs``, ``p0``, ``bounds``, the covariance and
            ``result.names``.
        var: Main variable name in ``expr``. Required for a symbolic model; a
            label only for a callable, where it defaults to ``"x"``.
        active_ratio: Fraction of the leading data used for window placement
            under ``window_mode="uniform"``. Defaults to ``1.0``, all samples:
            the fitter must not silently discard trailing data.
            ``active_ratio=0.8`` is the tuned recipe for signals whose
            informative transient leads, a step take-off or a saturating rise,
            concentrating the windows on that transient and dropping the flat
            tail. Ignored by ``"curvature"`` placement, which spans all the
            data.
        n_windows: Number of integration windows (area equations). Defaults to
            ``2 * n_params`` for an overdetermined, noise-averaging fit. Must
            be ``>= n_params``; clamped so each window keeps at least 3
            samples.
        window_mode: Window placement. ``"uniform"`` gives equal spans over
            the active region; ``"curvature"`` places edges carrying equal
            cumulative absolute curvature, narrow where the signal bends and
            wide where it is smooth. ``"curvature"`` conditions the system
            better for signals with a localized transient (step take-off,
            sharp turn, a peak's rise), and the parameter-estimation domain
            study ranked it best on concentrated transients and
            rational-saturating shapes.
        bounds: Optional parameter bounds: a per-parameter ``(min, max)`` pair
            list in sorted-name order (the canonical form), a partial
            ``{name: (min, max)}`` dict where unnamed parameters stay
            unbounded, or a scipy-style ``(lower, upper)`` 2-tuple (see
            :func:`dtfit.methods.normalize_bounds` and its documented
            2-parameter ambiguity rule). Any bounds switch the solver to
            trust-region.
        loss: Least-squares loss, ``"linear"`` or ``"soft_l1"`` for outlier
            robustness as in the EAC paper. The loss acts on the window-area
            residuals and down-weights whole contaminated windows, so give it
            enough windows (``n_windows``) that outliers stay localized. Where
            contamination is dense and ``f_scale`` hard to tune,
            :func:`dtfit.ensemble_fit` is the complementary path.
        f_scale: Soft margin of the robust ``loss`` (scipy's ``f_scale``):
            residuals below it stay quadratic, above it are down-weighted.
            ``None`` (the default) auto-scales it to the data. A quick
            linear-loss seed fit is run and ``f_scale`` is set to a robust
            scale (``1.4826 * MAD``) of that fit's window-area residuals,
            which puts the margin where a robust ``loss`` actually engages
            instead of sitting in its quadratic regime. Pass an explicit value
            to override. Ignored when ``loss="linear"``.
        robust: If True, robustify the integrand itself rather than only the
            window areas: an IRLS loop winsorizes each sample's residual to
            the current model within ``huber_c`` robust sigmas (MAD) before
            re-integrating, keeping individual outlier samples from distorting
            a window's area. Finer-grained than ``loss=``, which down-weights
            whole window areas, and the two compose. Costs a few re-solves and
            needs no ``f_scale`` tuning.
        huber_c: Robust winsorization threshold in residual sigmas for
            ``robust=True`` (~3 leaves clean samples untouched).
        p0: Optional initial guess (defaults to ones): a sequence in
            parameter order (sorted-name for symbolic, signature order for a
            callable) or a full ``{name: value}`` dict.
        sigma: Optional per-sample measurement standard deviation of
            ``data_y``, the same length as the raw input. Each window's area
            residual is weighted by ``1 / sigma_area``, with
            ``sigma_area**2 = sum_i (simpson_weight_i * sigma_i)**2`` over the
            window, turning the area-matching system into weighted least
            squares for heteroscedastic data. ``None`` (the default) fits
            unweighted. Entries must be finite and strictly positive.
        absolute_sigma: If ``True``, treat ``sigma`` as absolute errors and
            leave the covariance unscaled by the reduced chi-square, the
            residual already being ``1/sigma``-scaled, matching
            :func:`scipy.optimize.curve_fit`. If ``False`` (the default) only
            the relative magnitudes of ``sigma`` matter and the covariance is
            scaled by the residual variance.
        solver_options: Optional mapping forwarded to
            :func:`scipy.optimize.least_squares` on every solve: the main fit,
            the ``f_scale`` seed fit and any robust IRLS re-solves. For
            example ``{"xtol": 1e-12, "max_nfev": 500}``. The fitter's managed
            keys (``loss`` / ``f_scale`` / ``bounds`` / ``method`` / ``jac``)
            always win over it.
        param_names: For a callable model, the parameter names in signature
            order when they cannot be introspected from the signature (a
            ``*args`` model or a signature-less builtin); validated against
            the introspected names otherwise. For a symbolic model it is
            optional and validated against the names parsed from the
            expression.
        nan_policy: ``"raise"`` (the default) rejects non-finite samples;
            ``"omit"`` drops NaN/inf ``(x, y)`` pairs before fitting, which
            suits gappy sensor or GPS telemetry.

    Returns:
        FittingResult with the fitted coefficients, a callable model and, when
        overdetermined, a parameter covariance estimate. It also carries the
        fit-quality diagnostics ``n_obs`` / ``rss`` / ``tss`` behind
        ``.rsquared`` / ``.aic`` / ``.bic``, plus the optimizer's ``nfev`` and
        ``cost``. For a callable model the result carries a bound ``f(x)``
        model and a ``param_model`` evaluator instead of ``expr``, so
        ``.to_dict`` raises as it does for any expression-less fit.
    """
    if window_mode not in ("uniform", "curvature"):
        raise ValueError(
            f"window_mode must be 'uniform' or 'curvature', got {window_mode!r}"
        )
    # A string or sympy.Expr keeps the sorted-name parameter order; a callable
    # uses signature order. Symbolic derivatives come from ``sp.diff``, a
    # callable is forward-differenced.
    spec = resolve_model(expr, var, param_names=param_names)
    names = list(spec.names)
    n = len(names)
    if n == 0:
        raise RuntimeError("Model expression has no free parameters to fit.")

    p0_arr = normalize_p0(p0, names)
    bounds_list = normalize_bounds(bounds, names)
    # scipy's least_squares wants the (lo_array, hi_array) form.
    scipy_bounds = (
        ([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        if bounds_list is not None else None
    )
    so: dict[str, Any] = dict(solver_options) if solver_options else {}

    # Coerce pandas Series / single-column DataFrame input to plain 1-D float
    # arrays. Gated on the pandas types: a raw ndarray or list passes through
    # untouched, the 2-D rejection in ``_validate_xy`` included.
    if is_series(data_x) or is_dataframe(data_x):
        data_x = to_1d_array(data_x, "data_x")
    if is_series(data_y) or is_dataframe(data_y):
        data_y = to_1d_array(data_y, "data_y")

    x, y = _validate_xy(data_x, data_y, min_size=2 * n, nan_policy=nan_policy)
    sigma_active = _resolve_sigma(sigma, data_x, data_y, x, nan_policy)

    # Contiguous window spans [start, stop) over the active region. The model
    # and its sensitivities are evaluated once over the whole region per solver
    # step, then integrated per window by the compiled Simpson kernel, rather
    # than re-evaluated window by window.
    x_active, y_active, starts, stops, data_areas_arr, m = _place_windows(
        x, y, n, window_mode=window_mode, active_ratio=active_ratio,
        n_windows=n_windows,
    )
    echo(f"EAC windows: {m} (params: {n}, mode: {window_mode})")

    # Per-window least-squares weights from the per-sample sigma (see
    # ``_area_weights``). ``x_active`` is always a leading slice of ``x``: a
    # prefix under "uniform", all of it under "curvature". The sigma aligns by
    # that same prefix. ``None`` leaves the system unweighted.
    area_w = (
        _area_weights(x_active, sigma_active[: x_active.size], starts, stops)
        if sigma_active is not None else None
    )

    def _clean(v: np.ndarray) -> np.ndarray:
        """Neutralize singular samples in a model value / sensitivity array.

        A transcendental sensitivity can be singular at an isolated sample
        while its integral over the window is finite: ``d/dn`` of ``x**n`` is
        ``x**n*log(x)``, NaN at ``x=0``, with limit 0 there. Such
        measure-zero blow-ups become 0 and the area stays well-posed. A
        widespread blow-up is a different animal, a diverging trial such as
        ``exp(b*x)`` with ``b`` runaway, and zeroing it would make a divergent
        model's area look small and let LM settle in the wrong basin. Those
        are capped at a large finite penalty matched to the data scale, which
        keeps the residual large and pushes the solver out of the divergent
        region.

        Both the model area and every parameter-sensitivity area pass through
        here.
        """
        v = np.ascontiguousarray(v, dtype=float)
        finite = np.isfinite(v)
        if finite.all():
            return v
        if 1.0 - float(finite.mean()) <= 0.05:  # isolated singularities
            return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        scale = float(np.max(np.abs(v[finite]))) if finite.any() else 1.0
        penalty = 1e6 * max(scale, 1.0)
        return np.nan_to_num(v, nan=penalty, posinf=penalty, neginf=-penalty)

    def _model_area(c: np.ndarray) -> np.ndarray:
        mv = _clean(spec.eval(x_active, c))
        return simpson_windows(mv, x_active, starts, stops)

    def residuals(c: np.ndarray) -> np.ndarray:
        r = _model_area(c) - data_areas_arr
        return r * area_w if area_w is not None else r

    def jacobian(c: np.ndarray) -> np.ndarray:
        rows = np.vstack([_clean(d) for d in spec.param_derivs(x_active, c)])
        j = simpson_windows_rows(rows, x_active, starts, stops).T
        return j * area_w[:, None] if area_w is not None else j

    guess = _validate_p0(p0_arr, names)
    if scipy_bounds is not None:
        # scipy's trf rejects an out-of-bounds x0. Clip the (all-ones by
        # default) seed into the user's box, as solve_weighted_nlls does.
        guess = np.clip(guess, scipy_bounds[0], scipy_bounds[1])

    nfev_total = 0
    have_nfev = False

    def _tally(s: Any) -> None:
        """Accumulate a solver's ``nfev``, tolerating a namespace without one."""
        nonlocal nfev_total, have_nfev
        nf = getattr(s, "nfev", None)
        if nf is not None:
            nfev_total += int(nf)
            have_nfev = True

    if scipy_bounds is not None or loss != "linear":
        method = "trf"
        fs = f_scale
        if loss != "linear" and fs is None:
            # Auto-scale the robust margin to the data. At the all-ones seed
            # the window-area residuals are huge, so the margin comes from a
            # quick linear-loss fit's residuals instead: f_scale = 1.4826 *
            # MAD, the robust scale of a clean window's area residual.
            seed_kwargs: dict[str, Any] = dict(so)
            seed_method = "trf" if scipy_bounds is not None else "lm"
            if scipy_bounds is not None:
                seed_kwargs["bounds"] = scipy_bounds
            seed = least_squares(
                residuals, guess, jac=cast(Any, jacobian),
                method=seed_method, **seed_kwargs
            )
            r0 = np.abs(np.asarray(seed.fun, dtype=float))
            mad = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
            fs = mad if mad > 0.0 else (float(np.median(r0)) or 1.0)
            guess = np.asarray(seed.x, dtype=float)
        elif fs is None:
            fs = 1.0  # linear loss ignores f_scale; keep scipy happy
        # Managed keys (loss/f_scale/bounds) win over any user solver_options.
        kwargs: dict[str, Any] = {**so, "loss": loss, "f_scale": fs}
        if scipy_bounds is not None:
            kwargs["bounds"] = scipy_bounds
        # cast: scipy's stub types `jac` as a str literal, omitting the
        # callable form the runtime accepts.
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method=method, **kwargs
        )
    else:
        sol = least_squares(
            residuals, guess, jac=cast(Any, jacobian), method="lm", **so
        )
    _tally(sol)
    coeffs = np.asarray(sol.x, dtype=np.float64)

    if robust:
        # Robust integral by IRLS: winsorize each sample's residual to the
        # current model within huber_c robust sigmas, then re-integrate,
        # keeping an outlier sample from distorting its window's area.
        # ``data_areas_arr`` is reassigned here and the ``residuals`` closure
        # reads it lazily, so the re-solve sees the winsorized areas. Three
        # passes suffice.
        for _ in range(3):
            mv = _clean(spec.eval(x_active, coeffs))
            resid = y_active - mv
            med = float(np.median(resid))
            rscale = 1.4826 * float(np.median(np.abs(resid - med)))
            if rscale <= 0.0:
                break
            clip = huber_c * rscale
            y_eff = mv + (med + np.clip(resid - med, -clip, clip))
            data_areas_arr = simpson_windows(
                np.ascontiguousarray(y_eff), x_active, starts, stops
            )
            if scipy_bounds is not None:
                sol = least_squares(residuals, coeffs, jac=cast(Any, jacobian),
                                    method="trf", bounds=scipy_bounds, **so)
            else:
                sol = least_squares(residuals, coeffs, jac=cast(Any, jacobian),
                                    method="lm", **so)
            _tally(sol)
            coeffs = np.asarray(sol.x, dtype=np.float64)
    echo("EAC fitted coefficients:", coeffs)

    cov = _covariance(sol.jac, sol.fun, n, absolute_sigma=absolute_sigma)
    # The robust path relabels the message; the status stays the solver's own.
    converged = bool(sol.success)
    message = f"robust IRLS ({sol.message})" if robust else str(sol.message)

    # Diagnostics over the full (x, y): rss/tss/n_obs feed R^2 and AIC/BIC,
    # while nfev (summed across any robust IRLS re-solves) and cost come from
    # the optimizer. A monkeypatched solver namespace may omit either, hence
    # the defensive reads.
    yhat = np.asarray(spec.eval(x, coeffs), dtype=float)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    cost_val = getattr(sol, "cost", None)

    # A symbolic model keeps the lambdify path (expr/var/names) that std bands
    # and to_dict run on; a callable carries a bound f(x) closure plus the
    # params-explicit evaluator for finite-differenced std bands. The model
    # rebuilds lazily, so reading only coeffs/cov spends no compile.
    return FittingResult(
        coeffs=coeffs, cov=cov,
        converged=converged, message=message,
        x_range=(float(np.min(x)), float(np.max(x))),
        n_obs=int(y.size), rss=rss, tss=tss,
        nfev=nfev_total if have_nfev else None,
        cost=None if cost_val is None else float(cost_val),
        **result_kwargs(spec, coeffs),
    )

