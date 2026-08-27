"""Stochastic-series estimators. Each recovers one parameter of a stochastic
model by feeding a deterministic functional of the data (its ACF, spectrum,
aggregated variance, or trend+cycle) to dtfit's integral fitters. The theory is
in :mod:`dtfit.stochastic`; :func:`dtfit.stochastic.fit_stochastic` composes
these behind significance gates."""

from __future__ import annotations

import numpy as np

from dtfit.methods import fit_lsi, fit_eac

__all__ = [
    "sample_acf", "hurst_aggvar", "hurst_spectral", "ar1_reversion",
    "ar_order", "fit_ar", "fractional_difference",
    "garch_persistence", "cycle_period", "decompose_trend_cycle",
]


# The sample autocorrelation is the deterministic functional most of these
# estimators feed to dtfit.
def sample_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Biased sample autocorrelation ``rho[0..nlags]`` (``rho[0] == 1``).

    Dividing by ``n`` rather than ``n - k`` keeps the sequence positive
    definite and lowers its variance; both matter to the moment estimators
    below. It also tapers the noisy long-lag tail toward zero instead of
    letting it blow up.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    denom = float(x @ x)
    if denom == 0.0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out
    # Wiener-Khinchin autocovariance: O(n log n) against the O(nlags*n) direct
    # loop, and numerically identical to it here because the 1/n cancels in
    # the ratio.
    m = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    acov = np.fft.irfft(f * np.conj(f), m)[: nlags + 1]
    return acov / acov[0]


def _loglog_slope(lx: np.ndarray, ly: np.ndarray, *, method: str) -> float:
    """Slope ``b`` of ``ly = a + b*lx``.

    ``method="lsi"`` fits the line with dtfit's reconditioned Legendre LSI
    (orthogonal-basis least squares); ``method="ols"`` is the plain
    ``numpy.polyfit`` baseline the domain experiment compares against.
    """
    if method == "ols":
        return float(np.polyfit(lx, ly, 1)[0])
    r = fit_lsi(lx, ly, "a + b*m", "m", k_star=1, filter_data=False)
    # sympy name-sorted order -> [a, b]
    return float(r.coeffs[1])


# Long memory / self-similarity: the Hurst exponent H, equivalently the
# fractional-integration order d = H - 1/2.
def hurst_aggvar(
    x: np.ndarray,
    *,
    n_scales: int = 14,
    min_block: int = 2,
    method: str = "lsi",
) -> dict[str, float]:
    """Hurst exponent via the aggregated-variance power law.

    Block-average the series at geometrically spaced scales ``m``; for a
    self-similar process the block-mean variance scales as
    ``Var(m) ~ c * m^(2H - 2)``. The exponent comes from a power-law fit of
    ``Var`` against ``m``. ``method="lsi"`` and ``method="ols"`` fit that in
    log-log space, where the slope is ``2H - 2``; ``method="eac"`` fits
    ``c*m**b`` directly in linear space with the equal-areas criterion.

    Returns ``{"H", "slope", "d"}`` (``d = H - 1/2``).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    max_block = max(min_block * 2, n // 4)
    blocks = np.unique(
        np.round(np.geomspace(min_block, max_block, n_scales)).astype(int)
    )
    blocks = blocks[blocks >= 2]

    ms, vs = [], []
    for m in blocks:
        nb = n // m
        if nb < 8:  # need enough blocks for a stable per-scale variance
            continue
        bmean = x[: nb * m].reshape(nb, m).mean(axis=1)
        # ddof=1: block counts vary across scales, and ddof=0 would tilt the
        # log-log slope.
        v = float(bmean.var(ddof=1))
        if v > 0:
            ms.append(float(m))
            vs.append(v)
    ms_a = np.asarray(ms)
    vs_a = np.asarray(vs)
    if ms_a.size < 3:
        raise RuntimeError("too few usable scales for aggregated-variance Hurst")

    if method == "eac":
        # Nonlinear power law c*m**b in linear space, no log transform.
        # model_params sorts the names to [b, c] and fit_eac maps p0 and
        # bounds positionally onto that order, so both are given exponent
        # first. b lies in [-2, 0], equivalently H in [0, 1]; the log-log seed
        # is clipped into the open bracket to keep it off the bounds.
        b0 = float(np.clip(np.polyfit(np.log(ms_a), np.log(vs_a), 1)[0], -1.999, -1e-9))
        r = fit_eac(ms_a, vs_a, "c*m**b", "m",
                    p0=[b0, float(vs_a[0])],
                    bounds=([-2.0, 1e-12], [0.0, 1e6]))
        slope = float(r.coeffs[0])  # sorted [b, c]: the exponent is first
    else:
        slope = _loglog_slope(np.log(ms_a), np.log(vs_a), method=method)

    H = (slope + 2.0) / 2.0
    return {"H": float(np.clip(H, 0.0, 1.0)), "slope": float(slope),
            "d": float(np.clip(H, 0.0, 1.0) - 0.5)}


def hurst_spectral(
    x: np.ndarray,
    *,
    n_freq: int | None = None,
    method: str = "lsi",
) -> dict[str, float]:
    """Hurst / fractional-integration order via the low-frequency spectrum.

    The log-periodogram of a long-memory process is ``log S(f) = const - 2d
    log f + noise`` for small ``f``, the GPH regression, so the slope of the
    low-frequency log-log periodogram gives ``d = H - 1/2``. Under
    ``method="lsi"`` dtfit supplies the orthogonal-basis line fit, its
    Savitzky-Golay pre-filter taming the periodogram's chi-squared scatter;
    ``method="ols"`` is the plain GPH baseline.

    Returns ``{"H", "d", "slope"}`` (``slope = -2d``).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    per = (np.abs(np.fft.rfft(x)) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0)
    if n_freq is None:
        n_freq = max(8, int(n ** 0.6))
    hi = min(1 + n_freq, freqs.size)
    f = freqs[1:hi]
    p = per[1:hi]
    keep = p > 0
    f, p = f[keep], p[keep]
    if f.size < 3:
        raise RuntimeError("too few usable frequencies for spectral Hurst")

    # The exact GPH design variable is log|2 sin(lambda/2)| at the angular
    # frequency lambda = 2*pi*f, not log(f). The two agree only as f -> 0,
    # where 2 sin(pi f) ~ 2 pi f and the constant offset is absorbed by the
    # intercept. Across a bandwidth of ~n^0.6 bins the sine curvature biases
    # a log(f) slope.
    lf, lp = np.log(2.0 * np.sin(np.pi * f)), np.log(p)
    if method == "ols":
        slope = float(np.polyfit(lf, lp, 1)[0])
    else:
        # denoise the log-periodogram before reading the slope
        r = fit_lsi(lf, lp, "a + b*m", "m", k_star=1, filter_data=True)
        slope = float(r.coeffs[1])
    d = -slope / 2.0
    H = d + 0.5
    return {"H": float(H), "d": float(d), "slope": float(slope)}


# Mean reversion: an OU / AR(1) process has a single-exponential ACF,
# exp(-k/tau).
def ar1_reversion(
    x: np.ndarray,
    *,
    nlags: int | None = None,
    method: str = "lsi",
) -> dict[str, float]:
    """Mean-reversion speed from an exponential fit to the ACF.

    An OU / AR(1) process has ``rho(k) = phi^k = exp(-k/tau)``. Fitting a
    decaying exponential to the sample ACF (``method="lsi"`` or ``"eac"``)
    reads off the AR(1) coefficient ``phi`` and the reversion time ``tau``
    from many lags at once, the integral fitters averaging the per-lag ACF
    noise. The ``method="acf1"`` baseline just takes the lag-1
    autocorrelation.

    Returns ``{"phi", "tau", "halflife"}``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if nlags is None:
        nlags = int(np.clip(n // 4, 10, 60))
    acf = sample_acf(x, nlags)

    if method == "acf1":
        phi = float(np.clip(acf[1], 1e-6, 0.999999))
    else:
        phi = _phi_from_acf(acf, n, method=method)
    tau = -1.0 / np.log(phi) if 0.0 < phi < 1.0 else np.inf
    halflife = tau * np.log(2.0) if np.isfinite(tau) else np.inf
    return {"phi": phi, "tau": float(tau), "halflife": float(halflife)}


def _phi_from_acf(acf: np.ndarray, n: int, *, method: str = "lsi") -> float:
    """AR(1) ``phi`` from a decaying-exponential dtfit fit to the ACF.

    The core of the batch :func:`ar1_reversion`.
    :class:`~dtfit.stochastic.StochasticFilter` re-derives this for its EWMA
    ACF and does not call it.

    The fit is restricted to lags where the ACF is still above the
    white-noise band (~``2/sqrt(n)``); past that the ACF is sampling noise
    and only drags the decay rate. The amplitude is anchored (``exp(-g*k)``,
    no free ``A``) because an AR(1) ACF is exactly ``phi^k`` and passes
    through 1 at lag 0. A free ``A`` would absorb the lag-1 noise and badly
    bias the fast-decay case.
    """
    nlags = acf.size - 1
    band = max(0.05, 2.0 / np.sqrt(max(n, 1)))
    above = np.where(np.abs(acf[1:]) >= band)[0]
    keff = int(above[-1] + 1) if above.size else nlags
    keff = int(np.clip(keff, 5, nlags))
    k = np.arange(1, keff + 1, dtype=float)
    y = acf[1: keff + 1]
    fitter = fit_eac if method == "eac" else fit_lsi
    r = fitter(k, y, "exp(-g*k)", "k", p0=[0.1])
    return float(np.exp(-abs(float(r.coeffs[0]))))


# Higher-order autoregression: AR(p) order selection and a Yule-Walker fit.
# The AR(1) route above whitens with a single lag, and a genuine AR(2)/AR(3)
# then leaves residual structure that can be mistaken for long memory. These
# two read the order and the AR coefficients off the sample ACF. That is what
# tells a low-order AR apart from true long memory.
def ar_order(x: np.ndarray, *, max_order: int = 8, ic: str = "aic") -> int:
    """Select an AR(p) order for ``x`` by an information criterion.

    Fits Yule-Walker AR(k) for ``k = 0..max_order`` and returns the ``k``
    minimizing AIC (``ic="aic"``) or BIC (``ic="bic"``). A value above 1 means
    single-lag AR(1) whitening would leave residual structure: the series is a
    higher-order autoregression, not long memory. Near-white input returns
    ``0``.
    """
    from scipy.linalg import toeplitz

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    cap = int(min(max_order, n // 2 - 1))
    if cap < 1:
        return 0
    acf = sample_acf(x, cap)
    gamma0 = float(x @ x) / n if n else 0.0
    if gamma0 <= 0.0:
        return 0
    pen = 2.0 if ic == "aic" else float(np.log(n))
    best_k, best_score = 0, n * float(np.log(gamma0)) + pen  # k=0: 1 param (sigma)
    for k in range(1, cap + 1):
        R = toeplitz(acf[:k])
        r = acf[1:k + 1]
        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            continue
        sigma2 = gamma0 * (1.0 - float(phi @ r))
        if sigma2 <= 0.0:
            continue
        score = n * float(np.log(sigma2)) + pen * (k + 1)
        if score < best_score:
            best_k, best_score = k, score
    return best_k


def fit_ar(
    x: np.ndarray, order: int | None = None, *, max_order: int = 8, ic: str = "aic"
) -> dict[str, object]:
    """Fit an AR(p) model ``x_t = sum_j phi_j x_{t-j} + eps`` by Yule-Walker.

    An ``order`` of ``None`` is chosen by :func:`ar_order` (AIC/BIC). Returns
    ``{"order", "phi", "sigma"}`` with the AR coefficients at lags ``1..p`` and
    the innovation standard deviation. :func:`ar1_reversion` covers the AR(1)
    case alone; this is its general-order companion.
    """
    from scipy.linalg import toeplitz

    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    n = xc.size
    if order is None:
        order = ar_order(xc, max_order=max_order, ic=ic)
    order = int(max(0, min(order, n - 2)))
    gamma0 = float(xc @ xc) / n if n else 0.0
    if order == 0 or gamma0 <= 0.0:
        return {"order": 0, "phi": np.zeros(0), "sigma": float(np.sqrt(max(gamma0, 0.0)))}
    acf = sample_acf(xc, order)
    phi = np.linalg.solve(toeplitz(acf[:order]), acf[1:order + 1])
    sigma2 = gamma0 * (1.0 - float(phi @ acf[1:order + 1]))
    return {"order": order, "phi": np.asarray(phi, dtype=float),
            "sigma": float(np.sqrt(max(sigma2, 0.0)))}


def fractional_difference(
    x: np.ndarray, d: float, *, ntrunc: int | None = None
) -> np.ndarray:
    """Apply the fractional-difference filter ``(1 - B)^d`` to ``x``.

    This is the ARFIMA differencing operator. Differencing a long-memory
    series by its own fractional order ``d`` (from the Hurst read-out,
    ``d = H - 1/2``; see :func:`hurst_spectral`) whitens it, leaving a
    residual that can be checked or modelled as short-memory. The weights come
    from the truncated binomial expansion ``w_0 = 1``,
    ``w_k = w_{k-1} (k - 1 - d) / k``, applied causally, and the result has
    the same length as ``x``. ``d = 1`` recovers the ordinary first difference
    prepended with the first value; ``d = 0`` is the identity.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x.copy()
    trunc = int(min(n, ntrunc if ntrunc is not None else 1000))
    w = np.empty(trunc)
    w[0] = 1.0
    for k in range(1, trunc):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return np.asarray(np.convolve(x, w)[:n], dtype=float)


# Volatility clustering: the GARCH(1,1) persistence alpha+beta, read off an
# ACF of squared returns that decays geometrically as (alpha+beta)^k.
def garch_persistence(
    returns: np.ndarray,
    *,
    nlags: int | None = None,
    method: str = "lsi",
    use: str = "square",
) -> dict[str, float]:
    """Volatility persistence from the ACF of squared or absolute returns.

    For a GARCH(1,1) the autocorrelation of the squared returns decays
    geometrically with ratio ``alpha + beta``, the persistence. Fitting a
    decaying exponential to that ACF with dtfit recovers it without running a
    full GARCH likelihood. ``use="abs"`` fits the ACF of absolute returns,
    empirically often the cleaner one.

    Returns ``{"persistence", "tau"}``.
    """
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    z = np.abs(r) if use == "abs" else r ** 2
    n = z.size
    if nlags is None:
        nlags = int(np.clip(n // 8, 10, 50))
    acf = sample_acf(z, nlags)
    persistence = _persistence_from_acf(acf, method=method, n=n)
    tau = -1.0 / np.log(persistence) if 0.0 < persistence < 1.0 else np.inf
    return {"persistence": persistence, "tau": float(tau)}


def _persistence_from_acf(
    acf: np.ndarray, *, method: str = "lsi", n: int | None = None
) -> float:
    """Volatility persistence from a decaying-exponential dtfit fit to the ACF
    of absolute or squared returns.

    The core of the batch :func:`garch_persistence`.
    :class:`~dtfit.stochastic.StochasticFilter` re-derives this for its EWMA
    ACF and does not call it.

    Unlike the AR(1) fit, the amplitude ``A`` is left free. The squared-return
    ACF of a GARCH process carries a level offset and does not pass through 1
    at lag 1, so the geometric decay has to be read as ``A*exp(-g*k)`` with
    only the ratio ``exp(-g)`` being the persistence. Where ``n`` is known the
    fit is restricted to lags above the white-noise band (~``2/sqrt(n)``), as
    in :func:`_phi_from_acf`; past that the ACF is sampling noise and only
    drags the decay rate.
    """
    nlags = acf.size - 1
    keff = nlags
    if n is not None:
        band = max(0.05, 2.0 / np.sqrt(max(int(n), 1)))
        above = np.where(np.abs(acf[1:]) >= band)[0]
        keff = int(above[-1] + 1) if above.size else nlags
        keff = int(np.clip(keff, 3, nlags))
    k = np.arange(1, keff + 1, dtype=float)
    fitter = fit_eac if method == "eac" else fit_lsi
    rr = fitter(k, acf[1: keff + 1], "A*exp(-g*k)", "k",
                p0=[float(acf[1]) or 0.2, 0.1])
    g = float(rr.coeffs[1])
    return float(np.clip(np.exp(-abs(g)), 0.0, 0.9999))


# Pseudo-cycle: an AR(2) with complex roots has a damped-cosine ACF.
def cycle_period(
    x: np.ndarray,
    *,
    nlags: int | None = None,
) -> dict[str, float]:
    """Dominant cycle period from a damped-cosine fit to the ACF.

    An AR(2) with complex roots, a stochastic pseudo-cycle such as a business
    cycle, has ``rho(k) = r^k cos(w k + phi)``. dtfit's oscillatory recipe
    (FFT-seeded angular frequency, no smoothing, raised spectral order) fits
    that damped cosine to the sample ACF and returns the cycle period
    ``2*pi/w`` alongside the damping ``r``.

    Returns ``{"period", "w", "damping"}``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if nlags is None:
        nlags = int(np.clip(n // 2, 20, 100))
    acf = sample_acf(x, nlags)
    return _cycle_from_acf(acf)


def _cycle_from_acf(acf: np.ndarray) -> dict[str, float]:
    """Dominant cycle from a damped-cosine dtfit fit to the ACF.

    The core of the batch :func:`cycle_period`.
    :class:`~dtfit.stochastic.StochasticFilter` re-derives the same AR(2)
    principle for its EWMA ACF and does not call it.

    :func:`dtfit.fit_lsi`'s oscillatory recipe seeds the angular frequency
    ``w`` itself from the ACF's spectral peak (``freq_param="w"``), so only
    the damping and the frequency are read back.
    """
    nlags = acf.size - 1
    k = np.arange(nlags + 1, dtype=float)
    # Sorted param order is [A, g, p, w]. fit_lsi(freq_param="w") supplies the
    # frequency seed from the ACF's spectral peak, so p0's w entry is nominal.
    r = fit_lsi(k, acf, "A*exp(-g*k)*cos(w*k + p)", "k",
                freq_param="w", p0=[1.0, 0.05, 0.0, np.pi / 2.0])
    g, w = float(r.coeffs[1]), abs(float(r.coeffs[3]))
    # A near-zero recovered frequency means no cycle was found; report that
    # rather than an enormous spurious period.
    period = 2.0 * np.pi / w if w > (2.0 * np.pi / (2.0 * nlags)) else float("inf")
    return {"period": float(period), "w": float(w),
            "damping": float(np.exp(-abs(g)))}


# Structural decomposition: trend + cycle + stochastic residual.
def decompose_trend_cycle(
    t: np.ndarray,
    y: np.ndarray,
    *,
    trend_deg: int = 1,
    with_cycle: bool = True,
) -> dict[str, object]:
    """Split ``y`` into a deterministic trend plus cycle, both fit by dtfit,
    and a stochastic residual left for a noise model.

    The trend is a low-order LSI polynomial and the cycle is the LSI
    oscillatory recipe fit to the de-trended residual.

    Returns a dict with the fitted ``trend``/``cycle``/``residual`` arrays, the
    recovered ``slope`` and ``period``/``amp``, and a ``forecast(h, dt)``
    closure that extrapolates trend + cycle ``h`` steps ahead.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    terms = " + ".join(f"a{i}*x**{i}" for i in range(trend_deg + 1))
    rt = fit_lsi(t, y, terms, "x", k_star=max(2, trend_deg), filter_data=True)
    trend = np.asarray(rt.model(t), dtype=float)
    if np.ndim(trend) == 0:
        trend = np.full_like(t, float(trend))
    # sympy's name sort puts the coefficients in ascending order a0, a1, ...
    tcoeffs = [float(c) for c in rt.coeffs]
    slope = tcoeffs[1] if trend_deg >= 1 else 0.0

    resid = y - trend
    period = amp = w = phase = float("nan")
    a_sig = 0.0  # signed cycle amplitude used by the forecast closure
    cycle = np.zeros_like(t)
    if with_cycle and t.size >= 8:
        rc = fit_lsi(t, resid, "A*sin(w*x + p)", "x", freq_param="w",
                     p0=[float(np.std(resid)) + 1e-3, 0.5, 0.0])
        cycle = np.asarray(rc.model(t), dtype=float)
        if np.ndim(cycle) == 0:
            cycle = np.full_like(t, float(cycle))
        a_sig, phase, w = float(rc.coeffs[0]), float(rc.coeffs[1]), float(rc.coeffs[2])
        amp = abs(a_sig)
        period = 2.0 * np.pi / w if w > 1e-9 else float("nan")
    noise = resid - cycle

    def forecast(h: int, dt: float | None = None) -> np.ndarray:
        step = float(t[1] - t[0]) if (dt is None and t.size > 1) else (dt or 1.0)
        tf = t[-1] + step * np.arange(1, h + 1)
        tr = np.asarray(rt.model(tf), dtype=float)
        if np.ndim(tr) == 0:
            tr = np.full_like(tf, float(tr))
        cy = a_sig * np.sin(w * tf + phase) if np.isfinite(w) else np.zeros_like(tf)
        return tr + cy

    return {
        "trend": trend, "cycle": cycle, "residual": noise,
        "slope": float(slope), "period": float(period), "amp": float(amp),
        "noise_std": float(np.std(noise)), "forecast": forecast,
    }
