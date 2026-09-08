"""Stochastic-series estimators. Each recovers one parameter of a stochastic
model from a functional of the second-order image: its autocovariance, its
Blackman-Tukey spectrum, its aggregated variance or its seasonal read-out."""

from __future__ import annotations

from typing import Any

import numpy as np

from dtfit.image import fit_lsi, fit_eac
from dtfit.image.original import Original
from .image import SecondOrderImage

__all__ = [
    "as_image", "sample_acf", "hurst_aggvar", "hurst_spectral",
    "ar1_reversion", "ar_order", "fit_ar", "fractional_difference",
    "garch_persistence", "cycle_period", "decompose_trend_cycle",
    "dickey_fuller", "adf_pvalue", "seasonal_series",
]

DEFAULT_LAG = 256
DEFAULT_NFREQ = 512


def as_image(
    data: Any,
    *,
    lag: int = DEFAULT_LAG,
    nfreq: int = DEFAULT_NFREQ,
    scales: int | None = None,
) -> SecondOrderImage:
    """The second-order image of ``data``.

    Args:
        data: a :class:`SecondOrderImage` (returned unchanged), an
            :class:`~dtfit.image.Original` on a uniform grid, a pandas
            Series, or a 1-D array of values at unit spacing.
        lag, nfreq, scales: budgets for an image that has to be built; see
            :meth:`SecondOrderImage.of`.

    Raises:
        ValueError: fewer than two samples, or a non-uniform Original.
    """
    if isinstance(data, SecondOrderImage):
        return data
    if not isinstance(data, Original):
        from dtfit._pandas import to_1d_array
        data = to_1d_array(data, "data")
    return SecondOrderImage.of(data, lag=lag, nfreq=nfreq, scales=scales)


def _decay_lags(rho: np.ndarray, n: int, floor: int) -> int:
    """Last lag at which the autocorrelation is still above the white-noise
    band ``max(0.05, 2/sqrt(n))``, floored at ``floor``."""
    nlags = rho.size - 1
    band = max(0.05, 2.0 / np.sqrt(max(n, 1)))
    above = np.where(np.abs(rho[1:]) >= band)[0]
    keff = int(above[-1] + 1) if above.size else nlags
    return int(np.clip(keff, floor, nlags))


def sample_acf(data: Any, nlags: int) -> np.ndarray:
    """Biased sample autocorrelation ``rho[0..k]`` (``rho[0] == 1``).

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        nlags: lags asked for, at least 1.

    Returns:
        ``min(nlags, lag) + 1`` entries, with ``lag`` the lag budget of the
        image: an image handed in caps the answer at its own budget, and an
        image built here caps it at ``n - 1``. Dividing by ``n`` rather than
        ``n - k`` keeps the sequence positive definite and lowers its
        variance, and tapers the noisy long-lag tail toward zero.
    """
    img = as_image(data, lag=max(1, int(nlags)))
    g = img.acov()
    k = min(int(nlags), img.lag) + 1
    out = np.zeros(k)
    if g[0] <= 0.0:
        out[0] = 1.0
        return out
    out[:] = g[:k] / g[0]
    return out


def _loglog_slope(lx: np.ndarray, ly: np.ndarray, *, method: str) -> float:
    """Slope ``b`` of ``ly = a + b*lx``; ``method="lsi"`` fits the line with
    the Legendre preset, ``"ols"`` with ``numpy.polyfit``."""
    if method == "ols":
        return float(np.polyfit(lx, ly, 1)[0])
    r = fit_lsi(lx, ly, "a + b*m", "m", k_star=1)
    return float(r.coeffs[1])


def hurst_aggvar(data: Any, *, method: str = "lsi") -> dict[str, float]:
    """Hurst exponent from the image's aggregated variance.

    The variance of the block means at scale ``m`` follows
    ``Var(m) ~ c m^(2H-2)``. ``method="lsi"`` and ``"ols"`` fit that power law
    in log-log space, ``method="eac"`` fits ``c*m**b`` in linear space with
    the block-basis preset.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        method: ``"lsi"``, ``"ols"`` or ``"eac"``.

    Returns:
        ``{"H", "slope", "d"}`` with ``d = H - 1/2``, ``H`` clipped to
        ``[0, 1]``.

    Raises:
        RuntimeError: fewer than three usable scales in the image.
    """
    img = as_image(data)
    ms, vs = img.aggregated_variance()
    if ms.size < 3:
        raise RuntimeError(
            "too few usable scales for aggregated-variance Hurst")
    if method == "eac":
        b0 = float(np.clip(
            np.polyfit(np.log(ms), np.log(vs), 1)[0], -1.999, -1e-9))
        r = fit_eac(ms, vs, "c*m**b", "m", p0=[b0, float(vs[0])],
                    bounds=([-2.0, 1e-12], [0.0, 1e6]))
        slope = float(r.coeffs[0])
    else:
        slope = _loglog_slope(np.log(ms), np.log(vs), method=method)
    h = float(np.clip((slope + 2.0) / 2.0, 0.0, 1.0))
    return {"H": h, "slope": float(slope), "d": h - 0.5}


def hurst_spectral(
    data: Any, *, n_freq: int | None = None, method: str = "lsi"
) -> dict[str, float]:
    """Hurst exponent from the low-frequency Blackman-Tukey spectrum.

    ``log S(f) = const - 2d log|2 sin(pi f)|`` near zero frequency, the GPH
    regression, run on the image's windowed spectrum rather than on the raw
    periodogram.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        n_freq: frequencies in the regression; ``None`` takes
            ``max(8, n**0.6)``.
        method: ``"lsi"`` fits the line with the Legendre preset, ``"ols"``
            with ``numpy.polyfit``.

    Returns:
        ``{"H", "d", "slope"}`` with ``slope = -2d``.

    Raises:
        RuntimeError: fewer than three usable frequencies.
    """
    img = as_image(data)
    if n_freq is None:
        n_freq = max(8, int(img.n ** 0.6))
    f, s = img.spectrum(n_freq=int(n_freq))
    keep = s > 0
    f, s = f[keep], s[keep]
    if f.size < 3:
        raise RuntimeError("too few usable frequencies for spectral Hurst")
    lf, lp = np.log(2.0 * np.sin(np.pi * f)), np.log(s)
    slope = _loglog_slope(lf, lp, method=method)
    d = -slope / 2.0
    return {"H": float(d + 0.5), "d": float(d), "slope": float(slope)}


def ar1_reversion(
    data: Any, *, nlags: int | None = None, method: str = "yw"
) -> dict[str, float]:
    """Mean-reversion speed of an OU / AR(1) process.

    ``method="yw"`` reads the Yule-Walker coefficient ``gamma_1 / gamma_0``
    off the image's autocovariance; ``"lsi"`` and ``"eac"`` fit the decaying
    exponential ``exp(-g*k)`` to the autocorrelation over the lags above the
    white-noise band; ``"acf1"`` returns the lag-1 autocorrelation clipped
    into ``(0, 1)``.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        nlags: lags the exponential fit sees; ``None`` takes
            ``clip(n // 4, 10, 60)``, capped at the image's lag budget.
        method: ``"yw"``, ``"lsi"``, ``"eac"`` or ``"acf1"``.

    Returns:
        ``{"phi", "tau", "halflife"}``, ``tau`` and ``halflife`` infinite
        outside ``0 < phi < 1``.
    """
    img = as_image(data)
    g = img.acov()
    if g[0] <= 0.0:
        return {"phi": 0.0, "tau": float("inf"), "halflife": float("inf")}
    rho = g / g[0]
    if method == "yw":
        phi = float(np.clip(rho[1], 1e-6, 0.999999))
    elif method == "acf1":
        phi = float(np.clip(rho[1], 1e-6, 0.999999))
    else:
        if nlags is None:
            nlags = int(np.clip(img.n // 4, 10, 60))
        nlags = int(min(nlags, img.lag))
        keff = _decay_lags(rho[: nlags + 1], img.n, 5)
        k = np.arange(1, keff + 1, dtype=float)
        fitter = fit_eac if method == "eac" else fit_lsi
        r = fitter(k, rho[1: keff + 1], "exp(-g*k)", "k", p0=[0.1])
        phi = float(np.exp(-abs(float(r.coeffs[0]))))
    tau = -1.0 / np.log(phi) if 0.0 < phi < 1.0 else np.inf
    half = float(tau * np.log(2.0)) if np.isfinite(tau) else np.inf
    return {"phi": phi, "tau": float(tau), "halflife": half}


def ar_order(data: Any, *, max_order: int = 8, ic: str = "aic") -> int:
    """AR order of the series by an information criterion.

    Fits Yule-Walker AR(k) for ``k = 0..max_order`` on the image's
    autocorrelation and returns the ``k`` minimizing AIC (``ic="aic"``) or
    BIC (``ic="bic"``). An order above 1 means single-lag whitening would
    leave structure: the series is a higher-order autoregression, not long
    memory. Near-white input returns 0.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        max_order: largest order considered, capped at ``n // 2 - 1`` and at
            the image's lag budget.
        ic: ``"aic"`` or ``"bic"``.
    """
    from scipy.linalg import toeplitz

    img = as_image(data)
    n = img.n
    cap = int(min(max_order, n // 2 - 1, img.lag - 1))
    if cap < 1:
        return 0
    g = img.acov()
    if g[0] <= 0.0:
        return 0
    rho = g / g[0]
    pen = 2.0 if ic == "aic" else float(np.log(n))
    best_k, best_score = 0, n * float(np.log(g[0])) + pen
    for k in range(1, cap + 1):
        r = rho[1:k + 1]
        try:
            phi = np.linalg.solve(toeplitz(rho[:k]), r)
        except np.linalg.LinAlgError:
            continue
        sigma2 = g[0] * (1.0 - float(phi @ r))
        if sigma2 <= 0.0:
            continue
        score = n * float(np.log(sigma2)) + pen * (k + 1)
        if score < best_score:
            best_k, best_score = k, score
    return best_k


def fit_ar(
    data: Any, order: int | None = None, *, max_order: int = 8, ic: str = "aic"
) -> dict[str, object]:
    """Yule-Walker AR(p) fit ``x_t = sum_j phi_j x_{t-j} + eps``.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        order: AR order; ``None`` is chosen by :func:`ar_order`.
        max_order, ic: forwarded to :func:`ar_order`.

    Returns:
        ``{"order", "phi", "sigma"}``: the order, the coefficients at lags
        ``1..p`` and the innovation standard deviation.
    """
    from scipy.linalg import toeplitz

    img = as_image(data)
    g = img.acov()
    if order is None:
        order = ar_order(img, max_order=max_order, ic=ic)
    order = int(max(0, min(order, img.n - 2, img.lag)))
    if order == 0 or g[0] <= 0.0:
        return {"order": 0, "phi": np.zeros(0),
                "sigma": float(np.sqrt(max(g[0], 0.0)))}
    rho = g / g[0]
    phi = np.linalg.solve(toeplitz(rho[:order]), rho[1:order + 1])
    sigma2 = g[0] * (1.0 - float(phi @ rho[1:order + 1]))
    return {"order": order, "phi": np.asarray(phi, dtype=float),
            "sigma": float(np.sqrt(max(sigma2, 0.0)))}


def fractional_difference(
    x: np.ndarray, d: float, *, ntrunc: int | None = None
) -> np.ndarray:
    """Apply the fractional-difference filter ``(1 - B)^d`` to ``x``.

    The ARFIMA differencing operator: differencing a long-memory series by
    its own ``d = H - 1/2`` whitens it. The weights are the truncated
    binomial expansion ``w_0 = 1``, ``w_k = w_{k-1} (k - 1 - d) / k``, applied
    causally; the result has the length of ``x``. ``d = 1`` is the ordinary
    first difference prepended with the first value, ``d = 0`` the identity.

    Args:
        x: the series, per sample (this filter is not an image read-out).
        d: differencing order.
        ntrunc: weights kept; ``None`` takes ``min(n, 1000)``.
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


def garch_persistence(
    data: Any, *, nlags: int | None = None, method: str = "lsi"
) -> dict[str, float]:
    """Volatility persistence from the autocovariance of the squared series.

    For a GARCH(1,1) the autocorrelation of the squared returns decays
    geometrically with ratio ``alpha + beta``. The decaying exponential
    ``A*exp(-g*k)`` is fitted to that autocorrelation over the lags above the
    white-noise band; the amplitude is free because the squared-return
    autocorrelation carries a level offset.

    Args:
        data: the returns, as a series, an Original or a
            :class:`SecondOrderImage`.
        nlags: lags the fit sees; ``None`` takes ``clip(n // 8, 10, 50)``,
            capped at the image's lag budget.
        method: ``"lsi"`` or ``"eac"`` for the exponential fit.

    Returns:
        ``{"persistence", "tau"}``, the persistence clipped to
        ``[0, 0.9999]``.
    """
    img = as_image(data)
    return _persistence(img.acov_squares(), img.n, nlags, method)


def _persistence(
    g: np.ndarray, n: int, nlags: int | None, method: str
) -> dict[str, float]:
    if g[0] <= 0.0:
        return {"persistence": 0.0, "tau": float("inf")}
    rho = g / g[0]
    if nlags is None:
        nlags = int(np.clip(n // 8, 10, 50))
    nlags = int(min(nlags, rho.size - 1))
    keff = _decay_lags(rho[: nlags + 1], n, 3)
    k = np.arange(1, keff + 1, dtype=float)
    fitter = fit_eac if method == "eac" else fit_lsi
    r = fitter(k, rho[1: keff + 1], "A*exp(-g*k)", "k",
               p0=[float(rho[1]) or 0.2, 0.1])
    p = float(np.clip(np.exp(-abs(float(r.coeffs[1]))), 0.0, 0.9999))
    return {"persistence": p,
            "tau": float(-1.0 / np.log(p)) if 0.0 < p < 1.0 else float("inf")}


def cycle_period(data: Any, *, nlags: int | None = None) -> dict[str, float]:
    """Dominant cycle period from a damped-cosine fit to the autocorrelation.

    An AR(2) with complex roots has ``rho(k) = r^k cos(w k + p)``. The
    oscillatory Legendre preset seeds the angular frequency from the
    autocorrelation's spectral peak and fits that damped cosine.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        nlags: lags the fit sees; ``None`` takes ``clip(n // 2, 20, 100)``,
            capped at the image's lag budget.

    Returns:
        ``{"period", "w", "damping"}``; ``period`` is infinite when the
        recovered frequency is below one cycle in the lag window.
    """
    img = as_image(data)
    if nlags is None:
        nlags = int(np.clip(img.n // 2, 20, 100))
    nlags = int(min(nlags, img.lag))
    g = img.acov()
    rho = g[: nlags + 1] / g[0] if g[0] > 0 else g[: nlags + 1]
    k = np.arange(nlags + 1, dtype=float)
    r = fit_lsi(k, rho, "A*exp(-g*k)*cos(w*k + p)", "k",
                freq_param="w", p0=[1.0, 0.05, 0.0, np.pi / 2.0])
    gg, w = float(r.coeffs[1]), abs(float(r.coeffs[3]))
    period = (2.0 * np.pi / w if w > (2.0 * np.pi / (2.0 * nlags))
              else float("inf"))
    return {"period": float(period), "w": float(w),
            "damping": float(np.exp(-abs(gg)))}


def dickey_fuller(data: Any, *, lags: int | None = None) -> dict[str, float]:
    """Augmented Dickey-Fuller unit-root statistic from the image.

    The constant-plus-trend regression's normal equations are written in
    Toeplitz form over the image's autocovariances, so the statistic needs no
    per-sample regression.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        lags: difference lags; ``None`` takes the Schwert rule
            ``min(12 (n/100)^0.25, 12)``.

    Returns:
        ``{"tau", "pvalue"}``; MacKinnon's (1994) approximate asymptotic
        p-value for the ``ct`` regression with one series. A large p-value
        means a unit root cannot be rejected.
    """
    img = as_image(data)
    tau = img.dickey_fuller(lags=lags)
    return {"tau": float(tau), "pvalue": adf_pvalue(tau)}


_ADF_TAU_MAX = 0.7
_ADF_TAU_MIN = -16.18
_ADF_TAU_STAR = -2.89
_ADF_SMALLP = (3.2512, 1.6047, 0.049588)
_ADF_LARGEP = (2.5261, 0.61654, -0.37956, -0.060285)


def adf_pvalue(tau: float) -> float:
    """MacKinnon (1994) approximate p-value of an augmented Dickey-Fuller
    ``tau`` statistic for the constant-plus-trend regression with one series.

    Args:
        tau: the statistic; ``nan`` returns 1.0 (no rejection).
    """
    if not np.isfinite(tau):
        return 1.0
    if tau > _ADF_TAU_MAX:
        return 1.0
    if tau < _ADF_TAU_MIN:
        return 0.0
    from scipy.stats import norm
    coef = _ADF_SMALLP if tau <= _ADF_TAU_STAR else _ADF_LARGEP
    return float(norm.cdf(np.polyval(coef[::-1], tau)))


def decompose_trend_cycle(
    t: Any, y: Any = None, *, max_harmonics: int = 4, with_cycle: bool = True
) -> dict[str, object]:
    """Split a series into a deterministic trend plus seasonal cycle and a
    stochastic residual, both read from the image.

    Args:
        t: the time axis, or an :class:`~dtfit.image.Original` carrying both
            axes when ``y`` is omitted.
        y: the values when ``t`` is a time axis.
        max_harmonics: cap on the seasonal harmonics; the count is chosen by
            BIC.
        with_cycle: fit the seasonal part at all.

    Returns:
        ``{"trend", "cycle", "residual", "slope", "period", "amp",
        "noise_std", "forecast"}``: the fitted components per sample, the
        trend slope in ``t`` units, the cycle period in samples, its
        fundamental amplitude, the residual standard deviation and a
        ``forecast(h, dt=None)`` closure continuing trend plus cycle.

    Raises:
        ValueError: a non-uniform time axis, or fewer than two samples.
    """
    if y is None:
        orig = t if isinstance(t, Original) else None
        if orig is None:
            raise ValueError("give (t, y) or an Original")
        t_arr, y_arr = orig.x, orig.y
    else:
        t_arr = np.asarray(t, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        orig = Original(t_arr, y_arr)
    img = as_image(orig)
    slope, icpt = img.trend()
    trend = icpt + slope * t_arr
    cycle = np.zeros_like(t_arr)
    period = amp = float("nan")
    seas: dict[str, Any] = {"coef": np.zeros(0), "freq": 0.0, "n_harmonics": 0}
    if with_cycle and img.n >= 8:
        seas = img.seasonal(max_harmonics=max_harmonics)
        if seas["n_harmonics"]:
            idx = np.arange(img.n, dtype=float)
            cycle = seasonal_series(idx, seas["freq"], seas["coef"])
            period = float(seas["period"])
            amp = float(seas["amp"])
    noise = y_arr - trend - cycle

    def forecast(h: int, dt: float | None = None) -> np.ndarray:
        step = (float(t_arr[1] - t_arr[0])
                if (dt is None and t_arr.size > 1) else (dt or 1.0))
        tf = t_arr[-1] + step * np.arange(1, h + 1)
        idx = float(img.n - 1) + np.arange(1, h + 1, dtype=float)
        cy = (seasonal_series(idx, seas["freq"], seas["coef"])
              if seas["n_harmonics"] else np.zeros(h))
        return icpt + slope * tf + cy

    return {"trend": trend, "cycle": cycle, "residual": noise,
            "slope": float(slope), "period": period, "amp": amp,
            "noise_std": float(np.std(noise)), "forecast": forecast}


def seasonal_series(
    idx: np.ndarray, freq: float, coef: np.ndarray
) -> np.ndarray:
    """The harmonic series ``sum_j a_j cos(2 pi j f t) + b_j sin(2 pi j f t)``
    at the sample indices ``idx``, with ``coef`` the cosine/sine pairs
    :meth:`SecondOrderImage.seasonal` returns."""
    idx = np.asarray(idx, dtype=float)
    coef = np.asarray(coef, dtype=float)
    out = np.zeros_like(idx)
    for j in range(coef.size // 2):
        w = 2.0 * np.pi * (j + 1) * freq
        out = (out + coef[2 * j] * np.cos(w * idx)
               + coef[2 * j + 1] * np.sin(w * idx))
    return out
