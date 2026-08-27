"""Statistical helpers for the stochastic router, vendored so the stable
``dtfit`` wheel needs no statsmodels: a constant+trend augmented Dickey-Fuller
unit-root test matching ``statsmodels.adfuller`` to machine precision, an OLS
line fit, the spectral cycle strength, and the multi-harmonic seasonal design
and BIC fit. Pure numpy/scipy, with no dtfit dependencies."""

from __future__ import annotations

import warnings

import numpy as np


def _ols_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS line ``y ~ a + b*t``. Returns
    ``(slope, intercept, |t-stat of slope|)``.

    The slope t-stat uses a Newey-West (HAC) standard error rather than the
    classical OLS one. Autocorrelated residuals are the norm for these series,
    and the classical SE is badly anti-conservative under them: a persistent
    but stationary AR(1) shows a spuriously significant slope, precisely the
    spurious trend the router must not chase. A Bartlett-kernel HAC SE
    corrects the slope variance for that autocorrelation.
    """
    n = t.size
    tm, ym = t.mean(), y.mean()
    st = t - tm
    denom = float(st @ st)
    if denom <= 0 or n < 3:
        return 0.0, float(ym), 0.0
    slope = float(st @ (y - ym)) / denom
    intercept = float(ym - slope * tm)
    resid = y - (intercept + slope * t)
    # Newey-West HAC covariance of (intercept, slope): (X'X)^-1 S (X'X)^-1
    # with S the Bartlett-weighted autocovariance of the score x_i * u_i. At
    # lag 0 this reduces to White's HC0; higher lags add the autocorrelation
    # robustness.
    X = np.column_stack([np.ones(n), t.astype(float)])
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return slope, intercept, 0.0
    xu = X * resid[:, None]                       # (n, 2)
    S = xu.T @ xu                                 # lag 0
    bw = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West rule
    bw = max(0, min(bw, n - 1))
    for lag in range(1, bw + 1):
        w = 1.0 - lag / (bw + 1.0)                # Bartlett kernel
        g = xu[lag:].T @ xu[:-lag]               # (2, 2)
        S = S + w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else 0.0
    tstat = abs(slope) / se if se > 0 else (np.inf if slope != 0 else 0.0)
    return slope, intercept, float(tstat)


# Vendored augmented Dickey-Fuller unit-root test, with no statsmodels
# dependency. The gate is the load-bearing guard of the whole pipeline: it
# keeps a trend, cycle or long memory from being fitted to a wandering
# random-walk level, the classic spurious regression. This reproduces
# ``statsmodels.adfuller(y, regression="ct", autolag="AIC")`` to machine
# precision in pure numpy, cross-checked in the test suite. The p-value is
# MacKinnon's (1994) approximate asymptotic surface for the constant+trend
# (``ct``) regression with a single I(1) series (N = 1), and the constants
# below are his published response-surface coefficients.
_ADF_TAU_MAX = 0.7         # tau above this -> p = 1 (stationary-side cap)
_ADF_TAU_MIN = -16.18      # tau below this -> p = 0
_ADF_TAU_STAR = -2.89      # split between the small-p and large-p polynomials
_ADF_SMALLP = (3.2512, 1.6047, 0.049588)               # Phi^-1(p) = sum c_i tau^i
_ADF_LARGEP = (2.5261, 0.61654, -0.37956, -0.060285)


def _adf_pvalue(tau: float) -> float:
    """MacKinnon (1994) approximate p-value for an ADF ``tau`` stat (ct, N=1)."""
    if tau > _ADF_TAU_MAX:
        return 1.0
    if tau < _ADF_TAU_MIN:
        return 0.0
    from scipy.stats import norm
    coef = _ADF_SMALLP if tau <= _ADF_TAU_STAR else _ADF_LARGEP
    return float(norm.cdf(np.polyval(coef[::-1], tau)))


def _adf_design(x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """ADF (ct) regression design at ``lag`` difference lags. The response is
    the difference ``dy_t``; the columns are
    ``[y_{t-1}, dy_{t-1}, .., dy_{t-lag}, 1, t]``, with the level lag first so
    that its coefficient is the gamma being tested."""
    dx = np.diff(x)
    nobs = dx.size - lag
    cols = [x[lag:lag + nobs]]                          # level lag y_{t-1}
    for j in range(1, lag + 1):
        cols.append(dx[lag - j:lag - j + nobs])         # dy_{t-j}
    cols.append(np.ones(nobs))                          # const
    cols.append(np.arange(1, nobs + 1, dtype=float))    # linear trend
    return dx[lag:], np.column_stack(cols)


def _adf_tau(x: np.ndarray, maxlag: int | None = None) -> float:
    """ADF ``tau`` statistic with AIC lag selection (ct regression).

    The procedure matches statsmodels: AIC is compared on the fixed ``maxlag``
    sample, then the chosen lag is refit on its own larger sample for the
    reported t-stat. The Schwert ``maxlag`` is capped at
    ``min(12*(n/100)^.25, 12, n//3)``. The cap cuts the lag search
    several-fold without moving the unit-root verdict; the Nile, for one,
    stays trend-stationary rather than flipping to a random walk.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if maxlag is None:
        maxlag = int(min(12 * (n / 100.0) ** 0.25, 12, n // 3))
    maxlag = max(0, int(maxlag))
    # AIC selection on the fixed maxlag sample
    y0, x0 = _adf_design(x, maxlag)
    nobs = y0.size
    trend2 = x0[:, -2:]
    best_ic, best_lag = np.inf, 0
    for lag in range(0, maxlag + 1):
        cols = np.column_stack([x0[:, :lag + 1], trend2])
        beta, *_ = np.linalg.lstsq(cols, y0, rcond=None)
        resid = y0 - cols @ beta
        ssr = float(resid @ resid)
        if ssr <= 0:
            ic = -np.inf
        else:
            llf = -nobs / 2.0 * (np.log(2 * np.pi) + np.log(ssr / nobs) + 1.0)
            ic = -2.0 * llf + 2.0 * cols.shape[1]
        if ic < best_ic:
            best_ic, best_lag = ic, lag
    # Refit at the chosen lag on its own sample; tau is the t-stat on the
    # level lag.
    y1, x1 = _adf_design(x, best_lag)
    beta, *_ = np.linalg.lstsq(x1, y1, rcond=None)
    resid = y1 - x1 @ beta
    dof = y1.size - x1.shape[1]
    if dof <= 0:
        return 0.0
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se0 = np.sqrt(s2 * xtx_inv[0, 0])
    return float(beta[0] / se0) if se0 > 0 else 0.0


def _is_nonstationary(y: np.ndarray, *, alpha: float = 0.05) -> bool:
    """Unit-root gate: ``True`` when the series is I(1), a random walk, and
    must be differenced rather than have level structure fitted to it.

    The test is the vendored augmented Dickey-Fuller with a constant+trend
    regression, the trend term keeping a genuine trend-stationary series from
    being mislabelled a unit root. A degenerate regression falls back to a
    near-unit AR(1) coefficient on the level.
    """
    n = y.size
    if n < 12:
        return False
    try:
        return _adf_pvalue(_adf_tau(np.asarray(y, dtype=float))) > alpha
    except Exception as exc:
        warnings.warn(
            f"stochastic stage unit-root (ADF) failed: {exc}; falling back to "
            "the near-unit AR(1) coefficient gate", UserWarning, stacklevel=2)
        b = float(np.polyfit(y[:-1], y[1:], 1)[0])
        return b > 0.98


def _cycle_strength(y: np.ndarray, *, min_period: int = 4) -> tuple[float, float]:
    """Dominant interior spectral peak of a linearly-detrended series, as
    ``(period, strength)``. ``strength`` is the peak's share of the detrended
    power."""
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    tt = np.arange(n)
    yc = y - np.polyval(np.polyfit(tt, y, 1), tt)
    spec = np.abs(np.fft.rfft(yc)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    valid = (freqs > 1.0 / n) & (freqs <= 1.0 / min_period)
    tot = float(spec[1:].sum())
    if not valid.any() or tot == 0:
        return float("nan"), 0.0
    k = int(np.argmax(np.where(valid, spec, 0.0)))
    strength = float(spec[k] / tot)
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength


def _seasonal_design(t: np.ndarray, period: float, n_harmonics: int) -> np.ndarray:
    """Fourier design matrix: ``n_harmonics`` cos/sin pairs at the seasonal
    frequency and its harmonics. The harmonics are what make non-sinusoidal
    seasonal shapes representable, such as the CO2 sawtooth or the pulse of
    the sunspot cycle."""
    cols = []
    for j in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * j / period
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def _fit_seasonal(t: np.ndarray, y: np.ndarray, period: float,
                  max_harmonics: int) -> tuple[int, np.ndarray]:
    """Fit a multi-harmonic Fourier seasonal model by linear least squares,
    picking the number of harmonics by BIC. Fewer harmonics generalize better;
    the penalty is what stops the forecast overfitting the seasonal shape.
    Returns ``(K, coef)``."""
    n = y.size
    best = None
    for k in range(1, max_harmonics + 1):
        if 2 * k >= n:
            break
        x = _seasonal_design(t, period, k)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        rss = float(resid @ resid)
        bic = n * np.log(rss / n + 1e-12) + (2 * k) * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, k, coef)
    assert best is not None
    return best[1], best[2]
