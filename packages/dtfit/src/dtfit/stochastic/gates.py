"""The batch entry point: the gate sequence that routes a series to a regime,
every gate reading the second-order image."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dtfit._pandas import as_series, capture_index, extend_index
from .image import SecondOrderImage
from .estimators import (
    as_image, adf_pvalue, seasonal_series, _persistence,
)
from .forecast import (
    bind_forecaster, resolve_forecaster, fc_rw, fc_drift, fc_meanrev,
    fc_trend, make_seasonal_fc, make_seasonal_fc_anchored,
)
from .simulate import _sim_ar1, _sim_long_memory, _sim_garch, make_innovations

__all__ = ["StochasticModel", "fit_stochastic", "is_nonstationary"]


def bt_spectrum(
    g: np.ndarray, n: int, n_freq: int
) -> tuple[np.ndarray, np.ndarray]:
    """Blackman-Tukey spectrum of an autocovariance sequence under a Parzen
    lag window, at ``f = 1/n .. n_freq/n`` cycles per sample.

    Args:
        g: autocovariance sequence, ``g[0..lag]``.
        n: record length in samples, at least 1.
        n_freq: number of frequency bins to evaluate, at least 1.

    Returns:
        ``(f, s)``: the frequencies in cycles per sample and the spectrum
        at each, both length ``n_freq``.
    """
    lag = g.size - 1
    k = np.arange(lag + 1)
    u = k / lag
    w = np.where(u <= 0.5, 1 - 6 * u ** 2 + 6 * u ** 3, 2 * (1 - u) ** 3)
    f = np.arange(1, int(n_freq) + 1) / n
    s = g[0] + 2 * (w[1:] * g[1:]) @ np.cos(2 * np.pi * np.outer(k[1:], f))
    return f, s


def gph_slope(f: np.ndarray, s: np.ndarray) -> float:
    """GPH log-periodogram slope of a spectrum: ``log S = c - 2d log|2 sin
    pi f|``.

    Args:
        f: frequencies in cycles per sample.
        s: spectrum at each frequency, same shape as ``f``.

    Returns:
        ``-2d``, the slope of the least-squares fit over the bins where
        ``s`` is positive; ``0.0`` when fewer than 3 remain.
    """
    keep = s > 0
    if keep.sum() < 3:
        return 0.0
    lf = np.log(2.0 * np.sin(np.pi * f[keep]))
    return float(np.polyfit(lf, np.log(s[keep]), 1)[0])


def yule_walker(rho: np.ndarray, p: int) -> np.ndarray:
    """AR(``p``) coefficients from an autocorrelation sequence.

    Args:
        rho: autocorrelation sequence, ``rho[0..p]`` at least, ``rho[0] ==
            1``.
        p: AR order, at least 0.

    Returns:
        The ``p`` coefficients ``phi_1..phi_p``; an empty array when ``p``
        is below 1.
    """
    from scipy.linalg import toeplitz

    if p < 1:
        return np.zeros(0)
    return np.linalg.solve(toeplitz(rho[:p]), rho[1:p + 1])


def whitened_spectrum(
    g: np.ndarray, n: int, n_freq: int, phi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Spectrum of the AR-whitened residual: the residual spectrum divided by
    the AR transfer function ``|1 - sum phi_j exp(-2 pi i f j)|^-2``.

    Args:
        g: autocovariance sequence, ``g[0..lag]``.
        n: record length in samples, at least 1.
        n_freq: number of frequency bins to evaluate, at least 1.
        phi: AR coefficients whitening the series; an empty array leaves
            the spectrum unwhitened.

    Returns:
        ``(f, s)`` as :func:`bt_spectrum`, over the whitened spectrum.
    """
    f, s = bt_spectrum(g, n, n_freq)
    if phi.size == 0:
        return f, s
    j = np.arange(1, phi.size + 1)
    h = 1.0 - np.exp(-2j * np.pi * np.outer(f, j)) @ phi
    return f, s * np.abs(h) ** 2


def index_lag_weights(n: int, bw: int) -> np.ndarray:
    """``A_k = sum_{t >= k} (t - tbar)(t - k - tbar)`` for ``k = 0..bw`` over
    the indices ``t = 0..n-1``, ``tbar = (n - 1) / 2``.

    Args:
        n: record length in samples, at least 1.
        bw: largest lag, at least 0 and below ``n``.

    Returns:
        The ``bw + 1`` weights, in closed form: the running index sums are
        cubics in ``n`` and ``k``, so the cost and the memory are ``O(bw)``
        and neither grows with the record.
    """
    out = np.empty(int(bw) + 1)
    st = (n - 1) * n // 2
    tt = (n - 1) * n * (2 * n - 1) // 6
    tb = (n - 1) / 2.0
    for k in range(int(bw) + 1):
        s1 = st - (k - 1) * k // 2
        s2 = tt - (k - 1) * k * (2 * k - 1) // 6
        out[k] = s2 - (k + 2.0 * tb) * s1 + tb * (k + tb) * (n - k)
    return out


def trend_tstat(img: SecondOrderImage, g: np.ndarray) -> float:
    """``|t|`` of the least-squares slope with the residual autocorrelation
    carried, the image form of a heteroskedasticity- and
    autocorrelation-consistent standard error.

    ``Var(slope) = sum_{t,s} (t - tbar)(s - tbar) gamma(t - s) / S_tt^2`` with
    a Bartlett taper on ``gamma`` at the Newey-West bandwidth; the double sum
    closes in ``n`` and the lag through
    ``A_k = sum_{t >= k} (t - tbar)(t - k - tbar)``, so the read-out holds
    ``bw + 1`` numbers and no array over the record.

    Args:
        img: the image of the series.
        g: the residual autocovariance, ``gamma[0..lag]``.
    """
    n = img.n
    slope, _ = img._trend_index()
    stt = img.sum_t2 - img.sum_t ** 2 / n
    if stt <= 0:
        return 0.0
    bw = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    bw = int(max(0, min(bw, g.size - 1, n - 1)))
    a = index_lag_weights(n, bw)
    var = g[0] * a[0]
    for k in range(1, bw + 1):
        w = 1.0 - k / (bw + 1.0)
        var += 2.0 * w * g[k] * a[k]
    se = np.sqrt(max(var, 0.0)) / stt
    if se <= 0.0:
        return float("inf") if slope != 0.0 else 0.0
    return abs(slope) / se


def is_nonstationary(data: Any, *, alpha: float = 0.05) -> bool:
    """Unit-root gate: ``True`` when the series is I(1) and must be
    differenced rather than have level structure fitted to it.

    The augmented Dickey-Fuller statistic of the constant-plus-trend
    regression is computed from the image's autocovariances and read through
    MacKinnon's p-value surface.

    The augmentation lag is chosen by AIC rather than fixed, so the false
    positive rate stays a few percent from n=100 up: measured over white
    noise and AR(1) phi=0.5 (the worse of the two, 60 seeds each), the
    rate at which this gate wrongly returns True is n=40: 12%, n=100: 3%,
    n=200: 0%, n=400: 0%. Below n=40 the same rate climbs fast (n=20: 35%,
    n=24: 30%, n=28: 23%, n=32: 20%, n=36: 23%), so below ``n = 40`` the
    gate reports "not nonstationary" rather than trust a verdict from
    there.

    Args:
        data: a series, an Original or a :class:`SecondOrderImage`.
        alpha: significance level; a p-value above it means a unit root
            cannot be rejected.
    """
    img = as_image(data)
    if img.n < 40:
        return False
    return adf_pvalue(img.dickey_fuller()) > alpha


def excess_squared_acf(g2: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Autocorrelation of a squared series with the contribution of the
    series it squares removed.

    A Gaussian linear process with autocorrelation ``rho`` has squared-series
    autocorrelation ``rho^2`` and no volatility clustering of its own; what is
    left after subtracting it is the ARCH-type structure.

    Args:
        g2: autocovariance of the squares -- of the level on the stationary
            branch, of the increments on the unit-root one.
        rho: autocorrelation of the series that was squared, the same lags.

    Returns:
        The excess autocorrelation, ``1.0`` at lag 0, zeros when ``g2[0]`` is
        not positive.
    """
    if g2[0] <= 0.0:
        return np.zeros_like(g2)
    r2 = g2 / g2[0]
    k = min(r2.size, rho.size)
    out = r2.copy()
    out[:k] -= rho[:k] ** 2
    out[0] = 1.0
    return out


@dataclass
class StochasticModel:
    """A unified second-order characterization of a stochastic series.

    Produced by :func:`fit_stochastic`. It records which structural
    components the gates opened, the recovered parameter of each, the primary
    ``regime`` label for the level dynamics, and a :meth:`forecast` composing
    the deterministic mean with the stochastic level model.

    ``_mean_fn`` evaluates the fitted deterministic mean (trend plus
    seasonal) at a sample-index array; it is what :meth:`simulate` adds the
    stochastic residual to. ``_index`` is the pandas index the training data
    carried, or ``None`` for an ndarray fit; it is what :meth:`forecast`
    extends to label the output.
    """

    n: int
    level: float
    trend_slope: float                  # in t units (dt-scaled), not samples
    has_trend: bool
    cycle_period: float                 # in samples of the sample index
    cycle_amp: float
    has_cycle: bool
    n_harmonics: int
    seasonal: bool
    hurst: float
    has_long_memory: bool
    ar1_phi: float
    has_mean_reversion: bool
    vol_persistence: float
    has_vol_clustering: bool
    sigma: float                        # one-step innovation std (level)
    sigma_walk: float                   # std of first differences
    components: tuple[str, ...]
    regime: str
    forecaster_name: str
    _forecaster: Callable[[int], np.ndarray] = field(
        repr=False, default=lambda h: np.zeros(h))
    _mean_fn: Callable[[np.ndarray], np.ndarray] = field(
        repr=False,
        default=lambda t: np.zeros_like(np.asarray(t, dtype=float)))
    _index: Any = field(repr=False, default=None)

    def fingerprint(self) -> dict[str, object]:
        """The detected structure as a flat ``{name: value}`` dict."""
        return {
            "regime": self.regime,
            "components": ", ".join(self.components),
            "trend slope": (self.trend_slope if self.has_trend
                            else float("nan")),
            "cycle period": (self.cycle_period if self.has_cycle
                             else float("nan")),
            "Hurst H": self.hurst if self.has_long_memory else float("nan"),
            "AR(1) phi": (self.ar1_phi if self.has_mean_reversion
                          else float("nan")),
            "vol persistence": self.vol_persistence if self.has_vol_clustering
            else float("nan"),
            "forecaster": self.forecaster_name,
        }

    def summary(self) -> str:
        """A multi-line report of the detected structure."""
        lines = [f"StochasticModel  regime={self.regime!r}  n={self.n}",
                 f"  components: {', '.join(self.components)}"]
        if self.has_trend:
            lines.append(f"  trend       slope = {self.trend_slope:.4g}")
        if self.has_cycle:
            kind = "seasonal" if self.seasonal else "cycle"
            lines.append(f"  {kind:<11} period = {self.cycle_period:.4g} "
                         f"({self.n_harmonics} harmonic(s))")
        if self.has_long_memory:
            lines.append(f"  long memory H = {self.hurst:.3f} "
                         f"(d = {self.hurst - 0.5:.3f})")
        if self.has_mean_reversion:
            tau = (-1.0 / np.log(self.ar1_phi) if 0 < self.ar1_phi < 1
                   else np.inf)
            lines.append(f"  mean revert phi = {self.ar1_phi:.3f} "
                         f"(tau = {tau:.1f})")
        if self.has_vol_clustering:
            lines.append("  volatility  persistence = "
                         f"{self.vol_persistence:.3f}")
        lines.append(f"  innovation sigma = {self.sigma:.4g}")
        lines.append(f"  forecaster: {self.forecaster_name}")
        return "\n".join(lines)

    def forecast(self, h: int, *, return_conf_int: bool = False,
                 alpha: float = 0.05, dist: str = "normal", df: float = 7.0):
        """Forecast ``h`` steps with the backtest-selected forecaster named
        in :attr:`forecaster_name`.

        Args:
            h: steps ahead, at least 1. ``h`` counts samples, index steps of
                the fitted series, whatever the units of the time axis ``t``
                the model was fit with.
            return_conf_int: return ``(point, lower, upper)`` instead of the
                point forecast alone.
            alpha: two-sided band level, ``0 < alpha < 1``; 0.05 is a
                95 percent band.
            dist: band quantile, ``"normal"`` or ``"t"``; ``"t"`` widens the
                band with a Student-t quantile for fat-tailed innovations.
            df: degrees of freedom of that Student-t, above 2.

        Returns:
            The point forecast, and with ``return_conf_int`` the lower and
            upper bands, whose growth matches the chosen forecaster: bounded
            for mean reversion, ``h^(2H)`` for long memory, ``~sqrt(h)`` for a
            random walk or drift, and roughly constant for a trend-stationary
            trend or seasonal forecast. A model fit on a pandas ``Series``
            returns each as a ``Series`` labelled by the length-``h`` future
            index continuing the training index -- a ``DatetimeIndex`` at its
            frequency, an integer-like index by its step; a model fit on an
            ndarray returns ndarrays.

        Raises:
            ValueError: ``dist`` is neither ``"normal"`` nor ``"t"``.
        """
        steps = np.arange(1, h + 1, dtype=float)
        point = np.asarray(self._forecaster(h), dtype=float)
        fidx = (extend_index(self._index, h)
                if self._index is not None else None)
        if not return_conf_int:
            return as_series(point, fidx)
        from scipy.stats import norm, t as student_t
        name = self.forecaster_name
        if name.startswith("mean-reversion") and 0.0 < self.ar1_phi < 1.0:
            var = (self.sigma ** 2) * (1.0 - self.ar1_phi ** (2 * steps)) \
                / (1.0 - self.ar1_phi ** 2)
        elif name.startswith(("random walk", "drift")):
            if self.has_long_memory and 0.5 < self.hurst < 1.0:
                var = (self.sigma_walk ** 2) * steps ** (2.0 * self.hurst)
            else:
                var = (self.sigma_walk ** 2) * steps
        else:
            var = (self.sigma ** 2) * np.ones_like(steps)
        if dist == "t":
            z = float(student_t.ppf(1.0 - alpha / 2.0, df))
        elif dist == "normal":
            z = float(norm.ppf(1.0 - alpha / 2.0))
        else:
            raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
        sd = np.sqrt(np.clip(var, 0.0, None))
        return (as_series(point, fidx), as_series(point - z * sd, fidx),
                as_series(point + z * sd, fidx))

    def simulate(self, n: int | None = None, *, seed: int | None = None,
                 rng: np.random.Generator | None = None,
                 dist: str = "normal", df: float = 7.0) -> np.ndarray:
        """Draw a fresh realization of length ``n`` from the fitted model.

        The generative side of :class:`StochasticModel`: it composes the
        detected deterministic mean, trend plus multi-harmonic seasonal, with
        a stochastic residual drawn to match the detected second-order regime:

        * unit-root: an integrated random walk, drift plus innovations, with
          GARCH volatility clustering on the increments where detected;
        * mean-reverting: a stationary AR(1) residual at the recovered phi;
        * long-memory: an ARFIMA(0, d, 0) residual at ``d = H - 1/2``;
        * vol-clustering: a GARCH(1,1) residual at the recovered persistence;
        * white noise: i.i.d. innovations.

        Re-fitting the simulated path recovers the same regime and
        parameters.

        Args:
            n: length of the realization in samples, at least 1; ``None``
                takes the fitted ``n``.
            seed: seed of a fresh generator, used when ``rng`` is ``None``.
            rng: generator to draw from; takes precedence over ``seed``.
            dist: innovation distribution, ``"normal"`` or ``"t"``; ``"t"``
                draws a unit-variance Student-t, a fat-tailed generator for
                financial-style tail risk.
            df: degrees of freedom of that Student-t, above 2.

        Returns:
            The realization, ``(n,)``. The deterministic mean is evaluated
            from the fitted record first sample onward, so a seasonal path
            starts in the phase the record started in.

        Raises:
            ValueError: ``dist`` is neither ``"normal"`` nor ``"t"``.
        """
        n = int(self.n if n is None else n)
        if rng is None:
            rng = np.random.default_rng(seed)
        noise = make_innovations(dist, df)
        sig = (self.sigma if np.isfinite(self.sigma) and self.sigma > 0.0
               else (self.sigma_walk if np.isfinite(self.sigma_walk)
                     and self.sigma_walk > 0.0 else 1.0))
        if "unit-root" in self.components:
            if self.has_vol_clustering and np.isfinite(self.vol_persistence):
                steps = _sim_garch(n, self.vol_persistence, sig, rng,
                                   noise=noise)
            else:
                steps = noise(rng, n) * sig
            drift = self.trend_slope if np.isfinite(self.trend_slope) else 0.0
            return self.level + np.cumsum(drift + steps)
        t = np.arange(n, dtype=float)
        mean = np.asarray(self._mean_fn(t), dtype=float)
        if mean.shape != t.shape:
            mean = np.full(n, float(self.level))
        if self.has_long_memory and np.isfinite(self.hurst):
            resid = _sim_long_memory(n, self.hurst, sig, rng, noise=noise)
        elif self.has_mean_reversion and 0.0 < self.ar1_phi < 1.0:
            resid = _sim_ar1(n, self.ar1_phi, sig, rng, noise=noise)
        elif self.has_vol_clustering and np.isfinite(self.vol_persistence):
            resid = _sim_garch(n, self.vol_persistence, sig, rng, noise=noise)
        else:
            resid = noise(rng, n) * sig
        return mean + resid


def fit_stochastic(
    data: Any,
    t: Any = None,
    *,
    period: float | None = None,
    max_harmonics: int = 4,
    forecaster: object = "auto",
    trend_t: float = 3.0,
    cycle_strength: float = 0.08,
    min_cycles: float = 2.5,
    lm_hurst: float = 0.68,
    mr_phi: float = 0.15,
    vol_persist: float = 0.60,
    lag: int = 256,
    nfreq: int = 512,
) -> StochasticModel:
    """Characterize a series across every route at once and return one model.

    Every gate reads the second-order image of the series, in order: the
    unit-root gate (:func:`is_nonstationary`) routes to a random-walk-plus-
    drift-plus-GARCH model and returns early; otherwise a deterministic
    trend and seasonal cycle are fitted and removed, then the residual is
    tested for long memory (vetoed by a finite AR fit), mean reversion and
    volatility clustering, each opening a component of the returned model.

    Args:
        data: a series, an :class:`~dtfit.image.original.Original`, or a
            :class:`SecondOrderImage`. An image skips the deterministic-mean
            reconstruction that needs the raw values (the mean function and
            any callable forecaster) and cannot be backtest-selected.
        t: the time axis when ``data`` is a plain series; ``None`` (the
            default) takes a uniform unit index. Ignored when ``data`` is
            already a :class:`SecondOrderImage`.
        period: seasonal period in samples of the record (index steps of
            the series, not ``t`` units, so it does not change with the
            time axis's scale); ``None`` detects it from the spectrum.
        max_harmonics: cap on the Fourier harmonics of the seasonal
            component, at least 1.
        forecaster: forecast selection control: ``"auto"`` backtests every
            regime-appropriate candidate and keeps the best one within a
            margin of the random walk; a name from
            :data:`~dtfit.stochastic.forecast.FORECASTERS`; a callable
            ``(train, h) -> array`` fitted fresh each backtest fold; or a
            list mixing names, callables and ``(name, callable)`` pairs,
            backtest-selected among themselves.
        trend_t: minimum ``|t|`` of the trend-slope Newey-West statistic to
            call a deterministic trend, above 0.
        cycle_strength: minimum fundamental energy share (0-1) of the
            spectrum to call a periodic cycle.
        min_cycles: minimum number of cycles the record must span,
            ``n / period >= min_cycles``, above 0.
        lm_hurst: minimum Hurst exponent (0.5-1) to call long memory.
        mr_phi: minimum AR(1) coefficient (0-1) to call mean reversion.
        vol_persist: minimum GARCH persistence (0-1) to call volatility
            clustering.
        lag: autocovariance lag budget of the image built from ``data``, in
            samples; ignored when ``data`` is already an image.
        nfreq: spectral grid floor of that image, in frequency bins;
            ignored when ``data`` is already an image.

    Returns:
        The fitted :class:`StochasticModel`.

    Raises:
        ValueError: ``forecaster`` names an unknown built-in, or is an empty
            candidate list.
        TypeError: ``forecaster`` is a callable or contains one and ``data``
            is a :class:`SecondOrderImage` with no series to fit it on; or a
            candidate is neither a name, a callable nor a ``(name,
            callable)`` pair.

    Warns:
        UserWarning: a backtest cannot run -- fewer than 51 training samples
            falls back to the first candidate, and an image with no series
            takes the last one without scoring it; a backtest candidate that
            raises is scored infinite for that fold; and the long-memory
            stage or either volatility-clustering stage (unit-root or
            stationary) leaves its component off and reports the exception
            it caught rather than raise out of the fit.
    """
    index = capture_index(data)
    img = as_image(data, lag=lag, nfreq=nfreq)
    if t is not None and not isinstance(data, SecondOrderImage):
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        dt = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1.0
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0
        img.x0 = float(t_arr[0])
        img.dx = dt
    dt = img.dx
    n = img.n
    y_raw = None
    if not isinstance(data, SecondOrderImage):
        from dtfit._pandas import to_1d_array
        from dtfit.image.original import Original
        y_raw = (data.y if isinstance(data, Original)
                 else to_1d_array(data, "y"))
    level = img.mean()
    g_inc = img.acov_increments()
    sigma_walk = float(np.sqrt(max(g_inc[0], 0.0)))
    band = 2.0 / np.sqrt(max(n, 1))

    seas = img.seasonal(max_harmonics=max_harmonics)
    if period is not None:
        pre_cyclical = True
    else:
        pre_cyclical = (seas["strength"] > 0.12 and np.isfinite(seas["period"])
                        and 4 <= seas["period"] <= n / 5.0)

    if is_nonstationary(img) and not pre_cyclical:
        drift = (img.last() - img.first()) / max(n - 1, 1)
        drift_sig = abs(drift) > 2.0 * (sigma_walk / np.sqrt(max(n - 1, 1)))
        vol = float("nan")
        has_vol = False
        try:
            rho_dy = (g_inc / g_inc[0] if g_inc[0] > 0
                      else np.zeros_like(g_inc))
            rxd = excess_squared_acf(img.acov_volatility(), rho_dy)
            if n > 3 and float(rxd[1]) > band:
                vol = _persistence(rxd, n - 1, None, "lsi")["persistence"]
                has_vol = vol > vol_persist
        except Exception as exc:
            warnings.warn(
                f"stochastic stage unit-root vol-clustering failed: {exc}",
                UserWarning, stacklevel=2)
        comps = (["unit-root"] + (["drift"] if drift_sig else [])
                 + (["vol-clustering"] if has_vol else []))
        regime = "random walk + drift" if drift_sig else "random walk"
        ur_per = float(period) if period is not None else float(seas["period"])
        fname, ffn = resolve_forecaster(
            forecaster, [("random walk", fc_rw), ("drift", fc_drift)],
            per=ur_per, max_harmonics=max_harmonics, img=img, y=y_raw)
        return StochasticModel(
            n=n, level=level, trend_slope=drift if drift_sig else 0.0,
            has_trend=False,
            cycle_period=float("nan"), cycle_amp=float("nan"), has_cycle=False,
            n_harmonics=0, seasonal=False,
            hurst=float("nan"), has_long_memory=False,
            ar1_phi=float("nan"), has_mean_reversion=False,
            vol_persistence=vol, has_vol_clustering=has_vol,
            sigma=sigma_walk, sigma_walk=sigma_walk,
            components=tuple(comps), regime=regime, forecaster_name=fname,
            _forecaster=bind_forecaster(ffn, img), _index=index,
        )

    # deterministic mean: the trend, then the seasonal cycle
    g_detr = img.detrended_acov()
    slope_x, icpt_x = img.trend()
    tstat = trend_tstat(img, g_detr)
    var_y = img.acov()[0]
    trend_r2 = 1.0 - (g_detr[0] / var_y if var_y > 0 else 1.0)
    has_trend = (tstat > trend_t) and (trend_r2 > 0.10)
    if not has_trend:
        slope_x = 0.0
        g_detr = img.acov()

    if period is not None:
        per = float(period)
        seas = img.seasonal(max_harmonics=max_harmonics, freq=1.0 / per)
        has_cycle = 4 <= per <= n / 2.0
    else:
        per = float(seas["period"])
        has_cycle = (seas["strength"] > cycle_strength and np.isfinite(per)
                     and 4 <= per <= n / min_cycles)
    cyc_amp = float(seas["amp"]) if has_cycle else float("nan")
    n_harm = int(seas["n_harmonics"]) if has_cycle else 0
    coef = np.asarray(seas["coef"], dtype=float) if has_cycle else np.zeros(0)
    seas_r2 = float(seas["strength"]) if has_cycle else 0.0
    if has_cycle and has_trend:
        g_e = img.residual_acov(seas["freq"], coef)
    elif has_cycle:
        g_e = (img.residual_acov(seas["freq"], coef)
               + (img.acov() - img.detrended_acov()))
    else:
        g_e = g_detr
    is_seasonal = has_cycle and (period is not None or seas_r2 > 0.30)

    rho = g_e / g_e[0] if g_e[0] > 0 else np.zeros_like(g_e)
    phi = float(np.clip(rho[1], 1e-6, 0.999999))
    acf1 = float(rho[1])
    has_mr = (mr_phi < phi < 0.99) and (abs(acf1) > band)
    sigma = float(np.sqrt(max(g_e[0] * (1.0 - phi * phi), 0.0))) if has_mr \
        else float(np.sqrt(max(g_e[0], 0.0)))

    # long memory on the residual spectrum, vetoed by a finite-order AR
    hurst = float("nan")
    has_lm = False
    try:
        nf = max(8, int(n ** 0.6))
        f, s = bt_spectrum(g_e, n, nf)
        hurst = -gph_slope(f, s) / 2.0 + 0.5
        has_lm = hurst > lm_hurst
        if has_lm:
            p = _ar_order_from_rho(rho, n, 3)
            if p >= 1:
                phi_p = yule_walker(rho, p)
                fw, sw = whitened_spectrum(g_e, n, nf, phi_p)
                veto_h = 0.5 + 0.5 * (lm_hurst - 0.5)
                if n > 128 and (-gph_slope(fw, sw) / 2.0 + 0.5) <= veto_h:
                    has_lm = False
                    has_mr = True
    except Exception as exc:
        warnings.warn(f"stochastic stage Hurst/long-memory failed: {exc}",
                      UserWarning, stacklevel=2)

    # volatility clustering: the squared series with the level's own
    # contribution removed
    vol = float("nan")
    has_vol = False
    try:
        rx = excess_squared_acf(img.acov_squares(), rho)
        if n > 2 and float(rx[1]) > band:
            vol = _persistence(rx, n, None, "lsi")["persistence"]
            has_vol = vol > vol_persist
    except Exception as exc:
        warnings.warn(f"stochastic stage GARCH/vol-clustering failed: {exc}",
                      UserWarning, stacklevel=2)

    comps = []
    if has_trend:
        comps.append("trend")
    if has_cycle:
        comps.append("seasonal" if is_seasonal else "cycle")
    if has_lm:
        comps.append("long-memory")
    if has_mr:
        comps.append("mean-reversion")
    if has_vol:
        comps.append("vol-clustering")
    if has_trend and has_cycle:
        regime = "trend+seasonal" if is_seasonal else "trend+cycle"
    elif has_cycle:
        regime = "seasonal" if is_seasonal else "cyclical"
    elif has_trend:
        regime = "trend"
    elif has_lm:
        regime = "long-memory"
    elif has_mr:
        regime = "mean-reverting"
    elif has_vol:
        regime = "white-mean / vol-clustering"
    else:
        regime = "white noise / random walk"

    sel_margin = 0.98
    if has_cycle:
        seasonal_name = "trend+seasonal" if has_trend else "seasonal"
        candidates = [
            ("random walk", fc_rw),
            (seasonal_name, make_seasonal_fc(per, max_harmonics, has_trend)),
            (seasonal_name + " (anchored)",
             make_seasonal_fc_anchored(per, max_harmonics, has_trend)),
        ]
        sel_margin = 1.15
    elif has_trend:
        candidates = [("random walk", fc_rw), ("drift", fc_drift),
                      ("trend", fc_trend)]
        sel_margin = 1.15
    elif has_mr:
        candidates = [("random walk", fc_rw), ("mean-reversion", fc_meanrev)]
    else:
        candidates = [("random walk", fc_rw)]
    fname, ffn = resolve_forecaster(forecaster, candidates, per=per,
                                    max_harmonics=max_harmonics, img=img,
                                    y=y_raw, margin=sel_margin)

    def _mean_fn(tt, _a=icpt_x, _b=slope_x, _f=float(seas["freq"]),
                 _co=coef, _t0=img.x0 + img.dx * img.t0, _dt=dt,
                 _i0=img.t0):
        idx = np.asarray(tt, dtype=float)
        m = _a + _b * (_t0 + _dt * idx)
        if _co.size:
            m = m + seasonal_series(_i0 + idx, _f, _co)
        return m

    return StochasticModel(
        n=n, level=level, trend_slope=float(slope_x),
        has_trend=bool(has_trend),
        cycle_period=float(per) if has_cycle else float("nan"),
        cycle_amp=float(cyc_amp), has_cycle=bool(has_cycle),
        n_harmonics=int(n_harm), seasonal=bool(is_seasonal),
        hurst=hurst, has_long_memory=bool(has_lm),
        ar1_phi=float(phi), has_mean_reversion=bool(has_mr),
        vol_persistence=vol, has_vol_clustering=bool(has_vol),
        sigma=float(sigma), sigma_walk=float(sigma_walk),
        components=tuple(comps) if comps else ("none",), regime=regime,
        forecaster_name=fname, _forecaster=bind_forecaster(ffn, img),
        _mean_fn=_mean_fn, _index=index,
    )


def _ar_order_from_rho(rho: np.ndarray, n: int, max_order: int) -> int:
    """AR order by AIC from an autocorrelation sequence."""
    from scipy.linalg import toeplitz

    best_k, best = 0, n * np.log(1.0) + 2.0
    for k in range(1, int(max_order) + 1):
        try:
            phi = np.linalg.solve(toeplitz(rho[:k]), rho[1:k + 1])
        except np.linalg.LinAlgError:
            continue
        s2 = 1.0 - float(phi @ rho[1:k + 1])
        if s2 <= 0.0:
            continue
        score = n * float(np.log(s2)) + 2.0 * (k + 1)
        if score < best:
            best_k, best = k, score
    return best_k
