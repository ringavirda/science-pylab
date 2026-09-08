"""Batch stochastic-series entry point.

A genuinely random series has no ``y = f(t; theta)`` to fit, but its
deterministic functionals do have known forms: damped exponentials and cosines
for the autocovariance of an ARMA / OU process, ``S(f) ~ c f^{-2d}`` for a
long-memory spectrum near zero, ``Var(block mean at scale m) ~ c m^{2H-2}``
for aggregated variance, a structural curve for the conditional mean. The
estimators alongside feed each functional to :func:`dtfit.fit_lsi` or
:func:`dtfit.fit_eac`; :mod:`dtfit.stochastic` carries the theory.

:func:`fit_stochastic` composes those routes behind significance gates into a
single :class:`StochasticModel` that characterizes, forecasts and generates an
arbitrary series. The streaming counterpart is
:class:`~dtfit.stochastic.StochasticFilter`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from dtfit._pandas import as_series, capture_index, extend_index, to_1d_array
from dtfit.methods import fit_lsi
from ._estimators import (
    sample_acf, hurst_spectral, ar1_reversion, garch_persistence,
    ar_order, fit_ar,
)
from ._stats import (
    _is_nonstationary, _ols_line, _cycle_strength, _fit_seasonal, _seasonal_design,
)
from ._forecast import (
    _bind_forecaster, _fc_rw, _fc_drift, _fc_meanrev, _fc_trend,
    _make_seasonal_fc, _make_seasonal_fc_anchored, _resolve_forecaster,
)
from .simulate import (
    _sim_ar1, _sim_long_memory, _sim_garch, make_innovations,
)

__all__ = ["StochasticModel", "fit_stochastic"]


@dataclass
class StochasticModel:
    """A unified second-order characterization of a stochastic series.

    Produced by :func:`fit_stochastic`. It records which structural components
    were detected, each behind a significance gate so that white noise yields
    none, the recovered parameter of each, the primary ``regime`` label for the
    level dynamics, and a :meth:`forecast` that composes the deterministic mean
    with the stochastic mean reversion and emits regime-appropriate prediction
    intervals.
    """

    n: int
    level: float
    # deterministic mean (1st order)
    trend_slope: float
    has_trend: bool
    cycle_period: float   # in samples (index steps of y), not in t units
    cycle_amp: float
    has_cycle: bool
    n_harmonics: int
    seasonal: bool
    # stochastic structure (2nd order)
    hurst: float
    has_long_memory: bool
    ar1_phi: float
    has_mean_reversion: bool
    vol_persistence: float
    has_vol_clustering: bool
    sigma: float          # one-step innovation std (level)
    sigma_walk: float     # std of first differences (random-walk scale)
    components: tuple[str, ...]
    regime: str
    forecaster_name: str   # the candidate model chosen by backtest selection
    _forecaster: Callable[[int], np.ndarray] = field(
        repr=False, default=lambda h: np.zeros(h))
    # Deterministic mean (trend + multi-harmonic seasonal) as a function of the
    # sample index 0..n-1, mapping that index onto the fitted time axis
    # internally. :meth:`simulate` uses it to regenerate the structured part of
    # the series. The flat-zero default belongs to the unit-root branch, whose
    # mean is the integrated random walk rather than a function of ``t``.
    _mean_fn: Callable[[np.ndarray], np.ndarray] = field(
        repr=False,
        default=lambda t: np.zeros_like(np.asarray(t, dtype=float)))
    # pandas index of the training Series/DataFrame, ``None`` for ndarray
    # input. :meth:`forecast` labels the horizon with the continuing future
    # index. The ndarray path never consults it and stays bit-identical.
    _index: Any = field(repr=False, default=None)

    def fingerprint(self) -> dict[str, object]:
        """The detected structure as a flat ``{name: value}`` dict, for
        tabulating."""
        return {
            "regime": self.regime,
            "components": ", ".join(self.components),
            "trend slope": self.trend_slope if self.has_trend else float("nan"),
            "cycle period": self.cycle_period if self.has_cycle else float("nan"),
            "Hurst H": self.hurst if self.has_long_memory else float("nan"),
            "AR(1) phi": self.ar1_phi if self.has_mean_reversion else float("nan"),
            "vol persistence": self.vol_persistence if self.has_vol_clustering
            else float("nan"),
            "forecaster": self.forecaster_name,
        }

    def summary(self) -> str:
        lines = [f"StochasticModel  regime={self.regime!r}  n={self.n}",
                 f"  components: {', '.join(self.components)}"]
        if self.has_trend:
            lines.append(f"  trend       slope = {self.trend_slope:.4g}")
        if self.has_cycle:
            kind = "seasonal" if self.seasonal else "cycle"
            lines.append(f"  {kind:<11} period = {self.cycle_period:.4g} "
                         f"({self.n_harmonics} harmonic(s))")
        if self.has_long_memory:
            lines.append(f"  long memory H = {self.hurst:.3f} (d = {self.hurst - 0.5:.3f})")
        if self.has_mean_reversion:
            tau = -1.0 / np.log(self.ar1_phi) if 0 < self.ar1_phi < 1 else np.inf
            lines.append(f"  mean revert phi = {self.ar1_phi:.3f} (tau = {tau:.1f})")
        if self.has_vol_clustering:
            lines.append(f"  volatility  persistence = {self.vol_persistence:.3f}")
        lines.append(f"  innovation sigma = {self.sigma:.4g}")
        lines.append(f"  forecaster: {self.forecaster_name}")
        return "\n".join(lines)

    def forecast(self, h: int, *, return_conf_int: bool = False,
                 alpha: float = 0.05, dist: str = "normal", df: float = 7.0):
        """Forecast ``h`` steps with the backtest-selected forecaster named in
        :attr:`forecaster_name`. ``h`` counts samples, index steps of the
        fitted series, whatever the units of the time axis ``t`` the model was
        fit with.

        ``return_conf_int`` adds ``(lower, upper)`` bands whose growth matches
        the chosen forecaster: bounded for mean reversion, ``h^(2H)`` for long
        memory, ``~sqrt(h)`` for a random walk or drift, and roughly constant
        for a trend-stationary trend/seasonal forecast. ``dist="t"`` widens the
        band with a Student-t (``df``) quantile in place of the Gaussian one,
        for fat-tailed innovations.

        A model fit on a pandas ``Series`` returns the point forecast, and both
        bands, as pandas ``Series`` labelled by the length-``h`` future index
        continuing the training index: a ``DatetimeIndex`` at its frequency, an
        integer-like index by its step. An ndarray-fit model returns an
        ndarray, or a tuple of them."""
        steps = np.arange(1, h + 1, dtype=float)
        point = np.asarray(self._forecaster(h), dtype=float)
        # Future index continuing the training index for a pandas fit. ``None``
        # yields a plain ndarray, keeping ndarray-fit models and a pandas-free
        # environment bit-identical.
        fidx = extend_index(self._index, h) if self._index is not None else None
        if not return_conf_int:
            return as_series(point, fidx)
        from scipy.stats import norm, t as student_t
        # Band growth is keyed off the selected forecaster rather than the
        # detected flags. A trend+seasonal forecast with a stationary residual
        # must not fan out merely because a long-memory component was flagged
        # as well.
        name = self.forecaster_name
        if name.startswith("mean-reversion") and 0.0 < self.ar1_phi < 1.0:
            var = (self.sigma ** 2) * (1.0 - self.ar1_phi ** (2 * steps)) \
                / (1.0 - self.ar1_phi ** 2)
        elif name.startswith(("random walk", "drift")):
            if self.has_long_memory and 0.5 < self.hurst < 1.0:
                # Long memory forecasts as a random walk, but its h-step
                # forecast-error variance grows as h^(2H) with 2H > 1. The
                # plain sigma^2 * h random-walk band under-covers that, so the
                # band takes the h^(2H) growth directly.
                var = (self.sigma_walk ** 2) * steps ** (2.0 * self.hurst)
            else:
                var = (self.sigma_walk ** 2) * steps
        else:
            # Deterministic-mean forecasters: trend, seasonal, trend+seasonal
            # and the anchored variants. The residual around the fitted mean is
            # trend-stationary, putting its h-step forecast-error variance at a
            # near-constant sigma^2, not the random-walk sigma^2 * h that fans
            # the bands out and massively over-covers. Parameter-estimation
            # leverage adds a slow extra growth on top; computing it needs the
            # fit covariance and is left out here.
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

        This is the generative side of :class:`StochasticModel`. It composes
        the detected deterministic mean, trend plus multi-harmonic seasonal,
        with a stochastic residual drawn to match the detected second-order
        regime:

        * unit-root: an integrated random walk, drift plus innovations, with
          GARCH volatility clustering on the increments where detected;
        * mean-reverting: a stationary AR(1) residual at the recovered phi;
        * long-memory: an ARFIMA(0, d, 0) residual at ``d = H - 1/2``;
        * vol-clustering: a GARCH(1,1) residual at the recovered persistence;
        * white noise: i.i.d. innovations.

        Re-fitting the simulated path recovers the same regime and parameters.
        That round-trip is the test that the model generates the process it
        claims to characterize.

        Args:
            n: Length of the realization; defaults to the fitted ``n``.
            seed / rng: Randomness control, ``rng`` taking precedence.
            dist / df: Innovation distribution. ``"normal"`` is the default;
                ``"t"`` draws a unit-variance Student-t on ``df`` degrees of
                freedom, a fat-tailed generator for financial-style tail risk.
        """
        n = int(self.n if n is None else n)
        if rng is None:
            rng = np.random.default_rng(seed)
        noise = make_innovations(dist, df)
        sig = (self.sigma if np.isfinite(self.sigma) and self.sigma > 0.0
               else (self.sigma_walk if np.isfinite(self.sigma_walk)
                     and self.sigma_walk > 0.0 else 1.0))
        # Unit-root level: integrate drift plus innovations. The walk is its
        # own mean, so _mean_fn is not consulted.
        if "unit-root" in self.components:
            if self.has_vol_clustering and np.isfinite(self.vol_persistence):
                steps = _sim_garch(n, self.vol_persistence, sig, rng, noise=noise)
            else:
                steps = noise(rng, n) * sig
            drift = self.trend_slope if np.isfinite(self.trend_slope) else 0.0
            return self.level + np.cumsum(drift + steps)
        # Stationary or trend-stationary: deterministic mean plus a regime
        # residual.
        t = np.arange(n, dtype=float)
        mean = np.asarray(self._mean_fn(t), dtype=float)
        if mean.shape != t.shape:
            mean = np.full(n, float(self.level))
        # Residual regimes are tried in the dominance order the regime label
        # uses. Long memory subsumes a short-range AR(1) and comes first: a
        # long-memory series trips the mean-reversion flag too, and drawing it
        # as a bare AR(1) would erase the long memory and break the round-trip.
        if self.has_long_memory and np.isfinite(self.hurst):
            resid = _sim_long_memory(n, self.hurst, sig, rng, noise=noise)
        elif self.has_mean_reversion and 0.0 < self.ar1_phi < 1.0:
            resid = _sim_ar1(n, self.ar1_phi, sig, rng, noise=noise)
        elif self.has_vol_clustering and np.isfinite(self.vol_persistence):
            resid = _sim_garch(n, self.vol_persistence, sig, rng, noise=noise)
        else:
            resid = noise(rng, n) * sig
        return mean + resid


def _ar_whiten(e: np.ndarray, *, max_order: int = 5) -> np.ndarray:
    """Innovations of an AR(p) fit to ``e``, the order chosen by AIC, or the
    de-meaned series when ``p == 0``. This separates a genuine higher-order AR
    from long memory: an AR(p) is white after AR(p) whitening, long memory is
    not."""
    p = ar_order(e, max_order=max_order)
    ec = np.asarray(e, dtype=float) - float(np.mean(e))
    if p < 1 or ec.size <= p:
        return ec
    phi = np.asarray(fit_ar(e, order=p)["phi"], dtype=float)
    inn = ec[p:].copy()
    for j in range(1, p + 1):
        inn = inn - phi[j - 1] * ec[p - j:ec.size - j]
    return inn


def fit_stochastic(
    y: np.ndarray,
    t: np.ndarray | None = None,
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
) -> StochasticModel:
    """Characterize an arbitrary series across every route at once and return
    one coherent model.

    The routes run in the order the second-order theory dictates, each behind
    a significance gate:

    1. deterministic mean: an LSI trend, kept only if its slope is significant
       (``|t| > trend_t``), and an LSI cycle, kept only if a genuine interior
       spectral peak carries more than ``cycle_strength`` of the power and
       repeats at least ``min_cycles`` times;
    2. whiten: an AR(1) is fit to the residual and its innovations carry into
       the volatility test below;
    3. long memory: the spectral Hurst of the residual, declared when
       ``H > lm_hurst`` and a low-order AR fails to whiten it away. That veto
       is what keeps a near-unit-root AR(1) from being mislabelled;
    4. mean reversion: the AR(1) coefficient, kept when
       ``mr_phi < phi < 0.99`` and the lag-1 ACF is significant;
    5. volatility clustering: the persistence of the ``|residual|`` ACF, kept
       when significant and above ``vol_persist``.

    A series that opens none of these gates comes back as
    ``regime="white noise / random walk"`` with no components.

    Forecasting is by rolling-origin backtest over a regime-informed candidate
    set: random walk, drift, mean reversion, local-slope trend, multi-harmonic
    seasonal continuation. The winner is recorded in
    :attr:`StochasticModel.forecaster_name`, and the random walk takes it when
    nothing beats it. A series too short to backtest (``n <= 50``) falls back
    to the random walk with a :class:`UserWarning`, its recorded name suffixed
    ``" (short-series fallback)"``.

    All periods and horizons are in samples, the index steps of ``y``. A
    custom time axis ``t`` in seconds, years or any spacing is converted by
    the median spacing of ``t`` before the seasonal model is fit, so the
    fitted frequency stays correct on any axis.

    Args:
        y: The series. ``t`` defaults to ``0..n-1`` at uniform spacing. A
            custom ``t`` only rescales the time axis; detection gates and
            forecasts are unaffected by its units.
        period: A seasonal period to use, in samples rather than ``t`` units.
            Detected from the spectrum when omitted.
        max_harmonics: Cap on the Fourier harmonics of the seasonal component.
            The count actually used is chosen by BIC.
        forecaster: How to forecast. ``"auto"`` (the default) backtest-selects
            the RMSE-optimal candidate; a built-in name (one of
            :data:`FORECASTERS`) forces that model; a callable
            ``(train, h) -> array`` is used directly; a list of names,
            callables or ``(name, fn)`` pairs is a custom candidate set to
            backtest-select among.
        trend_t, cycle_strength, min_cycles, lm_hurst, mr_phi, vol_persist:
            Detection gates (see above).

    Returns:
        A :class:`StochasticModel` with the detected components, parameters
        and a backtest-selected :meth:`~StochasticModel.forecast`.
    """
    # Remember the training index a pandas Series/DataFrame carries, so the
    # forecast can be labelled with the continuing future index, then coerce
    # to a 1-D float ndarray. ``to_1d_array`` is bit-identical to
    # ``np.asarray(..., float)`` for a 1-D ndarray or list.
    index = capture_index(y)
    y = to_1d_array(y, "y")
    n = y.size
    t = np.arange(n, dtype=float) if t is None else to_1d_array(t, "t")
    # Median sample spacing of the possibly custom time axis. Periods are
    # detected and supplied in sample units while the seasonal model is fit
    # over ``t``, so they need converting to t units (samples * dt) first. On
    # the default axis dt = 1 and the conversion is the identity.
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1.0
    level = float(y.mean())
    sigma_walk = float(np.std(np.diff(y))) if n > 1 else 0.0
    band = 2.0 / np.sqrt(max(n, 1))

    # Stage 0: the unit-root gate. An I(1) level must be differenced, since
    # fitting a trend, cycle or long memory to its wandering level is the
    # classic spurious regression. The increments are characterized instead,
    # and for a financial series that is where the structure lives anyway.
    #
    # One exemption. A strongly cyclical series also has AR roots near the
    # unit circle, at the cycle frequency rather than at f=0, and ADF
    # mis-flags it. A genuine interior spectral peak means the persistence is
    # a cycle, so the series is kept for the stationary branch that can detect
    # it. The exemption is stricter than the general cycle gate on purpose: a
    # true cycle repeats many times with its peak well away from f=0, whereas
    # a random walk's one-off low-frequency wander also shows a peak.
    # Requiring 5 repetitions and a clear peak leaves FX and random-walk
    # levels differenced.
    if period is not None:
        pre_cyclical = True   # the caller declared a seasonal period
    else:
        pre_period, pre_strength = _cycle_strength(y) if n >= 8 else (float("nan"), 0.0)
        pre_cyclical = (pre_strength > 0.12 and np.isfinite(pre_period)
                        and 4 <= pre_period <= n / 5.0)
    if _is_nonstationary(y) and not pre_cyclical:
        w = np.diff(y)
        drift = float(w.mean())
        drift_sig = abs(drift) > 2.0 * (np.std(w) / np.sqrt(max(w.size, 1)))
        vol = float("nan")
        has_vol = False
        try:
            aw = np.abs(w - w.mean())
            if w.size > 2 and float(sample_acf(aw, 1)[1]) > band:
                vol = float(garch_persistence(w, use="abs")["persistence"])
                has_vol = vol > vol_persist
        except Exception as exc:
            warnings.warn(
                f"stochastic stage unit-root vol-clustering failed: {exc}",
                UserWarning, stacklevel=2)
        comps = (["unit-root"] + (["drift"] if drift_sig else [])
                 + (["vol-clustering"] if has_vol else []))
        regime = "random walk + drift" if drift_sig else "random walk"
        # An I(1) level is forecast by persistence or drift, never by
        # reversion to a sample mean: that mean is meaningless for a
        # non-stationary level and would pull an interest rate back to an
        # outdated long-run average. ``forecaster=`` still overrides.
        ur_per = float(period) if period is not None else _cycle_strength(w)[0]
        fname, ffn = _resolve_forecaster(
            forecaster,
            [("random walk", _fc_rw), ("drift", _fc_drift)],
            per=ur_per, max_harmonics=max_harmonics, y=y)
        return StochasticModel(
            n=n, level=level, trend_slope=drift if drift_sig else 0.0,
            has_trend=False,
            cycle_period=float("nan"), cycle_amp=float("nan"), has_cycle=False,
            n_harmonics=0, seasonal=False,
            hurst=float("nan"), has_long_memory=False,
            ar1_phi=float("nan"), has_mean_reversion=False,
            vol_persistence=vol, has_vol_clustering=has_vol,
            sigma=float(np.std(w)), sigma_walk=sigma_walk,
            components=tuple(comps), regime=regime, forecaster_name=fname,
            _forecaster=_bind_forecaster(ffn, y), _index=index,
        )

    # Stage 1: deterministic mean, the series being stationary or
    # trend-stationary.
    slope, intercept, tstat = _ols_line(t, y)
    resid_lin = y - (intercept + slope * t)
    sst = float((y - y.mean()) @ (y - y.mean())) + 1e-12
    trend_r2 = 1.0 - float(resid_lin @ resid_lin) / sst
    # The trend must be significant and explain real variance. A persistent
    # but stationary AR(1) can show a spuriously significant OLS slope that
    # explains almost none; the r2 gate rejects it.
    has_trend = (tstat > trend_t) and (trend_r2 > 0.10)
    if has_trend:
        rt = fit_lsi(t, y, "a0 + a1*x", "x", k_star=1)
        a0, a1 = float(rt.coeffs[0]), float(rt.coeffs[1])
        trend = a0 + a1 * t
        slope = a1
        mean_a0, mean_a1 = a0, a1
    else:
        trend = np.full(n, level)
        slope = 0.0
        mean_a0, mean_a1 = level, 0.0
    d1 = y - trend

    # Stage 1b: the seasonal or cyclical component, a multi-harmonic Fourier
    # model at the detected or caller-given period, fit by linear least
    # squares. The harmonics reach seasonal shapes a single sinusoid cannot,
    # the CO2 sawtooth and the sunspot pulse among them. That both reports the
    # cycle right and forecasts it far better.
    if period is not None:
        per, strength = float(period), 1.0
        has_cycle = 4 <= per <= n / 2.0
    else:
        per, strength = _cycle_strength(d1)
        has_cycle = (strength > cycle_strength and np.isfinite(per)
                     and 4 <= per <= n / min_cycles)
    cyc_amp = float("nan")
    n_harm = 0
    seas_r2 = 0.0
    # ``per`` is in sample units (FFT bins, the documented ``period=`` unit)
    # while the seasonal model is fit over the supplied ``t``. Convert first;
    # any non-unit spacing would otherwise fit the wrong frequency.
    per_t = per * dt
    if has_cycle:
        n_harm, s_coef = _fit_seasonal(t, d1, per_t, max_harmonics)
        x_seas = _seasonal_design(t, per_t, n_harm)
        cyc = x_seas @ s_coef
        cyc_amp = float(np.hypot(s_coef[0], s_coef[1]))   # fundamental amplitude
        seas_r2 = 1.0 - float(np.var(d1 - cyc)) / (float(np.var(d1)) + 1e-12)
    else:
        cyc = np.zeros(n)
        s_coef = None
    e = d1 - cyc  # stochastic residual
    # "seasonal" means a strong, clean repeating pattern; "cyclical" a weaker
    # stochastic one. A caller-given period counts as seasonal, as does a
    # component explaining a good share of the de-trended variance.
    is_seasonal = has_cycle and (period is not None or seas_r2 > 0.30)

    # Stages 2 to 4: whiten with AR(1), then test for long memory.
    try:
        phi = float(ar1_reversion(e)["phi"])
    except Exception as exc:
        warnings.warn(f"stochastic stage AR(1) failed: {exc}",
                      UserWarning, stacklevel=2)
        phi = 0.0
    acf1 = float(sample_acf(e, 1)[1]) if n > 2 else 0.0
    has_mr = (mr_phi < phi < 0.99) and (abs(acf1) > band)
    innov = e[1:] - phi * e[:-1] if has_mr else (e - e.mean())
    sigma = float(np.std(innov)) if innov.size else float(np.std(e))

    # Long memory: the Hurst is read off the raw residual, the only place the
    # true long-range strength survives, since AR(1) whitening removes part of
    # it. A finite-order-AR veto then gates that reading. The veto whitens with
    # a low-order AR, capped at 3: a genuine finite-order AR(1..3), a
    # near-unit-root AR(1) included, comes back white at Hurst ~0.5, while a
    # hyperbolic-ACF ARFIMA survives because no finite AR captures a
    # hyperbolic ACF. That separates a mean-reverting autoregression from true
    # long memory at any AR order.
    hurst = float("nan")
    has_lm = False
    try:
        hurst = float(hurst_spectral(e - e.mean())["H"])
        has_lm = hurst > lm_hurst
        if has_lm and ar_order(e, max_order=3) >= 1:
            inn_p = _ar_whiten(e, max_order=3)
            veto_h = 0.5 + 0.5 * (lm_hurst - 0.5)  # midpoint of 0.5 and the gate
            if inn_p.size > 128 and float(hurst_spectral(inn_p)["H"]) <= veto_h:
                has_lm = False
                has_mr = True  # a stationary finite-order AR reverts to its mean
    except Exception as exc:
        warnings.warn(f"stochastic stage Hurst/long-memory failed: {exc}",
                      UserWarning, stacklevel=2)

    # Stage 5: volatility clustering, tested on the whitened residual. A
    # persistent AR(1) level has trivially autocorrelated absolute values, so
    # the raw residual returns a false positive. Genuine ARCH-type clustering
    # survives whitening.
    vol = float("nan")
    has_vol = False
    try:
        aw = np.abs(innov - innov.mean())
        if innov.size > 2 and float(sample_acf(aw, 1)[1]) > band:
            vol = float(garch_persistence(innov, use="abs")["persistence"])
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

    # The candidate set is scoped to the detected structure so the chosen
    # forecaster reflects it. A cyclical series weighs its seasonal
    # continuation against RW instead of being handed a mean-reverting level
    # model that would forecast a flat line and erase the cycle; a trending
    # series gets trend and drift; a stationary mean-reverting one gets mean
    # reversion, toward a mean it actually has. ``forecaster=`` overrides.
    # A detected deterministic structure gets the lenient selection margin so
    # it survives unless clearly worse than RW; level regimes stay strict. See
    # :func:`_select_forecaster` for what the margin does.
    candidates: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]]
    sel_margin = 0.98
    if has_cycle:
        seasonal_name = "trend+seasonal" if has_trend else "seasonal"
        # Two seasonal forecasters go in and the backtest picks per series:
        # the unbiased fitted extrapolation for a noisy seasonal series, the
        # anchored one for a clean trend+seasonal series like CO2.
        candidates = [("random walk", _fc_rw),
                      (seasonal_name,
                       _make_seasonal_fc(per, max_harmonics, has_trend)),
                      (seasonal_name + " (anchored)",
                       _make_seasonal_fc_anchored(per, max_harmonics, has_trend))]
        sel_margin = 1.15
    elif has_trend:
        # A pure-trend regime keeps the conservative linear trend candidate.
        # On a short record such as the Nile at n=100 the backtest mis-selects
        # a curved fit that over-extrapolates a spurious level-shift "trend".
        # Curvature-aware extrapolation belongs to the seasonal path, where a
        # trend+seasonal series like CO2 has a genuinely accelerating trend.
        candidates = [("random walk", _fc_rw), ("drift", _fc_drift),
                      ("trend", _fc_trend)]
        sel_margin = 1.15
    elif has_mr:
        candidates = [("random walk", _fc_rw), ("mean-reversion", _fc_meanrev)]
    else:                                   # long memory or white noise
        candidates = [("random walk", _fc_rw)]
    fname, ffn = _resolve_forecaster(forecaster, candidates, per=per,
                                     max_harmonics=max_harmonics, y=y,
                                     margin=sel_margin)

    # Deterministic mean (trend + multi-harmonic seasonal) over the sample
    # index, captured for StochasticModel.simulate to regenerate the
    # structured part. The coefficients were fit against the supplied ``t``,
    # so the index is mapped onto that axis as t[0] + dt * index before
    # evaluating; on the default axis that is the identity. The fitted
    # coefficients are bound as defaults to avoid late binding.
    def _mean_fn(tt, _a0=mean_a0, _a1=mean_a1, _hc=bool(has_cycle),
                 _p=float(per_t) if has_cycle else float("nan"),
                 _nh=int(n_harm), _co=s_coef,
                 _t0=float(t[0]) if n else 0.0, _dt=dt):
        tt = _t0 + _dt * np.asarray(tt, dtype=float)
        m = _a0 + _a1 * tt
        if _hc and _co is not None:
            m = m + _seasonal_design(tt, _p, _nh) @ _co
        return m

    return StochasticModel(
        n=n, level=level, trend_slope=float(slope), has_trend=bool(has_trend),
        cycle_period=float(per) if has_cycle else float("nan"),
        cycle_amp=float(cyc_amp), has_cycle=bool(has_cycle),
        n_harmonics=int(n_harm), seasonal=bool(is_seasonal),
        hurst=hurst, has_long_memory=bool(has_lm),
        ar1_phi=float(phi), has_mean_reversion=bool(has_mr),
        vol_persistence=vol, has_vol_clustering=bool(has_vol),
        sigma=float(sigma), sigma_walk=float(sigma_walk),
        components=tuple(comps) if comps else ("none",), regime=regime,
        forecaster_name=fname, _forecaster=_bind_forecaster(ffn, y),
        _mean_fn=_mean_fn, _index=index,
    )
