"""The classical twin of ``dtfit.stochastic``: the same unified stochastic
characterization, built entirely from textbook (non-dtfit) estimators.

``dtfit.stochastic.fit_stochastic`` recovers a process's second-order structure by fitting
the deterministic functionals of the series (its ACF, spectrum, aggregated
variance, trend/cycle) with dtfit's own LSI/EAC integral fitters. To answer
whether the dtfit route actually improves on what a practitioner already uses,
this module re-implements the same model, with the same regime gates, the same
parameters and the same forecast routing, but swaps every estimator for its
established classical counterpart:

================  ==========================  ===============================
quantity          dtfit route                 classical route (here)
================  ==========================  ===============================
unit root         vendored ADF                ``statsmodels`` ADF (`adfuller`)
trend slope       LSI line fit                OLS line
cycle period      damped-cosine ACF fit (LSI) FFT periodogram peak
AR(1) phi         LSI exp fit to the ACF       OLS regression ``x_t ~ x_{t-1}``
Hurst H           LSI power-law spectral fit   detrended fluctuation analysis
vol persistence   LSI exp fit to |resid| ACF   GARCH(1,1) Gaussian QMLE
vol-cluster gate  ACF significance            ``statsmodels`` ARCH-LM test
================  ==========================  ===============================

:func:`fit_classical_stochastic` returns a :class:`ClassicalStochasticModel`
carrying the same public surface as :class:`dtfit.stochastic.StochasticModel`
(``regime``, ``components``, the ``has_*`` flags, the recovered parameters and
a ``forecast``), so both drop into one comparison harness. Everything is pure
NumPy/SciPy plus ``statsmodels``, already a suite dependency; GARCH is fit by a
compact in-house QMLE, so no ``arch`` package is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .baselines import hurst_dfa

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import het_arch
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover - statsmodels is a suite dependency
    HAVE_STATSMODELS = False

__all__ = [
    "ols_ar1", "garch_mle_persistence", "classical_decompose",
    "periodogram_period", "ClassicalStochasticModel", "fit_classical_stochastic",
]


# the individual classical estimators, each the standard counterpart of one
# dtfit stochastic functional
def ols_ar1(x) -> float:
    """AR(1) coefficient by OLS regression ``x_t ~ x_{t-1}`` on the centred
    series: the textbook Yule-Walker / least-squares estimator that dtfit
    replaces with an LSI exponential fit to the ACF."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    x0, x1 = x[:-1], x[1:]
    denom = float(x0 @ x0)
    return float(x1 @ x0 / denom) if denom > 0 else 0.0


def periodogram_period(y, *, min_period: int = 4) -> tuple[float, float]:
    """``(period, strength)`` of the dominant FFT-periodogram peak of a
    linearly-detrended series: the standard spectral cycle estimator.
    ``strength`` is the peak's share of detrended power in ``[0, 1]``."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    t = np.arange(n, dtype=float)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    spec = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    total = float(spec[1:].sum())
    if total <= 0.0:
        return float("nan"), 0.0
    k = int(np.argmax(spec))
    strength = float(spec[k] / total)
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength


def classical_decompose(t, y, *, trend_deg: int = 1) -> dict[str, float]:
    """Textbook trend+cycle decomposition: an OLS polynomial trend plus the
    FFT-periodogram period of the detrended residual. The classical counterpart
    of dtfit's ``decompose_trend_cycle`` (LSI trend, oscillatory cycle)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.polyfit(t, y, trend_deg)
    slope = float(c[trend_deg - 1]) if trend_deg >= 1 else 0.0
    resid = y - np.polyval(c, t)
    period, _ = periodogram_period(resid)
    return {"slope": slope, "period": period}


def garch_mle_persistence(r, *, maxlen: int = 3000) -> float:
    """GARCH(1,1) persistence ``alpha + beta`` by Gaussian quasi-MLE: the
    standard volatility-model estimator, here in a dependency-free NumPy/SciPy
    form that runs the conditional-variance recursion as a one-pole IIR filter,
    so each likelihood evaluation is O(n). The classical counterpart of dtfit's
    LSI exponential fit to the ACF of ``|returns|``.
    """
    from scipy.optimize import minimize
    from scipy.signal import lfilter

    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    if r.size > maxlen:
        r = r[-maxlen:]
    n = r.size
    var = float(np.var(r)) + 1e-12

    def negll(theta: np.ndarray) -> float:
        omega, alpha, beta = theta
        if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or alpha + beta >= 0.9999:
            return 1e10
        u = np.empty(n)
        u[0] = var
        u[1:] = omega + alpha * r[:-1] ** 2
        # s2[t] = beta*s2[t-1] + u[t], s2[-1]=0 -> s2[0]=u[0]=var: a one-pole IIR.
        s2 = np.clip(lfilter([1.0], [1.0, -beta], u), 1e-12, None)
        return 0.5 * float(np.sum(np.log(s2) + r ** 2 / s2))

    sol = minimize(negll, np.array([var * 0.1, 0.08, 0.9]),
                   method="Nelder-Mead",
                   options={"maxiter": 1500, "fatol": 1e-6, "xatol": 1e-5})
    alpha, beta = float(sol.x[1]), float(sol.x[2])
    return alpha + beta


def _is_unit_root(y: np.ndarray, *, alpha: float = 0.05) -> bool:
    """ADF unit-root decision (``statsmodels`` if available, else an AR(1)-near-1
    fallback)."""
    if HAVE_STATSMODELS:
        try:
            return bool(adfuller(y, autolag="AIC")[1] > alpha)
        except Exception:
            pass
    return ols_ar1(y) > 0.98


def _has_arch(resid: np.ndarray, *, alpha: float = 0.05) -> bool:
    """ARCH-LM test for conditional heteroskedasticity (``statsmodels`` if
    available, else a lag-1 ACF-of-squares fallback)."""
    if HAVE_STATSMODELS:
        try:
            return bool(het_arch(resid, nlags=10)[1] < alpha)
        except Exception:
            pass
    a = resid ** 2
    a = a - a.mean()
    denom = float(a[:-1] @ a[:-1])
    ac1 = float(a[1:] @ a[:-1] / denom) if denom > 0 else 0.0
    return ac1 > 2.0 / np.sqrt(max(resid.size, 1))


# the regime-appropriate forecasters, on dtfit's routing but built from OLS and
# persistence instead of LSI
def _fc_rw(tr: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(tr[-1]))


def _fc_mean(tr: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(np.mean(tr)))


def _fc_drift(tr: np.ndarray, h: int) -> np.ndarray:
    slope = (tr[-1] - tr[0]) / max(tr.size - 1, 1)
    return tr[-1] + slope * np.arange(1, h + 1)


def _fc_meanrev(tr: np.ndarray, h: int) -> np.ndarray:
    m = float(np.mean(tr))
    phi = ols_ar1(tr)
    return m + (tr[-1] - m) * phi ** np.arange(1, h + 1, dtype=float)


def _fc_trend(tr: np.ndarray, h: int) -> np.ndarray:
    t = np.arange(tr.size, dtype=float)
    c = np.polyfit(t, tr, 1)
    return np.polyval(c, np.arange(tr.size, tr.size + h, dtype=float))


def _seasonal_design(t: np.ndarray, period: float, n_harm: int,
                     with_trend: bool) -> np.ndarray:
    cols = [np.ones_like(t)]
    if with_trend:
        cols.append(t)
    for k in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def _make_seasonal_fc(period: float, with_trend: bool, n_harm: int = 3):
    def fc(tr: np.ndarray, h: int) -> np.ndarray:
        t = np.arange(tr.size, dtype=float)
        design = _seasonal_design(t, period, n_harm, with_trend)
        coef, *_ = np.linalg.lstsq(design, tr, rcond=None)
        tf = np.arange(tr.size, tr.size + h, dtype=float)
        return _seasonal_design(tf, period, n_harm, with_trend) @ coef
    return fc


@dataclass
class ClassicalStochasticModel:
    """The classical-estimator counterpart of
    :class:`dtfit.stochastic.StochasticModel`, with the same public surface
    (regime, components, ``has_*`` flags, parameters and :meth:`forecast`).
    Produced by :func:`fit_classical_stochastic`."""

    n: int
    level: float
    trend_slope: float
    has_trend: bool
    cycle_period: float
    has_cycle: bool
    seasonal: bool
    hurst: float
    has_long_memory: bool
    ar1_phi: float
    has_mean_reversion: bool
    vol_persistence: float
    has_vol_clustering: bool
    sigma: float
    sigma_walk: float
    components: tuple[str, ...]
    regime: str
    forecaster_name: str
    _forecaster: Callable[[int], np.ndarray] = field(
        repr=False, default=lambda h: np.zeros(h))

    def forecast(self, h: int) -> np.ndarray:
        return np.asarray(self._forecaster(h), dtype=float)

    def fingerprint(self) -> dict[str, object]:
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


def fit_classical_stochastic(
    y: np.ndarray,
    t: np.ndarray | None = None,
    *,
    period: float | None = None,
    trend_t: float = 3.0,
    cycle_strength: float = 0.08,
    min_cycles: float = 2.5,
    lm_hurst: float = 0.68,
    mr_phi: float = 0.15,
    vol_persist: float = 0.60,
) -> ClassicalStochasticModel:
    """Characterize ``y`` on the gated routing of :func:`dtfit.stochastic.fit_stochastic`
    but with the classical estimators throughout: an ADF unit-root test, an OLS
    trend and AR(1), an FFT-periodogram cycle, a DFA Hurst and a GARCH-QMLE
    volatility persistence. Returns a :class:`ClassicalStochasticModel`
    mirroring dtfit's, so the two can be scored head-to-head. The detection
    gates match ``fit_stochastic``'s exactly; the estimator is then the only
    difference the comparison measures."""
    y = np.asarray(y, dtype=float)
    n = y.size
    t = np.arange(n, dtype=float) if t is None else np.asarray(t, dtype=float)
    level = float(y.mean())
    sigma_walk = float(np.std(np.diff(y))) if n > 1 else 0.0
    band = 2.0 / np.sqrt(max(n, 1))

    # Stage 0: unit-root gate (with the same strong-cycle exemption).
    if period is not None:
        pre_cyclical = True
    else:
        pre_p, pre_s = periodogram_period(y) if n >= 8 else (float("nan"), 0.0)
        pre_cyclical = (pre_s > 0.12 and np.isfinite(pre_p)
                        and 4 <= pre_p <= n / 5.0)
    if _is_unit_root(y) and not pre_cyclical:
        w = np.diff(y)
        drift = float(w.mean())
        drift_sig = abs(drift) > 2.0 * (np.std(w) / np.sqrt(max(w.size, 1)))
        vol = float("nan")
        has_vol = False
        try:
            if w.size > 20 and _has_arch(w - w.mean()):
                vol = garch_mle_persistence(w)
                has_vol = vol > vol_persist
        except Exception:
            pass
        comps = (["unit-root"] + (["drift"] if drift_sig else [])
                 + (["vol-clustering"] if has_vol else []))
        regime = "random walk + drift" if drift_sig else "random walk"
        fc = _fc_drift if drift_sig else _fc_rw
        fname = "drift" if drift_sig else "random walk"
        return ClassicalStochasticModel(
            n=n, level=level, trend_slope=drift if drift_sig else 0.0,
            has_trend=False, cycle_period=float("nan"), has_cycle=False,
            seasonal=False, hurst=float("nan"), has_long_memory=False,
            ar1_phi=float("nan"), has_mean_reversion=False,
            vol_persistence=vol, has_vol_clustering=has_vol,
            sigma=float(np.std(w)), sigma_walk=sigma_walk,
            components=tuple(comps), regime=regime, forecaster_name=fname,
            _forecaster=lambda h: fc(y, h))

    # Stage 1: deterministic mean (OLS trend).
    slope, intercept = np.polyfit(t, y, 1)
    resid_lin = y - (intercept + slope * t)
    sst = float((y - y.mean()) @ (y - y.mean())) + 1e-12
    trend_r2 = 1.0 - float(resid_lin @ resid_lin) / sst
    # OLS slope t-stat
    se = np.sqrt(float(resid_lin @ resid_lin) / max(n - 2, 1)
                 / (float((t - t.mean()) @ (t - t.mean())) + 1e-12))
    tstat = abs(float(slope)) / (se + 1e-12)
    has_trend = (tstat > trend_t) and (trend_r2 > 0.10)
    if has_trend:
        trend = intercept + slope * t
    else:
        trend = np.full(n, level)
        slope = 0.0
    d1 = y - trend

    # 1b. cycle (periodogram peak).
    if period is not None:
        per, has_cycle = float(period), 4 <= float(period) <= n / 2.0
        seas_r2 = 1.0
    else:
        per, strength = periodogram_period(d1)
        has_cycle = (strength > cycle_strength and np.isfinite(per)
                     and 4 <= per <= n / min_cycles)
        seas_r2 = 0.0
    if has_cycle:
        design = _seasonal_design(t, per, 3, False)
        coef, *_ = np.linalg.lstsq(design, d1, rcond=None)
        cyc = design @ coef
        seas_r2 = 1.0 - float(np.var(d1 - cyc)) / (float(np.var(d1)) + 1e-12)
    else:
        cyc = np.zeros(n)
    e = d1 - cyc
    is_seasonal = has_cycle and (period is not None or seas_r2 > 0.30)

    # 2-4. AR(1) (OLS) + long memory (DFA) on the innovations.
    phi = ols_ar1(e)
    acf1_denom = float((e - e.mean())[:-1] @ (e - e.mean())[:-1])
    acf1 = (float((e - e.mean())[1:] @ (e - e.mean())[:-1] / acf1_denom)
            if acf1_denom > 0 else 0.0)
    has_mr = (mr_phi < phi < 0.99) and (abs(acf1) > band)
    innov = e[1:] - phi * e[:-1] if has_mr else (e - e.mean())
    sigma = float(np.std(innov)) if innov.size else float(np.std(e))

    hurst = float("nan")
    has_lm = False
    try:
        target = innov if innov.size > 128 else e
        hurst = float(hurst_dfa(target))
        has_lm = np.isfinite(hurst) and hurst > lm_hurst
    except Exception:
        pass

    # 5. volatility clustering (ARCH-LM gate + GARCH QMLE) on the innovations.
    vol = float("nan")
    has_vol = False
    try:
        if innov.size > 20 and _has_arch(innov - innov.mean()):
            vol = garch_mle_persistence(innov)
            has_vol = np.isfinite(vol) and vol > vol_persist
    except Exception:
        pass

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

    # regime-appropriate classical forecaster, on dtfit's routing
    if has_cycle:
        fname = ("trend+seasonal" if has_trend else "seasonal")
        fc = _make_seasonal_fc(per, has_trend)
    elif has_trend:
        fname, fc = "trend", _fc_trend
    elif has_mr:
        fname, fc = "mean-reversion", _fc_meanrev
    elif has_lm:
        fname, fc = "random walk", _fc_rw
    else:
        fname, fc = "mean", _fc_mean

    return ClassicalStochasticModel(
        n=n, level=level, trend_slope=float(slope), has_trend=bool(has_trend),
        cycle_period=float(per) if has_cycle else float("nan"),
        has_cycle=bool(has_cycle), seasonal=bool(is_seasonal),
        hurst=hurst, has_long_memory=bool(has_lm),
        ar1_phi=float(phi), has_mean_reversion=bool(has_mr),
        vol_persistence=vol, has_vol_clustering=bool(has_vol),
        sigma=sigma, sigma_walk=sigma_walk,
        components=tuple(comps), regime=regime, forecaster_name=fname,
        _forecaster=lambda h: fc(y, h))
