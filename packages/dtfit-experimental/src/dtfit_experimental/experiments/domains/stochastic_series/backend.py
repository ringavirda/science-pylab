"""Backend for the stochastic-series domain experiment.

The question it exists to answer: can dtfit, a deterministic curve fitter, be
put to work on genuinely random series such as economic and financial data?
dtfit cannot fit a martingale path, but it can fit a stochastic process's
deterministic functionals, its autocovariance, spectrum, aggregated variance
and trend/cycle, and recover the process's parameters from those. The
estimators live in :mod:`dtfit.stochastic`; this module supplies the
ground-truth data generators and the evaluation harness that measures, for each
possibility, whether the dtfit route really does recover the known parameter
and how it compares against the standard estimator.

Each experiment simulates a process with a known parameter, runs the dtfit
estimator and a plain baseline over several seeds, and reports the mean
recovery error with a categorical verdict of VIABLE, MARGINAL or NOT VIABLE:

* E1, long memory by aggregated variance: ARFIMA(0,d,0), recovering the Hurst
  exponent ``H = d + 1/2`` from the power-law aggregated-variance curve.
* E2, long memory by spectrum: the same series, recovering ``H`` from the
  low-frequency log-periodogram slope, the GPH route, through LSI.
* E3, mean reversion: AR(1)/OU, recovering the AR(1) coefficient from an
  exponential fit to the ACF.
* E4, volatility persistence: GARCH(1,1), recovering ``alpha + beta`` from an
  exponential fit to the ACF of squared returns.
* E5, stochastic cycle: AR(2) with complex roots, recovering the cycle period
  from a damped-cosine fit to the ACF.
* E6, trend and cycle decomposition: a structural series plus noise, recovering
  the trend slope and cycle period and leaving a stochastic residual.

Later experiments in the same module go past parameter recovery: the merged
router (E7), forecast skill and the head-to-head against the classical
estimators (E8, E8b), real data (E9, E10), the streaming filter (E11), the
generative round-trip (E12) and the AR-versus-long-memory guard (E13, E14).

``run()`` returns the rows; ``summary()`` renders the markdown verdict table.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dtfit.stochastic import (
    hurst_aggvar,
    hurst_spectral,
    ar1_reversion,
    garch_persistence,
    cycle_period,
    decompose_trend_cycle,
    fit_stochastic,
    StochasticFilter,
    ar_order,
    fit_ar,
    fractional_difference,
)
from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl
from dtfit_experimental.experiments.common.classical_stochastic import (
    fit_classical_stochastic, garch_mle_persistence, classical_decompose,
)

__all__ = [
    "gen_arfima", "gen_ar1", "gen_garch", "gen_ar2_cycle", "gen_trend_cycle",
    "exp_hurst_aggvar", "exp_hurst_spectral", "exp_ar1", "exp_garch",
    "exp_cycle", "exp_decompose",
    "exp_merged_router", "forecast_skill", "exp_forecast_skill",
    "exp_model_comparison",
    "load_series", "exp_real_data",
    "REAL_DATASETS", "hurst_comparison", "exp_real_suite", "panel_forecasts",
    "exp_online_filter", "filter_trace",
    "exp_simulate_roundtrip", "simulate_example",
    "filter_break_demo", "filter_characteristics",
    "exp_ar_discrimination", "exp_ar_order_recovery",
    "exp_fracdiff_whitening", "exp_student_t",
    "run", "summary",
]


# the ground-truth data generators
def gen_arfima(n: int, d: float, rng: np.random.Generator,
               *, ntrunc: int = 1200) -> np.ndarray:
    """ARFIMA(0, d, 0): white noise fractionally integrated to order ``d``.

    Built from the truncated MA(inf) expansion of ``(1 - B)^(-d)``
    (``psi_0 = 1``, ``psi_j = psi_{j-1} (j-1+d)/j``). Exhibits long memory with
    Hurst exponent ``H = d + 1/2``.
    """
    psi = np.empty(ntrunc)
    psi[0] = 1.0
    for j in range(1, ntrunc):
        psi[j] = psi[j - 1] * (j - 1 + d) / j
    e = rng.standard_normal(n + ntrunc)
    x = np.convolve(e, psi)[ntrunc: ntrunc + n]
    return np.asarray(x, dtype=float)


def gen_ar1(n: int, phi: float, rng: np.random.Generator,
            *, sigma: float = 1.0, burn: int = 200) -> np.ndarray:
    """AR(1) / discretely-sampled Ornstein-Uhlenbeck: ``x_t = phi x_{t-1} + e``."""
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def gen_garch(n: int, omega: float, alpha: float, beta: float,
              rng: np.random.Generator, *, burn: int = 500) -> np.ndarray:
    """GARCH(1,1) returns; volatility persistence is ``alpha + beta``."""
    N = n + burn
    z = rng.standard_normal(N)
    s2 = np.empty(N)
    r = np.empty(N)
    s2[0] = omega / max(1e-9, 1.0 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, N):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:]


def gen_ar2_cycle(n: int, period: float, damping: float,
                  rng: np.random.Generator, *, burn: int = 300) -> np.ndarray:
    """AR(2) with complex roots: a stochastic pseudo-cycle of the given period
    and damping (root modulus). ``phi1 = 2 r cos(2pi/period)``, ``phi2 = -r^2``."""
    phi1 = 2.0 * damping * np.cos(2.0 * np.pi / period)
    phi2 = -(damping ** 2)
    N = n + burn
    e = rng.standard_normal(N)
    x = np.zeros(N)
    for t in range(2, N):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x[burn:]


def gen_trend_cycle(n: int, slope: float, period: float, amp: float,
                    noise_sd: float, rng: np.random.Generator,
                    *, intercept: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic trend + cycle plus i.i.d. noise -> ``(t, y)``."""
    t = np.arange(n, dtype=float)
    y = (intercept + slope * t + amp * np.sin(2.0 * np.pi * t / period)
         + rng.normal(0.0, noise_sd, n))
    return t, y


# verdict helper
def _verdict(err: float, good: float, ok: float) -> str:
    if not np.isfinite(err):
        return "NOT VIABLE"
    if err <= good:
        return "VIABLE"
    if err <= ok:
        return "MARGINAL"
    return "NOT VIABLE"


def _rel(est: float, truth: float) -> float:
    return abs(est - truth) / abs(truth) * 100.0 if truth != 0 else float("nan")


def _mean(vals: list[float]) -> float:
    a = np.asarray(vals, dtype=float)
    return float(np.nanmean(a)) if a.size and np.any(np.isfinite(a)) else float("nan")


# E1, long memory via aggregated variance
def exp_hurst_aggvar(seeds: int = 8, *, n: int = 4096,
                     ds: tuple[float, ...] = (0.2, 0.3, 0.4)) -> dict:
    dt_e, base_e, rs_e, dfa_e = [], [], [], []
    for d in ds:
        H = d + 0.5
        for s in range(seeds):
            x = gen_arfima(n, d, np.random.default_rng(1000 + s))
            try:
                dt_e.append(abs(hurst_aggvar(x, method="lsi")["H"] - H))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(abs(hurst_aggvar(x, method="ols")["H"] - H))
            except Exception:
                base_e.append(np.nan)
            try:
                rs_e.append(abs(bl.hurst_rs(x) - H))
            except Exception:
                rs_e.append(np.nan)
            try:
                dfa_e.append(abs(bl.hurst_dfa(x) - H))
            except Exception:
                dfa_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E1", "name": "long memory (aggregated variance)",
        "param": "Hurst H", "metric": "MAE(H)",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 0.07, 0.15),
        "method": "LSI power-law fit to Var(block mean) vs scale",
        "extra": {"OLS log-log": _mean(base_e), "R/S": _mean(rs_e),
                  "DFA": _mean(dfa_e)},
    }


# E2, long memory via the low-frequency spectrum (GPH)
def exp_hurst_spectral(seeds: int = 8, *, n: int = 4096,
                       ds: tuple[float, ...] = (0.2, 0.3, 0.4)) -> dict:
    dt_e, base_e, rs_e, dfa_e = [], [], [], []
    for d in ds:
        H = d + 0.5
        for s in range(seeds):
            x = gen_arfima(n, d, np.random.default_rng(2000 + s))
            try:
                dt_e.append(abs(hurst_spectral(x, method="lsi")["H"] - H))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(abs(hurst_spectral(x, method="ols")["H"] - H))
            except Exception:
                base_e.append(np.nan)
            try:
                rs_e.append(abs(bl.hurst_rs(x) - H))
            except Exception:
                rs_e.append(np.nan)
            try:
                dfa_e.append(abs(bl.hurst_dfa(x) - H))
            except Exception:
                dfa_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E2", "name": "long memory (low-freq spectrum / GPH)",
        "param": "Hurst H", "metric": "MAE(H)",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 0.10, 0.20),
        "method": "LSI slope of log-periodogram (smoothed)",
        "extra": {"GPH/OLS": _mean(base_e), "R/S": _mean(rs_e),
                  "DFA": _mean(dfa_e)},
    }


# E3, mean reversion in an AR(1) / OU process
def exp_ar1(seeds: int = 8, *, n: int = 1500,
            phis: tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
    dt_e, base_e = [], []
    for phi in phis:
        for s in range(seeds):
            x = gen_ar1(n, phi, np.random.default_rng(3000 + s))
            try:
                dt_e.append(_rel(ar1_reversion(x, method="lsi")["phi"], phi))
            except Exception:
                dt_e.append(np.nan)
            try:
                base_e.append(_rel(ar1_reversion(x, method="acf1")["phi"], phi))
            except Exception:
                base_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E3", "name": "mean reversion (AR(1) / OU)",
        "param": "AR(1) phi", "metric": "rel.err %",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 8.0, 20.0),
        "method": "LSI exponential fit to the ACF",
    }


# E4, volatility persistence in a GARCH(1,1)
def exp_garch(seeds: int = 8, *, n: int = 4000,
              params: tuple[tuple[float, float, float], ...] = (
                  (0.05, 0.08, 0.90), (0.05, 0.10, 0.85), (0.02, 0.05, 0.93))
              ) -> dict:
    dt_e, base_e = [], []
    for (omega, alpha, beta) in params:
        persist = alpha + beta
        for s in range(seeds):
            r = gen_garch(n, omega, alpha, beta, np.random.default_rng(4000 + s))
            try:
                est = garch_persistence(r, method="lsi")["persistence"]
                dt_e.append(_rel(est, persist))
            except Exception:
                dt_e.append(np.nan)
            # the baseline: textbook GARCH(1,1) Gaussian QMLE persistence
            try:
                base_e.append(_rel(garch_mle_persistence(r), persist))
            except Exception:
                base_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E4", "name": "volatility persistence (GARCH(1,1))",
        "param": "alpha+beta", "metric": "rel.err %",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI exponential fit to ACF of squared returns",
        "extra": {"GARCH(1,1) QMLE": _mean(base_e)},
    }


# E5, the stochastic cycle of an AR(2) with complex roots
def exp_cycle(seeds: int = 8, *, n: int = 1200,
              periods: tuple[float, ...] = (10.0, 16.0, 25.0),
              damping: float = 0.97) -> dict:
    dt_e, base_e = [], []
    for P in periods:
        for s in range(seeds):
            x = gen_ar2_cycle(n, P, damping, np.random.default_rng(5000 + s))
            try:
                dt_e.append(_rel(cycle_period(x)["period"], P))
            except Exception:
                dt_e.append(np.nan)
            # the baseline: the FFT periodogram's peak period
            try:
                xx = x - x.mean()
                spec = np.abs(np.fft.rfft(xx)) ** 2
                freqs = np.fft.rfftfreq(xx.size, d=1.0)
                spec[0] = 0.0
                pk = freqs[int(np.argmax(spec))]
                base_e.append(_rel(1.0 / pk if pk > 0 else np.inf, P))
            except Exception:
                base_e.append(np.nan)
    err = _mean(dt_e)
    return {
        "id": "E5", "name": "stochastic cycle (AR(2) complex roots)",
        "param": "cycle period", "metric": "rel.err %",
        "dt_err": err, "base_err": _mean(base_e),
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI damped-cosine fit to the ACF",
    }


# E6, trend and cycle decomposition
def exp_decompose(seeds: int = 8, *, n: int = 600, slope: float = 0.02,
                  period: float = 50.0, amp: float = 3.0,
                  noise_sd: float = 1.0) -> dict:
    slope_e, period_e = [], []
    bslope_e, bperiod_e = [], []
    for s in range(seeds):
        t, y = gen_trend_cycle(n, slope, period, amp, noise_sd,
                               np.random.default_rng(6000 + s))
        try:
            dec = decompose_trend_cycle(t, y)
            slope_e.append(_rel(dec["slope"], slope))
            period_e.append(_rel(dec["period"], period))
        except Exception:
            slope_e.append(np.nan)
            period_e.append(np.nan)
        # the baseline: a textbook OLS trend plus an FFT-periodogram cycle
        try:
            bdec = classical_decompose(t, y, trend_deg=1)
            bslope_e.append(_rel(bdec["slope"], slope))
            bperiod_e.append(_rel(bdec["period"], period))
        except Exception:
            bslope_e.append(np.nan)
            bperiod_e.append(np.nan)
    err = _mean([_mean(slope_e), _mean(period_e)])
    base_err = _mean([_mean(bslope_e), _mean(bperiod_e)])
    return {
        "id": "E6", "name": "trend + cycle decomposition",
        "param": "slope & period", "metric": "rel.err %",
        "dt_err": err, "base_err": base_err,
        "verdict": _verdict(err, 12.0, 30.0),
        "method": "LSI trend + oscillatory-recipe cycle",
        "extra": {"slope_err": _mean(slope_e), "period_err": _mean(period_e),
                  "OLS+periodogram": base_err},
    }


# E7, the merged solution: does the single fit_stochastic router identify the
# right regime for each process? This is the analogue of the other domains'
# merged pipeline and applicability map.
def _regime_match(model, expect: str) -> bool:
    """Does the model's detected regime / components match the expected label?"""
    reg = model.regime.lower()
    if expect == "white noise":
        return reg.startswith("white noise")
    if expect == "random walk":
        return reg.startswith("random walk")
    if expect == "mean-reverting":
        return "mean-revert" in reg
    if expect == "long-memory":
        return "long-memory" in reg
    if expect == "vol-clustering":
        return bool(model.has_vol_clustering)
    if expect == "cyclical":
        return any(w in reg for w in ("cyclical", "cycle", "seasonal"))
    if expect == "trend+cycle":
        return "trend+cycle" in reg or "trend+seasonal" in reg
    return False


def _router_cases() -> list[tuple[str, str, object]]:
    """``(process name, expected regime, generator(seed)->series)``."""
    return [
        ("white noise", "white noise",
         lambda s: np.random.default_rng(s).standard_normal(1500)),
        ("random walk", "random walk",
         lambda s: np.cumsum(np.random.default_rng(s).standard_normal(1500))),
        ("AR(1)/OU", "mean-reverting",
         lambda s: gen_ar1(1500, 0.7, np.random.default_rng(s))),
        ("ARFIMA (long memory)", "long-memory",
         lambda s: gen_arfima(4096, 0.3, np.random.default_rng(s))),
        ("GARCH(1,1)", "vol-clustering",
         lambda s: gen_garch(4000, 0.05, 0.08, 0.90, np.random.default_rng(s))),
        ("AR(2) cycle", "cyclical",
         lambda s: gen_ar2_cycle(1500, 16.0, 0.97, np.random.default_rng(s))),
        ("trend + cycle", "trend+cycle",
         lambda s: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0,
                                   np.random.default_rng(s))[1]),
    ]


def exp_merged_router(seeds: int = 8) -> dict:
    """Run ``fit_stochastic`` over every process and score its regime-ID
    accuracy head-to-head against the classical-estimator twin
    ``fit_classical_stochastic``, on the same gated routing; the estimator is
    then the only difference between them."""
    rows = []
    correct = total = 0
    c_correct = 0
    for name, expect, gen in _router_cases():
        hits = c_hits = 0
        for s in range(seeds):
            series = gen(700 + s)
            try:
                if _regime_match(fit_stochastic(series), expect):
                    hits += 1
            except Exception:
                pass
            try:
                if _regime_match(fit_classical_stochastic(series), expect):
                    c_hits += 1
            except Exception:
                pass
        rows.append({"process": name, "expected": expect,
                     "detected %": 100.0 * hits / seeds,
                     "classical %": 100.0 * c_hits / seeds})
        correct += hits
        c_correct += c_hits
        total += seeds
    acc = 100.0 * correct / total if total else float("nan")
    c_acc = 100.0 * c_correct / total if total else float("nan")
    return {
        "id": "E7", "name": "merged router (regime identification)",
        "param": "regime", "metric": "accuracy %",
        "dt_err": 100.0 - acc, "base_err": 100.0 - c_acc,
        "verdict": _verdict(100.0 - acc, 10.0, 25.0),
        "method": "fit_stochastic gated composition", "accuracy": acc,
        "classical_accuracy": c_acc, "rows": rows,
    }


# forecast skill: the merged solution against the established forecasters
def forecast_skill(y: np.ndarray, h: int) -> tuple[dict, str]:
    """Hold out the last ``h`` points and return ``({method: RMSE}, regime)``
    for the merged solution and the established baselines: random walk, drift,
    AR(1), ARIMA, ETS and Theta, the statsmodels ones skipped where
    unavailable."""
    y = np.asarray(y, dtype=float)
    train, test = y[:-h], y[-h:]
    out: dict[str, float] = {}
    model = fit_stochastic(train)
    out["dtfit merged"] = metrics(test, model.forecast(h))["RMSE"]
    out["random walk"] = metrics(test, bl.random_walk_forecast(train, h))["RMSE"]
    out["drift"] = metrics(test, bl.drift_forecast(train, h))["RMSE"]
    for label, fn in [
        ("AR(1)", lambda: bl.arima_forecast(train, h, order=(1, 0, 0))),
        ("ARIMA(2,1,2)", lambda: bl.arima_forecast(train, h, order=(2, 1, 2))),
        ("ETS", lambda: bl.ets_forecast(train, h)),
        ("Theta", lambda: bl.theta_forecast(train, h)),
    ]:
        try:
            out[label] = metrics(test, fn())["RMSE"]
        except Exception:
            out[label] = float("nan")
    return out, model.regime


def exp_forecast_skill(seeds: int = 5) -> dict:
    """Forecast skill of the merged solution against the baselines, on the
    structured processes where extrapolation means anything at all: trend+cycle
    and mean-reverting. Reported as an RMSE ratio to the random-walk benchmark,
    so below 1 beats the random walk."""
    cases = [
        ("trend + cycle", lambda s: gen_trend_cycle(
            400, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(s))[1], 40),
        ("AR(1) mean-revert", lambda s: gen_ar1(
            800, 0.6, np.random.default_rng(s)), 30),
    ]
    rows = []
    for name, gen, h in cases:
        ratios = {}
        for s in range(seeds):
            sk, _ = forecast_skill(gen(800 + s), h)
            rw = sk.get("random walk", np.nan)
            for k, v in sk.items():
                ratios.setdefault(k, []).append(v / rw if rw else np.nan)
        rows.append({"process": name,
                     **{k: _mean(v) for k, v in ratios.items()}})
    return {"id": "E8", "name": "forecast skill (RMSE ratio to random walk)",
            "rows": rows}


# E8b, the head-to-head: one stochastic model, dtfit's estimators against the
# classical ones. It settles whether routing the characterization through
# dtfit's integral fitters improves on the textbook toolkit, on the one axis
# that matters end to end, held-out forecast skill.
def exp_model_comparison(seeds: int = 5) -> dict:
    """``fit_stochastic`` against its classical twin
    ``fit_classical_stochastic``, on held-out forecasting across the structured
    regimes.

    Both models run the same gated routing and the same regime-appropriate
    forecaster set, leaving the estimator as the only difference: dtfit's
    LSI/EAC integral fits against OLS, GARCH-QMLE, periodogram and DFA. For
    each process the last ``h`` points are held out and scored by RMSE, and the
    table reports each model's RMSE ratio to the random walk, below 1 beating
    it, together with which model wins, so that any improvement from the dtfit
    route is explicit.
    """
    cases = [
        ("trend + cycle", lambda s: gen_trend_cycle(
            400, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(s))[1], 40),
        ("AR(2) cycle", lambda s: gen_ar2_cycle(
            1200, 16.0, 0.97, np.random.default_rng(s)), 40),
        ("AR(1) mean-revert", lambda s: gen_ar1(
            800, 0.6, np.random.default_rng(s)), 30),
        ("trend (linear)", lambda s: gen_trend_cycle(
            400, 0.05, 1e9, 0.0, 1.0, np.random.default_rng(s))[1], 40),
    ]
    rows = []
    dtfit_wins = 0
    for name, gen, h in cases:
        dt_r, cl_r = [], []
        for s in range(seeds):
            y = gen(900 + s)
            train, test = y[:-h], y[-h:]
            rw = metrics(test, bl.random_walk_forecast(train, h))["RMSE"] + 1e-12
            try:
                dt_r.append(metrics(test, fit_stochastic(train).forecast(h))["RMSE"] / rw)
            except Exception:
                dt_r.append(np.nan)
            try:
                cl_r.append(metrics(
                    test, fit_classical_stochastic(train).forecast(h))["RMSE"] / rw)
            except Exception:
                cl_r.append(np.nan)
        dt_ratio, cl_ratio = _mean(dt_r), _mean(cl_r)
        winner = ("dtfit" if dt_ratio < cl_ratio else "classical"
                  if np.isfinite(dt_ratio) and np.isfinite(cl_ratio) else "--")
        dtfit_wins += int(winner == "dtfit")
        rows.append({"process": name, "h": h, "dtfit/RW": dt_ratio,
                     "classical/RW": cl_ratio, "winner": winner})
    return {
        "id": "E8b", "name": "model head-to-head (dtfit vs classical estimators)",
        "rows": rows, "dtfit_wins": dtfit_wins, "n_cases": len(cases),
        "verdict": ("dtfit improves" if dtfit_wins > len(cases) / 2
                    else "comparable" if dtfit_wins == len(cases) / 2
                    else "classical better"),
    }


# E9, real economic data: the USD/UAH 2014-15 daily rate
def load_series(name: str, col: int = 1) -> np.ndarray:
    """Load one column from a bundled real-data CSV (``experiments/data``)."""
    import csv

    rows = list(csv.reader((EXPERIMENTS_DIR / "data" / name).open()))[1:]
    return np.array([float(r[col]) for r in rows if r and r[col] not in ("", "NA")])


def exp_real_data() -> dict:
    """Characterize a real economic series with the merged solution and check the
    forecast against the established baselines.

    The USD/UAH 2014-15 daily exchange rate. Its level is a near-random walk,
    the classic finding that nothing beats persistence on an FX level, while
    its log-returns are the stationary object carrying the volatility
    clustering. The merged router should therefore call the level a random walk
    and tie the random walk on forecast, and flag the clustering in the
    returns.
    """
    rate = load_series("usd_uah_2014_2015.csv")
    logret = np.diff(np.log(rate))

    m_level = fit_stochastic(rate)
    m_ret = fit_stochastic(logret)

    h = 20
    skill, regime = forecast_skill(rate, h)

    # Hurst of |log-returns|, volatility long memory: dtfit vs the classics
    absr = np.abs(logret - logret.mean())
    hurst_abs = {
        "dtfit spectral": _safe(lambda: hurst_spectral(absr)["H"]),
        "dtfit aggvar": _safe(lambda: hurst_aggvar(absr)["H"]),
        "R/S": _safe(lambda: bl.hurst_rs(absr)),
        "DFA": _safe(lambda: bl.hurst_dfa(absr)),
    }
    return {
        "id": "E9", "name": "real data (USD/UAH 2014-15)",
        "level_regime": m_level.regime,
        "level_components": ", ".join(m_level.components),
        "returns_vol_clustering": bool(m_ret.has_vol_clustering),
        "returns_vol_persistence": m_ret.vol_persistence,
        "forecast_regime": regime,
        "forecast_rmse": skill,
        "abs_returns_hurst": hurst_abs,
    }


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return float("nan")


# E10, a gallery of real datasets, one per regime, against known literature
def _sm_series(key: str) -> np.ndarray:
    """Load a canonical real series from ``statsmodels.datasets`` (no download)."""
    import statsmodels.api as sm

    if key == "nile":
        return sm.datasets.nile.load_pandas().data["volume"].to_numpy(float)
    if key == "sunspots":
        return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)
    if key == "co2":
        s = sm.datasets.co2.load_pandas().data["co2"]
        return s.interpolate().to_numpy(float)
    if key == "gdp":
        return sm.datasets.macrodata.load_pandas().data["realgdp"].to_numpy(float)
    if key == "tbill":
        return sm.datasets.macrodata.load_pandas().data["tbilrate"].to_numpy(float)
    raise KeyError(key)


# Each entry pairs a real series with the regime domain knowledge and the
# literature predict for it, and with the established result it should
# reproduce. ``source`` is a thunk, so a missing statsmodels or data file fails
# that row alone.
REAL_DATASETS = [
    dict(key="nile", title="Nile annual volume (1871-1970)", domain="hydrology",
         expect="long memory / trend", source=lambda: _sm_series("nile"),
         lit="Hurst's canonical long-memory series (H ~ 0.9)"),
    dict(key="sunspots", title="Sunspot number (1700-2008)", domain="solar physics",
         expect="cyclical", source=lambda: _sm_series("sunspots"),
         lit="the ~11-year solar cycle"),
    dict(key="co2", title="Mauna Loa CO2 (weekly)", domain="climate",
         expect="trend+cycle", source=lambda: _sm_series("co2"),
         lit="rising trend + annual cycle"),
    dict(key="gdp", title="US real GDP (quarterly)", domain="macroeconomics",
         expect="random walk + drift", source=lambda: _sm_series("gdp"),
         lit="Nelson-Plosser: GDP is a random walk with drift"),
    dict(key="tbill", title="US 3-month T-bill rate (quarterly)",
         domain="macroeconomics", expect="random walk / mean-revert",
         source=lambda: _sm_series("tbill"),
         lit="short rates are near-unit-root (highly persistent)"),
    dict(key="usd_uah", title="USD/UAH (2014-15, daily)", domain="FX",
         expect="random walk", source=lambda: load_series("usd_uah_2014_2015.csv"),
         lit="an FX level is a near-random walk; clustering in the returns"),
    dict(key="fx_ltsf", title="LTSF exchange rate (daily)", domain="FX",
         expect="random walk", source=lambda: load_series("ltsf/exchange_rate.csv", -1),
         lit="FX level near-random walk; long memory in volatility"),
]


def hurst_comparison(y: np.ndarray) -> dict[str, float]:
    """Hurst of ``y`` by every estimator (dtfit spectral / aggvar, R/S, DFA)."""
    return {
        "dtfit spectral": _safe(lambda: hurst_spectral(y)["H"]),
        "dtfit aggvar": _safe(lambda: hurst_aggvar(y)["H"]),
        "R/S": _safe(lambda: bl.hurst_rs(y)),
        "DFA": _safe(lambda: bl.hurst_dfa(y)),
    }


def _suite_horizon(n: int) -> int:
    """Held-out forecast horizon scaled to the series length."""
    return int(min(max(n // 8, 12), 40))


def exp_real_suite() -> dict:
    """Run the merged solution on every real dataset, reporting the detected
    regime and the held-out forecast skill of the merged solution and of every
    baseline, as an RMSE ratio to the random walk where below 1 beats it."""
    rows = []
    for ds in REAL_DATASETS:
        try:
            y = ds["source"]()
        except Exception as exc:
            rows.append({"dataset": ds["title"], "domain": ds["domain"], "n": 0,
                         "expected": ds["expect"],
                         "detected regime": f"(unavailable: {type(exc).__name__})",
                         "components": "", "h": 0, "merged/RW": float("nan"),
                         "skill": {}, "literature": ds["lit"]})
            continue
        m = fit_stochastic(y)
        h = _suite_horizon(y.size)
        ratio = float("nan")
        skill: dict[str, float] = {}
        if y.size > h + 60:
            try:
                sk, _ = forecast_skill(y, h)
                rw = sk.get("random walk", np.nan)
                skill = {k: (v / rw if rw else np.nan) for k, v in sk.items()}
                ratio = skill.get("dtfit merged", np.nan)
            except Exception:
                pass
        rows.append({
            "dataset": ds["title"], "domain": ds["domain"], "n": int(y.size),
            "expected": ds["expect"], "detected regime": m.regime,
            "components": ", ".join(m.components), "h": h, "merged/RW": ratio,
            "skill": skill, "literature": ds["lit"],
        })
    return {"id": "E10", "name": "real-data gallery (regime + forecast skill)",
            "rows": rows}


def panel_forecasts(y: np.ndarray, h: int) -> dict:
    """Held-out forecast of every method for one series, as arrays for plotting.

    Returns ``{"train", "test", "preds": {method: array}}``: the merged
    solution plus the established baselines over the last ``h`` points."""
    y = np.asarray(y, dtype=float)
    train, test = y[:-h], y[-h:]
    preds = {
        "dtfit merged": fit_stochastic(train).forecast(h),
        "random walk": bl.random_walk_forecast(train, h),
        "drift": bl.drift_forecast(train, h),
    }
    for label, fn in [
        ("ARIMA(2,1,2)", lambda: bl.arima_forecast(train, h, order=(2, 1, 2))),
        ("Theta", lambda: bl.theta_forecast(train, h)),
    ]:
        try:
            preds[label] = np.asarray(fn(), dtype=float)
        except Exception:
            preds[label] = np.full(h, np.nan)
    return {"train": train, "test": test, "preds": preds}


# E11, streaming characterization with online regime-change detection: the
# online twin of fit_stochastic's second-order stage, and the stochastic
# counterpart of the embedded domain's streaming filter and fault detector.
def filter_trace(y: np.ndarray, *, nlags: int = 24, halflife: float = 150.0,
                 warmup: int = 80, settle: int = 500,
                 z_thresh: float = 4.0) -> dict:
    """Run a :class:`StochasticFilter` over ``y`` and return the online phi and
    volatility traces with the detected change-point times, as arrays ready to
    plot."""
    f = StochasticFilter(nlags=nlags, halflife=halflife, warmup=warmup,
                         settle=settle, z_thresh=z_thresh)
    phi = np.empty(y.size)
    vol = np.empty(y.size)
    for i, x in enumerate(np.asarray(y, dtype=float)):
        f.update(x)
        p = f.params_
        phi[i] = p["ar1_phi"]
        vol[i] = p["sigma"]
    return {"phi": phi, "sigma": vol, "flags": list(f.flag_times_),
            "snapshot": f.snapshot()}


def exp_online_filter(seeds: int = 8) -> dict:
    """Score the streaming filter on three axes: whether the online AR(1) phi
    converges to the truth, whether it flags a structural break (a persistence
    jump, a volatility switch) promptly, and how often it false-alarms on a
    stationary stream. The streaming analogue of E7's batch router."""
    # (1) online phi tracking error against ground truth
    phi_err = []
    for phi in (0.3, 0.6, 0.85):
        for s in range(seeds):
            y = gen_ar1(3000, phi, np.random.default_rng(700 + s))
            f = StochasticFilter(halflife=300, warmup=100)
            f.partial_fit(y)
            phi_err.append(abs(f.params_["ar1_phi"] - phi))
    track_err = _mean(phi_err)

    # (2) regime-break detection, on a persistence jump and a volatility switch
    # at the midpoint, reporting hit-rate and median detection latency
    def _break_case(kind, s):
        r = np.random.default_rng(800 + s)
        if kind == "persistence":
            a = gen_ar1(1500, 0.2, r)
            b = gen_ar1(1500, 0.9, r)
            b = b - b.mean() + a[-1]
            return np.concatenate([a, b])
        a = r.normal(0.0, 1.0, 1500)
        b = r.normal(0.0, 2.0, 1500)
        return np.concatenate([a, b])

    hits, lats = 0, []
    total = 0
    for kind in ("persistence", "volatility"):
        for s in range(seeds):
            total += 1
            f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
            f.partial_fit(_break_case(kind, s))
            near = [t for t in f.flag_times_ if 1500 <= t <= 1900]
            if near:
                hits += 1
                lats.append(near[0] - 1500)
    hit_rate = 100.0 * hits / total if total else float("nan")
    latency = float(np.median(lats)) if lats else float("nan")

    # (3) the false-alarm rate on a stationary stream
    fa = []
    for s in range(seeds):
        f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
        f.partial_fit(gen_ar1(3000, 0.6, np.random.default_rng(900 + s)))
        fa.append(f.n_flags_)
    false_alarms = _mean(fa)

    return {
        "id": "E11", "name": "streaming filter (online tracking + break detection)",
        "param": "phi track / break hit-rate", "metric": "MAE / %",
        "dt_err": track_err, "base_err": float("nan"),
        "verdict": _verdict(track_err, 0.05, 0.12),
        "method": "EWMA-autocovariance StochasticFilter + fused change detector",
        "track_mae": track_err, "break_hit_rate": hit_rate,
        "median_latency": latency, "false_alarms_per_3000": false_alarms,
    }


# E12, the generative model on a fit-simulate-refit round trip.
# ``StochasticModel.simulate`` draws a fresh realization from the detected
# components, and re-fitting that must recover the same regime. It is the
# honest proof that the batch model is a faithful generator and not only a
# summary.
def exp_simulate_roundtrip(seeds: int = 5) -> dict:
    """Per-regime round-trip recovery rate of ``StochasticModel.simulate``."""
    cases = [
        ("trend + cycle",
         lambda r: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, r)[1], "has_cycle"),
        ("AR(1) mean-reverting", lambda r: gen_ar1(1500, 0.7, r),
         "has_mean_reversion"),
        ("ARFIMA long-memory", lambda r: gen_arfima(4096, 0.3, r),
         "has_long_memory"),
        ("GARCH vol-clustering",
         lambda r: gen_garch(4000, 0.05, 0.08, 0.90, r), "has_vol_clustering"),
        ("random walk", lambda r: np.cumsum(r.standard_normal(1500)), None),
        ("white noise", lambda r: r.standard_normal(1500), None),
    ]
    rows = []
    for name, gen, attr in cases:
        hits = 0
        for s in range(seeds):
            m = fit_stochastic(gen(np.random.default_rng(10 + s)))
            m2 = fit_stochastic(m.simulate(seed=s))
            if attr is None:
                hits += int(m.regime.split()[0] == m2.regime.split()[0])
            else:
                hits += int(bool(getattr(m2, attr)))
        rows.append({"process": name, "round-trip recovery %": 100.0 * hits / seeds})
    return {"id": "E12", "name": "generative round-trip (fit->simulate->refit)",
            "rows": rows}


def simulate_example(seed: int = 0) -> dict:
    """One original vs simulated realization per regime, for the figure."""
    specs = [
        ("trend + cycle",
         gen_trend_cycle(500, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(1))[1]),
        ("AR(1) mean-reverting", gen_ar1(800, 0.75, np.random.default_rng(2))),
        ("GARCH vol-clustering",
         gen_garch(2000, 0.05, 0.08, 0.90, np.random.default_rng(3))),
    ]
    out = []
    for name, y in specs:
        m = fit_stochastic(y)
        out.append({"name": name, "regime": m.regime,
                    "original": y, "simulated": m.simulate(seed=seed)})
    return {"examples": out}


# The streaming-filter demonstrations that accompany E11: a change-point trace,
# and the flat-memory, bounded-cost profile that matches dtfit's own streaming
# filters.
def filter_break_demo(*, n_seg: int = 1500, seed: int = 0) -> dict:
    """A stream carrying a known structural break at the midpoint, an AR(1)
    persistence jump from 0.2 to 0.9, run through :class:`StochasticFilter`.
    Returns the series, the true break, the online phi trace and the flags the
    filter raised."""
    r = np.random.default_rng(seed)
    a = gen_ar1(n_seg, 0.2, r)
    b = gen_ar1(n_seg, 0.9, r)
    b = b - b.mean() + a[-1]
    y = np.concatenate([a, b])
    tr = filter_trace(y, halflife=220.0, warmup=80, settle=500, z_thresh=4.0)
    return {"y": y, "true_break": n_seg, "phi": tr["phi"], "sigma": tr["sigma"],
            "flags": tr["flags"], "true_phi": (0.2, 0.9)}


def filter_characteristics(lengths: tuple[int, ...] = (1000, 10000, 100000)) -> dict:
    """Memory and per-sample-cost profile of :class:`StochasticFilter` against
    stream length, with a reference dtfit ``LSIFilter`` per-sample cost. This
    is the evidence that the filter has the flat-memory, bounded-cost
    characteristics of dtfit's own streaming filters: its state is independent
    of stream length and its cost per sample does not grow."""
    import sys
    import time

    rng = np.random.default_rng(0)
    rows = []
    for n in lengths:
        f = StochasticFilter()
        y = gen_ar1(n, 0.6, rng)
        t0 = time.perf_counter()
        f.partial_fit(y)
        us = (time.perf_counter() - t0) / n * 1e6
        state = (sys.getsizeof(f._buf) + f._acov.nbytes + f._vacov.nbytes
                 + sys.getsizeof(f.flag_times_))
        rows.append({"stream length N": int(n), "filter state (bytes)": int(state),
                     "us / sample": round(us, 1)})
    ref = float("nan")
    try:
        from dtfit import LSIFilter
        f2 = LSIFilter("a0 + a1*t", "t", window_size=60)
        y = np.cumsum(rng.standard_normal(5000))
        t0 = time.perf_counter()
        for i, v in enumerate(y):
            f2.partial_fit(float(i), float(v))
        ref = round((time.perf_counter() - t0) / y.size * 1e6, 1)
    except Exception:
        pass
    return {"rows": rows, "lsifilter_us_per_sample": ref}


# E13 and E14: the regime router's finite-order-AR veto, and the estimator
# capabilities ar_order, fit_ar, fractional_difference and Student-t
# innovations.
def gen_arp(n: int, phi: tuple[float, ...], rng: np.random.Generator,
            *, sigma: float = 1.0, burn: int = 300) -> np.ndarray:
    """General AR(p): ``x_t = sum_j phi_j x_{t-j} + e_t`` (lag 1..p)."""
    p = len(phi)
    N = n + burn
    e = rng.normal(0.0, sigma, N)
    x = np.zeros(N)
    phi_a = np.asarray(phi, dtype=float)
    for t in range(p, N):
        x[t] = float(phi_a @ x[t - p:t][::-1]) + e[t]
    return x[burn:]


# The ground-truth zoo for the AR-versus-long-memory discrimination guard. Each
# entry is ``(name, expect_long_memory, generator(seed) -> series)``. The AR(2)
# complex-root case and the AR(3) case are the ones a router without the veto
# mislabels as long memory, their slowly-decaying ACF spoofing the Hurst
# read-out; the veto whitens with a capped AR(p<=3) fit first.
def _ar_discrim_cases(
    n: int,
) -> list[tuple[str, bool, Callable[[int], np.ndarray]]]:
    # ARFIMA is generated at the canonical long-memory sample size, 4096 as in
    # E1/E2, where the Hurst read-out is reliable; the short-memory AR
    # processes use the router's own default working length ``n``.
    n_lm = 4096
    return [
        ("AR(1) phi=0.70", False,
         lambda s: gen_ar1(n, 0.70, np.random.default_rng(s))),
        ("AR(1) phi=0.95", False,
         lambda s: gen_ar1(n, 0.95, np.random.default_rng(s))),
        ("AR(2) cycle (P~16, r=0.97)", False,
         lambda s: gen_ar2_cycle(n, 16.0, 0.97, np.random.default_rng(s))),
        ("AR(3)", False,
         lambda s: gen_arp(n, (0.5, -0.3, 0.2), np.random.default_rng(s))),
        ("ARFIMA d=0.3", True,
         lambda s: gen_arfima(n_lm, 0.3, np.random.default_rng(s))),
        ("ARFIMA d=0.4", True,
         lambda s: gen_arfima(n_lm, 0.4, np.random.default_rng(s))),
    ]


def exp_ar_discrimination(seeds: int = 8, n: int = 1500) -> dict:
    """The headline regression guard for ``fit_stochastic``'s finite-order-AR
    veto. A short-memory AR(2) or AR(3) has a slowly-decaying ACF that spoofs
    the raw Hurst read-out into calling it long memory; the gate whitens with a
    capped AR(p<=3) fit and, if the whitened Hurst falls back toward white,
    reclassifies the process as mean-reverting. This experiment simulates each
    process over several seeds and records how often ``fit_stochastic`` flags
    long memory.

    The contract the verdict asserts: every AR(1), AR(2) and AR(3) must come
    out 0% long-memory, since they are mean-reverting, and being the
    load-bearing regression guard that is strict; while both ARFIMA d=0.3 and
    d=0.4 must be a strong long-memory majority at 75% or more. A lone
    borderline realization of ARFIMA d=0.4 can genuinely read as short-memory
    near the detector's boundary, which is honest sampling noise and not a
    router bug. A regression that reintroduces the mislabel shows up as a
    non-zero long-memory fraction on an AR row."""
    rows = []
    ok = True
    for name, is_lm, gen in _ar_discrim_cases(n):
        lm = mr = 0
        for s in range(seeds):
            try:
                m = fit_stochastic(gen(1300 + s))
                lm += int(bool(m.has_long_memory))
                mr += int(bool(m.has_mean_reversion))
            except Exception:
                pass
        lm_frac = 100.0 * lm / seeds
        mr_frac = 100.0 * mr / seeds
        # The per-row contract: an AR process must be strictly 0% long-memory,
        # which is the veto; ARFIMA must be a strong long-memory majority.
        if is_lm:
            row_ok = lm_frac >= 75.0
        else:
            row_ok = lm_frac == 0.0
        ok = ok and row_ok
        rows.append({"process": name, "expect": "long-memory" if is_lm
                     else "mean-reverting", "long-memory %": lm_frac,
                     "mean-reverting %": mr_frac, "pass": row_ok})
    return {
        "id": "E13", "name": "AR(p) vs long-memory discrimination (veto guard)",
        "param": "regime", "metric": "long-memory %",
        "rows": rows, "contract_holds": ok,
        "verdict": "PASS (veto holds)" if ok else "FAIL (regime mislabel)",
        "method": "fit_stochastic finite-order-AR (p<=3) whitening veto",
    }


def exp_ar_order_recovery(seeds: int = 8) -> dict:
    """Order and coefficient recovery for ``ar_order`` and ``fit_ar``.

    AR(1), AR(2) and AR(3) with known coefficients. Per true order it reports
    how often ``ar_order`` picks the true ``p`` by AIC, the mean
    per-coefficient recovery error ``mean|phi_hat - phi|`` over the shortest
    shared length, and the mean recovered innovation ``sigma``, whose truth is
    1.0."""
    specs = [
        ("AR(1)", (0.6,)),
        ("AR(2)", (0.5, -0.3)),
        ("AR(3)", (0.5, -0.3, 0.2)),
    ]
    rows = []
    for name, phi in specs:
        p = len(phi)
        phi_a = np.asarray(phi, dtype=float)
        order_hits = 0
        coef_err: list[float] = []
        sig: list[float] = []
        for s in range(seeds):
            x = gen_arp(2000, phi, np.random.default_rng(1400 + s))
            try:
                order_hits += int(ar_order(x, max_order=8, ic="aic") == p)
            except Exception:
                pass
            try:
                fa = fit_ar(x, order=p)
                phi_hat = np.asarray(fa["phi"], dtype=float)
                coef_err.append(float(np.mean(np.abs(phi_hat - phi_a))))
                sig.append(float(np.asarray(fa["sigma"], dtype=float)))
            except Exception:
                coef_err.append(np.nan)
                sig.append(np.nan)
        rows.append({
            "process": name, "true order": p,
            "order picked %": 100.0 * order_hits / seeds,
            "mean|phi err|": _mean(coef_err), "mean sigma (truth 1.0)": _mean(sig),
        })
    return {"id": "E14a", "name": "AR order + coefficient recovery (ar_order/fit_ar)",
            "rows": rows,
            "method": "Yule-Walker AR(p), order gate by AIC"}


def exp_fracdiff_whitening(seeds: int = 8, n: int = 2048) -> dict:
    """``fractional_difference`` as the long-memory whitener.

    For ARFIMA(0,d,0) with d in {0.3, 0.4} (Hurst ``H = d + 1/2`` ~ 0.8 / 0.9):
    read ``d_hat = hurst_spectral(x)["H"] - 0.5`` off the series, apply
    ``(1 - B)^d_hat``, and measure the Hurst BEFORE vs AFTER. A working
    fractional difference should drive the after-Hurst back toward 0.5 (white)."""
    rows = []
    for d in (0.3, 0.4):
        H = d + 0.5
        h_before: list[float] = []
        h_after: list[float] = []
        d_hats: list[float] = []
        for s in range(seeds):
            x = gen_arfima(n, d, np.random.default_rng(1500 + s))
            try:
                hb = float(hurst_spectral(x)["H"])
                d_hat = hb - 0.5
                w = fractional_difference(x, d_hat)
                ha = float(hurst_spectral(w)["H"])
                h_before.append(hb)
                h_after.append(ha)
                d_hats.append(d_hat)
            except Exception:
                h_before.append(np.nan)
                h_after.append(np.nan)
                d_hats.append(np.nan)
        rows.append({
            "process": f"ARFIMA d={d}", "true H": H,
            "d_hat (mean)": _mean(d_hats),
            "H before": _mean(h_before), "H after": _mean(h_after),
        })
    return {"id": "E14b", "name": "fractional differencing whitens long memory",
            "rows": rows,
            "method": "d_hat = H_spectral - 0.5 ; (1 - B)^d_hat filter"}


def exp_student_t(seeds: int = 6, n: int = 4000) -> dict:
    """Student-t innovations in ``StochasticModel.simulate``.

    Fit a fat-tailed AR(1), driven by heavy-tailed Student-t shocks, with
    ``fit_stochastic``, then draw a Gaussian path (``dist="normal"``) and a
    Student-t path (``dist="t", df=5``). Both share the same fitted
    second-order scale, so their path variances should stay within a factor of
    two of each other, while the Student-t path should carry markedly heavier
    tails, a higher excess kurtosis. This checks the innovation-distribution
    knob, not a forecast."""
    def _excess_kurtosis(a: np.ndarray) -> float:
        """Fisher excess kurtosis, 0 for a Gaussian; numpy only, no scipy."""
        a = np.asarray(a, dtype=float)
        m = a.mean()
        s2 = a.var()
        if s2 <= 0:
            return float("nan")
        return float(np.mean((a - m) ** 4) / s2 ** 2 - 3.0)

    var_n: list[float] = []
    var_t: list[float] = []
    kurt_n: list[float] = []
    kurt_t: list[float] = []
    for s in range(seeds):
        r = np.random.default_rng(1600 + s)
        # an AR(1) driven by heavy-tailed Student-t (df=4) innovations
        shocks = r.standard_t(4, n + 200)
        x = np.empty(n + 200)
        x[0] = shocks[0]
        for t in range(1, n + 200):
            x[t] = 0.5 * x[t - 1] + shocks[t]
        x = x[200:]
        try:
            m = fit_stochastic(x)
            sn = np.asarray(m.simulate(n, seed=s, dist="normal"), dtype=float)
            st = np.asarray(m.simulate(n, seed=s, dist="t", df=5.0), dtype=float)
            # the variance of the path itself is the honest scale check here
            var_n.append(float(np.var(sn)))
            var_t.append(float(np.var(st)))
            kurt_n.append(_excess_kurtosis(sn))
            kurt_t.append(_excess_kurtosis(st))
        except Exception:
            var_n.append(np.nan)
            var_t.append(np.nan)
            kurt_n.append(np.nan)
            kurt_t.append(np.nan)
    vn, vt = _mean(var_n), _mean(var_t)
    kn, kt = _mean(kurt_n), _mean(kurt_t)
    var_ratio = vt / vn if np.isfinite(vn) and vn != 0 else float("nan")
    return {
        "id": "E14c", "name": "Student-t innovations (fatter tails, matched scale)",
        "rows": [
            {"dist": "normal", "path variance": vn, "excess kurtosis": kn},
            {"dist": "t (df=5)", "path variance": vt, "excess kurtosis": kt},
        ],
        "var_ratio_t_over_normal": var_ratio,
        "kurt_gain": kt - kn,
        "verdict": ("VIABLE (fatter tails, matched variance)"
                    if np.isfinite(kt) and np.isfinite(kn) and kt > kn
                    and np.isfinite(var_ratio) and 0.5 < var_ratio < 2.0
                    else "MARGINAL"),
        "method": "fit_stochastic -> simulate(dist=normal) vs simulate(dist=t, df=5)",
    }


# the driver
def run(seeds: int = 8, *, quick: bool = False) -> list[dict]:
    """Run the six recovery experiments, the merged router and the streaming
    filter, and return their rows."""
    if quick:
        seeds = max(2, seeds // 2)
    return [
        exp_hurst_aggvar(seeds, n=2048 if quick else 4096),
        exp_hurst_spectral(seeds, n=2048 if quick else 4096),
        exp_ar1(seeds, n=1200 if quick else 1500),
        exp_garch(seeds, n=2500 if quick else 4000),
        exp_cycle(seeds, n=1000 if quick else 1200),
        exp_decompose(seeds, n=500 if quick else 600),
        exp_merged_router(seeds),
        exp_online_filter(max(3, seeds // 2) if quick else seeds),
    ]


def summary(rows: list[dict]) -> str:
    """Render the verdict table as GitHub-flavoured markdown."""
    head = ("| ID | possibility | parameter | dtfit err | baseline err | "
            "verdict |")
    sep = "|----|-------------|-----------|-----------|--------------|---------|"
    lines = [head, sep]
    for r in rows:
        de = "--" if not np.isfinite(r["dt_err"]) else f"{r['dt_err']:.3g}"
        be = "--" if not np.isfinite(r["base_err"]) else f"{r['base_err']:.3g}"
        lines.append(
            f"| {r['id']} | {r['name']} | {r['param']} ({r['metric']}) | "
            f"{de} | {be} | {r['verdict']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import sys

    quick = "--quick" in sys.argv
    rows = run(seeds=6, quick=quick)
    print(summary(rows))
    for r in rows:
        if "extra" in r:
            print(f"  {r['id']} detail: {r['extra']}")
        if r["id"] == "E7":
            print(f"  E7 regime-ID accuracy: dtfit {r['accuracy']:.0f}%  vs "
                  f"classical {r['classical_accuracy']:.0f}%")
            for rr in r["rows"]:
                print(f"     {rr['process']:<22} {rr['expected']:<14} "
                      f"dtfit {rr['detected %']:.0f}%  classical {rr['classical %']:.0f}%")
        if r["id"] == "E11":
            print(f"  E11 streaming: phi track MAE {r['track_mae']:.3f}, "
                  f"break hit-rate {r['break_hit_rate']:.0f}%, "
                  f"median latency {r['median_latency']:.0f} steps, "
                  f"false alarms {r['false_alarms_per_3000']:.2f}/3000")

    print("\n---- E8 forecast skill (RMSE ratio to random walk; <1 beats RW) ----")
    fs = exp_forecast_skill(seeds=4)
    for rr in fs["rows"]:
        print(" ", rr["process"], {k: round(v, 2) for k, v in rr.items()
                                    if k != "process"})

    print("\n---- E8b model head-to-head: dtfit vs classical estimators ----")
    cmp = exp_model_comparison(seeds=4)
    for rr in cmp["rows"]:
        print(f"  {rr['process']:<20} dtfit/RW={rr['dtfit/RW']:.2f} "
              f"classical/RW={rr['classical/RW']:.2f}  -> {rr['winner']}")
    print(f"  verdict: {cmp['verdict']} (dtfit wins {cmp['dtfit_wins']}/"
          f"{cmp['n_cases']} forecast cases; the dtfit edge is in regime ID, see E7)")

    print("\n---- E9 real data: USD/UAH 2014-15 ----")
    rd = exp_real_data()
    print(f"  level regime    : {rd['level_regime']}  [{rd['level_components']}]")
    print(f"  returns vol-clust: {rd['returns_vol_clustering']} "
          f"(persistence {rd['returns_vol_persistence']:.3f})")
    print("  |returns| Hurst  : "
          + ", ".join(f"{k}={v:.3f}" for k, v in rd["abs_returns_hurst"].items()))
    print(f"  forecast ({rd['forecast_regime']}) RMSE: "
          + ", ".join(f"{k}={v:.4g}" for k, v in rd["forecast_rmse"].items()))

    print("\n---- E10 real-data gallery (one per regime) ----")
    suite = exp_real_suite()
    for rr in suite["rows"]:
        ratio = rr["merged/RW"]
        ratio_s = "--" if not np.isfinite(ratio) else f"{ratio:.2f}xRW"
        print(f"  {rr['dataset']:<34} n={rr['n']:<5} -> {rr['detected regime']:<22} "
              f"[{rr['components']}]  fc {ratio_s}")
        print(f"      expect {rr['expected']:<22} | {rr['literature']}")
