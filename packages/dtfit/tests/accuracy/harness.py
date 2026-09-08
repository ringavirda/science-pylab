"""Scoring and fitting helpers shared by the accuracy corpus.

Fitting a scenario, scoring its recovery or its curve quality, running the
gold-standard NLLS baseline against it: each is implemented once here. The
accuracy gate, the seed-robustness tests, the golden snapshot and the
``dtfit-experimental`` harness all judge a fit through the same code.
"""

from __future__ import annotations

import warnings

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from .scenarios import Scenario


def ordered_params(scn: Scenario) -> list[str]:
    """Model parameter names in the sorted order the fitters lay coeffs out
    in."""
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    return [
        str(s) for s in sorted((s for s in f.free_symbols if s != t), key=str)
    ]


def r2(clean: np.ndarray, pred: np.ndarray) -> float:
    """R^2 of ``pred`` against the clean signal, not the noisy
    observations."""
    pred = np.asarray(pred, float)
    if pred.ndim == 0:
        pred = np.full_like(clean, float(pred))
    if not np.all(np.isfinite(pred)):
        return -np.inf
    ss_res = float(np.sum((pred - clean) ** 2))
    ss_tot = float(np.sum((clean - clean.mean()) ** 2)) or 1e-30
    return 1.0 - ss_res / ss_tot


def param_err(scn: Scenario, names: list[str], est: np.ndarray) -> float:
    """Max over parameters of the relative recovery error
    |est - true|/|true|."""
    errs = []
    for nm, v in zip(names, np.asarray(est, float)):
        tv = scn.true[nm]
        errs.append(abs(float(v) - tv) / (abs(tv) + 1e-9))
    return float(max(errs)) if errs else float("nan")


def predict(res, x: np.ndarray) -> np.ndarray:
    pred = np.asarray(res.model(x), float)
    return np.full_like(x, float(pred)) if pred.ndim == 0 else pred


# Noise draws every corpus measurement is medianed over. One draw of a
# recovery error is a half-normal draw: measured over the corpus, the value
# at a single seed moves by up to 8.1x (90th percentile) between disjoint
# seed blocks, against 3.8x for the median of five, so a single seed carries
# no information about the method and only the median is worth pinning.
SEEDS = range(5)


def metrics_for(scn: Scenario, noise: float, seeds=SEEDS) -> dict:
    """Recovery metrics for the self-seeded ``Model.fit`` path, medianed
    over ``seeds`` noise draws.

    This is the measurement the golden baseline snapshots and the regression
    guard re-checks. Returns ``{"metric", "perr", "r2"}``, each the median
    over the draws; a non-finite ``perr`` becomes 1e9 and a non-finite ``r2``
    becomes -1e9 before the median, which keeps the dict JSON-serialisable.
    """
    import warnings

    names = ordered_params(scn)
    perrs, r2s = [], []
    for seed in seeds:
        x, y, clean = scn.make(noise, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scn.model().fit(x, y)
        perr = param_err(scn, names, np.asarray(res.coeffs, float))
        got_r2 = r2(clean, predict(res, x))
        perrs.append(float(perr) if np.isfinite(perr) else 1e9)
        r2s.append(float(got_r2) if np.isfinite(got_r2) else -1e9)
    return {
        "metric": scn.metric,
        "perr": float(np.median(perrs)),
        "r2": float(np.median(r2s)),
    }


def curve_fit_baseline(scn: Scenario, x, y, names):
    """``(popt, pred)`` from ``scipy.optimize.curve_fit``, seeded by the
    model's own data-driven guess. This is the reference NLLS fit the methods
    are measured against. A bounded scenario goes down curve_fit's trf branch
    while an unbounded one uses Levenberg-Marquardt. Returns
    ``(None, reason)`` if curve_fit raises."""
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    syms = [sp.Symbol(n) for n in names]
    fn = sp.lambdify((t, *syms), f, "numpy")
    p0, bounds = m._seed_arrays(x, y)
    if p0 is None:
        p0 = [1.0] * len(names)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bounds is not None:
                lo = [b[0] for b in bounds]
                hi = [b[1] for b in bounds]
                popt, _ = curve_fit(
                    fn, x, y, p0=p0, bounds=(lo, hi), maxfev=20000
                )
            else:
                popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
        pred = np.asarray(fn(x, *popt), dtype=float)
        return np.asarray(popt, float), pred
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
