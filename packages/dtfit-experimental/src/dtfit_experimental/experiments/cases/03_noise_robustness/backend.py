"""Data generator, sweeps and baselines for the noise-robustness case study.

``03_noise_robustness.ipynb`` imports this module and owns the presentation.

The study maps fitting accuracy along three axes, noise level, outlier fraction
and sample size, for four model families that between them cover the shapes a
curve fitter runs into: a pure exponential, a transcendental arctangent, an
oscillation, and an additive mix of polynomial and exponential terms. Every
method is scored against the clean signal, not the noisy samples: the reward
is for recovering the truth, never for chasing the noise.

:func:`noise_sweep` walks Gaussian noise levels. :func:`outlier_sweep` adds
gross outliers and stands the stock fits beside the two robustness routes, the
overlapping-window ensemble (#3) and the soft-L1 loss.
:func:`param_grid_parallel` reports median EAC parameter-recovery error over a
noise-by-size grid, fanned out through :func:`dtfit.fit_many`.

The dtfit integral fits are scored against SciPy ``curve_fit``, a degree-5
``numpy.polyfit`` and a scikit-learn MLP. SciPy and scikit-learn are optional;
a missing one surfaces as ``nan`` in that method's column instead of aborting
the sweep.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit import FittingProblem, fit_many, ensemble_fit

from dtfit_experimental.experiments.common import metrics
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "FAMILIES",
    "r2_clean",
    "noise_sweep",
    "outlier_sweep",
    "param_grid_parallel",
]

# Each family carries the dtfit expr plus a NumPy twin ``f`` with the
# (x, *params) signature curve_fit wants, so both see the identical model.
FAMILIES = {
    "exponential": dict(
        expr="a*exp(b*x)", var="x", true={"a": 1.0, "b": 1.2},
        clean=lambda x: 1.0 * np.exp(1.2 * x),
        f=lambda x, a, b: a * np.exp(b * x), p0=[1.0, 1.0],
        x=lambda n: np.linspace(0, 1.5, n)),
    "transcendental": dict(
        expr="a*atan(w*x)", var="x", true={"a": 2.0, "w": 3.0},
        clean=lambda x: 2.0 * np.arctan(3.0 * x),
        f=lambda x, a, w: a * np.arctan(w * x), p0=[1.0, 1.0],
        x=lambda n: np.linspace(0, 1.5, n)),
    "sine": dict(
        expr="A*sin(w*x)", var="x", true={"A": 2.0, "w": 1.5},
        clean=lambda x: 2.0 * np.sin(1.5 * x),
        f=lambda x, A, w: A * np.sin(w * x), p0=[1.5, 1.3],
        x=lambda n: np.linspace(0, 4 * np.pi, n)),
    "mixed": dict(
        expr="a0 + a1*x + a2*exp(a3*x)", var="x",
        true={"a0": 0.5, "a1": 0.2, "a2": 0.3, "a3": 0.4},
        clean=lambda x: 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x),
        f=lambda x, a0, a1, a2, a3: a0 + a1 * x + a2 * np.exp(a3 * x),
        p0=[1.0, 1.0, 1.0, 1.0], x=lambda n: np.linspace(0, 3, n)),
}


def _noisy(fam, n, noise, seed, outlier_frac=0.0):
    rng = np.random.default_rng(seed)
    x = fam["x"](n)
    clean = fam["clean"](x)
    y = clean + rng.normal(0, noise * clean.std(), n)
    if outlier_frac > 0:
        k = max(1, int(outlier_frac * n))
        idx = rng.choice(n, k, replace=False)
        y[idx] += rng.choice([-1, 1], k) * 6 * clean.std()
    return x, y, clean


def r2_clean(clean, pred):
    return metrics(clean, pred)["R2"]


def noise_sweep(fam, noises, n=120, seeds=4):
    """R2-vs-clean for each method across noise levels (averaged over seeds)."""
    out = {m: [] for m in ["EAC", "LSI", "curve_fit", "polyfit", "MLP"]}
    for noise in noises:
        acc = {m: [] for m in out}
        for s in range(seeds):
            x, y, clean = _noisy(fam, n, noise, s)
            try:
                acc["EAC"].append(r2_clean(clean, np.asarray(
                    dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["EAC"].append(np.nan)
            try:
                acc["LSI"].append(r2_clean(clean, np.asarray(
                    dt.fit_lsi(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["LSI"].append(np.nan)
            try:
                p = bl.scipy_curve_fit(x, y, fam["f"], fam["p0"])
                acc["curve_fit"].append(r2_clean(clean, fam["f"](x, *p)))
            except Exception:
                acc["curve_fit"].append(np.nan)
            try:
                acc["polyfit"].append(r2_clean(clean, bl.polyfit_predict(x, y, x, deg=5)))
            except Exception:
                acc["polyfit"].append(np.nan)
            try:
                acc["MLP"].append(r2_clean(clean, bl.mlp_curve(x, y, x, max_iter=800)))
            except Exception:
                acc["MLP"].append(np.nan)
        for m in out:
            out[m].append(np.nanmean(acc[m]) if any(np.isfinite(acc[m])) else np.nan)
    return out


def outlier_sweep(fam, fracs, n=120, seeds=5):
    """R2-vs-clean under outliers: stock fits vs the two robust routes."""
    methods = ["EAC", "LSI", "curve_fit", "EAC-ensemble", "EAC-softl1"]
    out = {m: [] for m in methods}
    for fr in fracs:
        acc = {m: [] for m in methods}
        for s in range(seeds):
            x, y, clean = _noisy(fam, n, 0.05, s, outlier_frac=fr)
            try:
                acc["EAC"].append(r2_clean(clean, np.asarray(
                    dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["EAC"].append(np.nan)
            try:
                acc["LSI"].append(r2_clean(clean, np.asarray(
                    dt.fit_lsi(x, y, fam["expr"], fam["var"], p0=fam["p0"]).model(x))))
            except Exception:
                acc["LSI"].append(np.nan)
            try:
                p = bl.scipy_curve_fit(x, y, fam["f"], fam["p0"])
                acc["curve_fit"].append(r2_clean(clean, fam["f"](x, *p)))
            except Exception:
                acc["curve_fit"].append(np.nan)
            try:
                e = ensemble_fit(x, y, fam["expr"], fam["var"], method="eac",
                                 n_windows=10, overlap=0.5, p0=fam["p0"])
                acc["EAC-ensemble"].append(r2_clean(clean, e.predict(x)))
            except Exception:
                acc["EAC-ensemble"].append(np.nan)
            try:
                r = dt.fit_eac(x, y, fam["expr"], fam["var"], p0=fam["p0"],
                               loss="soft_l1",
                               bounds=[(-10, 10)] * len(fam["p0"]))
                acc["EAC-softl1"].append(r2_clean(clean, np.asarray(r.model(x))))
            except Exception:
                acc["EAC-softl1"].append(np.nan)
        for m in methods:
            out[m].append(np.nanmean(acc[m]) if any(np.isfinite(acc[m])) else np.nan)
    return out


def param_grid_parallel(fam, noises, sizes, seeds=3):
    """EAC param-recovery error over a (noise x size) grid, fanned via fit_many."""
    names = list(fam["true"])
    tv = np.array([fam["true"][k] for k in names])
    grid = np.full((len(noises), len(sizes)), np.nan)
    probs, coord = [], []
    for i, noise in enumerate(noises):
        for j, n in enumerate(sizes):
            for s in range(seeds):
                x, y, _ = _noisy(fam, n, noise, s)
                probs.append(FittingProblem(x=x, y=y, expr=fam["expr"],
                                        var=fam["var"], method="eac",
                                        kwargs={"p0": fam["p0"]}))
                coord.append((i, j))
    results = fit_many(probs, n_jobs=-1, backend="loky")
    bucket: dict[tuple, list] = {}
    for (i, j), r in zip(coord, results):
        if r.error is None and r.coeffs.size == len(names):
            err = float(np.mean(np.abs((r.coeffs - tv) / tv)) * 100)
            bucket.setdefault((i, j), []).append(err)
    for (i, j), errs in bucket.items():
        grid[i, j] = float(np.median(errs))
    return grid
