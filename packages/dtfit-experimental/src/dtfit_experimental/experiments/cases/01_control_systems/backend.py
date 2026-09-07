"""Simulation and estimation code for the control-systems case study.

``01_control_systems.ipynb`` imports this module and owns the presentation.

The task is the control engineer's: recover a plant's physical parameters from
its noisy response. Two textbook scenarios stand in for the general case, an
underdamped second-order free response and a first-order step. Each is
identified by EAC and LSI against SciPy ``curve_fit``, the NLLS gold standard
for exactly this job, and against an sklearn MLP, which fits the curve well but
recovers no physical parameters at all. That contrast is the point of the
comparison.

Two further benches exercise the adaptations rather than the base estimators.
:func:`regime_change` runs an :class:`EACFilter` across a mid-run damping jump
to see whether the online filter re-adapts and flags the break;
:func:`mimo_joint` identifies a three-output plant whose channels share one
natural frequency, jointly and then per channel for contrast.

The SciPy and sklearn baselines are imported lazily in
:mod:`...common.baselines`, and the ``with_scipy`` / ``with_mlp`` flags let the
notebook drop a row whose dependency is missing.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit_experimental import fit_joint
from dtfit.streaming import EACFilter

from dtfit_experimental.experiments.common import metrics, timed
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "DAMP_EXPR", "FO_EXPR",
    "scenario_damped", "scenario_first_order", "param_err",
    "damped_table", "first_order_table", "regime_change", "mimo_joint",
]

# Underdamped free response; dtfit sorts the params, giving order A, w, z.
DAMP_EXPR = "A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)"
# First-order step; param order K, tau.
FO_EXPR = "K*(1-exp(-t/tau))"


def _damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def scenario_damped(rng, n=240, noise=0.05):
    A, z, w = 2.0, 0.15, 3.0
    t = np.linspace(0, 6, n)
    clean = _damped(t, A, w, z)
    y = clean + rng.normal(0, noise * clean.std(), n)
    return t, y, clean, {"A": A, "w": w, "z": z}


def scenario_first_order(rng, n=200, noise=0.03):
    K, tau = 3.0, 1.2
    t = np.linspace(0, 6, n)
    clean = K * (1 - np.exp(-t / tau))
    y = clean + rng.normal(0, noise * clean.std(), n)
    return t, y, clean, {"K": K, "tau": tau}


def param_err(est: dict, true: dict) -> float:
    """Mean relative parameter-recovery error (%)."""
    return float(np.mean([abs(est[k] - true[k]) / abs(true[k]) for k in true]) * 100)


def damped_table(t, y, clean, true, *, with_scipy=True, with_mlp=True):
    """Identify the damped second-order response with EAC, LSI and baselines.

    Returns ``(rows, preds)``. Each row carries method / param err % / R2 /
    RMSE / fit (ms); the MLP's param error is ``None`` because it recovers no
    physical parameters. ``preds`` maps each method name to its fitted curve.
    Pass ``with_scipy=False`` or ``with_mlp=False`` to drop a baseline whose
    optional dependency is missing.
    """
    names = ["A", "w", "z"]
    p0, lo, hi = [1.0, 2.0, 0.1], [0.1, 1.0, 0.01], [5, 6, 0.9]
    rows, preds = [], {}

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        rows.append({"method": label,
                     "param err %": param_err(est, true) if est else None,
                     "R2": m["R2"], "RMSE": m["RMSE"], "fit (ms)": ms})
        preds[label] = pred

    r, ms = timed(lambda: dt.fit_eac(t, y, DAMP_EXPR, "t", p0=p0, bounds=(lo, hi)))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, DAMP_EXPR, "t", p0=p0,
                                     bounds=list(zip(lo, hi))))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)
    if with_scipy:
        p, ms = timed(lambda: bl.scipy_curve_fit(t, y, _damped, p0, bounds=(lo, hi)))
        add("SciPy curve_fit", p, _damped(t, *p), ms)
    if with_mlp:
        yhat, ms = timed(lambda: bl.mlp_curve(t, y, t, hidden=(64, 64)))
        add("sklearn MLP", None, yhat, ms)
    return rows, preds


def first_order_table(t, y, clean, true, *, with_scipy=True, with_mlp=True):
    """Identify the first-order step response (DC gain ``K``, time constant
    ``tau``). Same return shape and skip flags as :func:`damped_table`."""
    names = ["K", "tau"]
    rows, preds = [], {}

    def add(label, coeffs, pred, ms):
        est = dict(zip(names, coeffs)) if coeffs is not None else None
        m = metrics(clean, pred)
        rows.append({"method": label,
                     "param err %": param_err(est, true) if est else None,
                     "R2": m["R2"], "RMSE": m["RMSE"], "fit (ms)": ms})
        preds[label] = pred

    r, ms = timed(lambda: dt.fit_eac(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("EAC", r.coeffs, np.asarray(r.model(t)), ms)
    r, ms = timed(lambda: dt.fit_lsi(t, y, FO_EXPR, "t", p0=[1.0, 1.0]))
    add("LSI", r.coeffs, np.asarray(r.model(t)), ms)

    def f(tt, K, tau):
        return K * (1 - np.exp(-tt / tau))

    if with_scipy:
        p, ms = timed(lambda: bl.scipy_curve_fit(t, y, f, [1.0, 1.0]))
        add("SciPy curve_fit", p, f(t, *p), ms)
    if with_mlp:
        yhat, ms = timed(lambda: bl.mlp_curve(t, y, t))
        add("sklearn MLP", None, yhat, ms)
    return rows, preds


def regime_change(rng, n=900):
    """Damping z jumps mid-run; the online filter should re-adapt and flag it.

    Returns ``(t, y, clean, track, z_hist, drift_idx, half)``: the time grid,
    the noisy response, the clean signal, the filter's online track, its
    tracked damping estimate, the sample indices it flagged as structural
    breaks, and the true change index.
    """
    t = np.linspace(0, 18, n)
    half = n // 2
    z1, z2, A, w = 0.08, 0.30, 2.0, 2.5
    z_arr = np.where(np.arange(n) < half, z1, z2)
    wd = w * np.sqrt(1 - z_arr ** 2)
    # phase-continuous across the change: integrate the (piecewise) frequency
    dtt = np.diff(t, prepend=t[0])
    phase = np.cumsum(wd * dtt)
    clean = A * np.exp(-z_arr * w * t) * np.sin(phase)
    y = clean + rng.normal(0, 0.05, n)
    flt = EACFilter(DAMP_EXPR, "t", p0=[2.0, 2.5, 0.1],
                    window_size=60, q_diag=[1e-3, 1e-3, 1e-3],
                    order=3)
    track, z_hist, drift_idx = [], [], []
    for i in range(n):
        flt.partial_fit(t[i], y[i])
        if flt.drift_flag_:
            drift_idx.append(i)
        track.append(float(flt.predict(np.array([t[i]]))[0]) if len(flt._t) else np.nan)
        z_hist.append(flt.params_["z"])
    return t, y, clean, np.array(track), np.array(z_hist), drift_idx, half


def mimo_joint(rng, n=200):
    """3-output plant sharing a natural frequency w; identify jointly.

    Returns ``(w_true, amps, j, indep_w, chans, t)``: the true shared
    frequency, the per-channel amplitudes, the :func:`fit_joint` result ``j``,
    the per-channel independent-EAC frequency estimates, the channels, and the
    time grid.
    """
    t = np.linspace(0, 6, n)
    w_true, z_true = 3.0, 0.12
    amps = [1.0, 2.0, 3.0]
    chans = [(t, _damped(t, A, w_true, z_true) + rng.normal(0, 0.04, n))
             for A in amps]
    j = fit_joint(chans, DAMP_EXPR, "t", shared=["w"], n_windows=6,
                  p0_shared=[2.5], p0_private=[1.0, 0.1])
    # independent per-channel EAC for contrast
    indep_w = []
    for (tx, yx) in chans:
        r = dt.fit_eac(tx, yx, DAMP_EXPR, "t", p0=[1.0, 2.5, 0.1],
                       bounds=([0.1, 1, 0.01], [5, 6, 0.9]))
        indep_w.append(dict(zip(["A", "w", "z"], r.coeffs))["w"])
    return w_true, amps, j, indep_w, chans, t
