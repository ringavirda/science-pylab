"""Model families and estimation code for the parameter-estimation experiment.

``parameter_estimation.ipynb`` imports this module and does the presentation.
The job it sets up: given the noisy response of a system whose parametric form
is known, recover the physical parameters as accurately as the NLLS gold
standard, where a black-box learner recovers none. What it provides:

* the model families, :data:`MODELS`, sixteen families nonlinear in their
  parameters across mechanics, electronics, spectroscopy, kinetics, biology,
  reliability and signal processing, with their per-family closure functions;
* the data generator :func:`gen`, driving the noise, outlier and sparse sweeps;
* the dtfit estimators :func:`est_lsi`, :func:`est_eac`, :func:`est_adaptive`,
  :func:`est_ensemble` and :func:`est_merged`, plus the joint multi-channel fit
  through :func:`dtfit_experimental.fit_joint`, each returning a
  ``{name: value}`` dict;
* the established baselines :func:`est_nlls` (SciPy ``curve_fit``) and
  :func:`est_robust_nlls` (soft-L1 ``least_squares``), and the no-parameter
  learners :func:`mlp_curve` and :func:`gp_curve`, re-exported from
  ``experiments.common.baselines``;
* the scoring and sweep helpers :func:`param_err`, :func:`safe` and a
  re-exported :func:`metrics`, the sweep drivers :func:`noise_sweep`,
  :func:`outlier_sweep` and :func:`learner_curve_fit`, the special-regime
  helpers :func:`regime_rows` and :func:`joint_channels`, and the real-data
  loader :func:`load_data`.

LSI coefficients come back in sympy's name-sorted order, so the estimators zip
them against ``sorted(names)``; the baselines keep the declared ``names``
order.
"""

from __future__ import annotations


import numpy as np

import dtfit as dt
from dtfit import fit_eac, ensemble_fit
from dtfit_experimental import fit_joint

from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl
from dtfit_experimental.experiments.common.baselines import (
    mlp_curve, gp_curve,
    prony_fit, matrix_pencil_fit, varpro_fit, moment_match_fit,
)

__all__ = [
    "MODELS", "FAMILY_REASON",
    "gen", "param_err", "safe", "metrics",
    "est_lsi", "est_eac", "est_adaptive", "est_ensemble", "est_merged",
    "est_nlls", "est_robust_nlls", "est_moment", "mlp_curve", "gp_curve",
    "prony_fit", "matrix_pencil_fit", "varpro_fit", "moment_match_fit",
    "A_METHODS", "DT_LABELS", "DT_DIAGNOSTIC_LABELS", "applicability_verdict",
    "family_recovery_row",
    "noise_sweep", "outlier_sweep", "learner_curve_fit",
    "regime_rows", "joint_channels", "subspace_rate_recovery", "load_data",
    "load_puromycin", "real_puromycin", "exp_model_mismatch",
    "f_expgrow",
]


def param_err(est, true):
    return float(np.mean([abs(est[k] - true[k]) / abs(true[k]) for k in true]) * 100)


# the model families, nonlinear in their parameters, each from a real domain
def _f_damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def _f_sine(t, A, c, p, w):
    return c + A * np.sin(w * t + p)


def _f_firstorder(t, K, tau):
    return K * (1 - np.exp(-t / tau))


def _f_biexp(t, a, b, c, d):
    return a * np.exp(-b * t) + c * np.exp(-d * t)


def _f_decay_offset(t, a, b, c):
    return c + a * np.exp(-b * t)


def _f_expgrow(t, a, b):
    return a * np.exp(b * t)


def _f_power(t, a, b):
    return a * (t + 1.0) ** b


def _f_stretched(t, A, q, tau):
    return A * np.exp(-(t / tau) ** q)


def _f_gauss(t, A, mu, s):
    return A * np.exp(-(t - mu) ** 2 / (2 * s ** 2))


def _f_lorentz(t, A, g, mu):
    return A / (1 + ((t - mu) / g) ** 2)


def _f_double_gauss(t, A1, A2, m1, m2, s1, s2):
    return (A1 * np.exp(-(t - m1) ** 2 / (2 * s1 ** 2))
            + A2 * np.exp(-(t - m2) ** 2 / (2 * s2 ** 2)))


def _f_logistic(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))


def _f_gompertz(t, A, b, c):
    return A * np.exp(-b * np.exp(-c * t))


def _f_weibull(t, K, k, lam):
    return K * (1 - np.exp(-(t / lam) ** k))


def _f_mm(t, Km, Vmax):
    return Vmax * t / (Km + t)


def _f_hill(t, K, Vmax, nh):
    return Vmax * t ** nh / (K ** nh + t ** nh)


# A public alias the notebook calls directly for its real-data exponential fit.
f_expgrow = _f_expgrow


# Each model carries: key, domain, sympy expr, names in any order (the
# estimators match LSI results to sympy's name-sorted layout), func with its
# signature in `names` order, the true params, the t-range, p0 and bounds in
# `names` order, and optionally ``osc``, naming the frequency parameter, which
# routes the fit through the oscillatory recipe: smoothing off, high order, an
# FFT frequency seed.
MODELS = [
    dict(key="damped", domain="mechanical / control", shape="oscillatory",
         expr="A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)", names=["A", "w", "z"],
         func=_f_damped, true={"A": 2.0, "w": 3.0, "z": 0.15}, t=(0, 6),
         p0=[1.0, 2.0, 0.1], bounds=[(0.1, 5), (1, 6), (0.01, 0.9)], osc="w"),
    dict(key="sine", domain="signal / vibration", shape="oscillatory",
         expr="c + A*sin(w*t + p)", names=["A", "c", "p", "w"], func=_f_sine,
         true={"A": 2.0, "c": 1.5, "p": 0.5, "w": 3.0}, t=(0, 6),
         p0=[2.0, 1.5, 0.0, 3.0],
         bounds=[(0.3, 5), (0, 4), (-np.pi, np.pi), (1, 6)], osc="w"),
    dict(key="firstorder", domain="electrical / RC", shape="saturating-exp",
         expr="K*(1-exp(-t/tau))", names=["K", "tau"], func=_f_firstorder,
         true={"K": 3.0, "tau": 1.2}, t=(0, 6), p0=[1.0, 1.0],
         bounds=[(0.1, 10), (0.05, 5)]),
    dict(key="biexp", domain="pharmacokinetics", shape="multi-exp",
         expr="a*exp(-b*t) + c*exp(-d*t)", names=["a", "b", "c", "d"],
         func=_f_biexp, true={"a": 2.0, "b": 2.0, "c": 1.0, "d": 0.3},
         t=(0, 6), p0=[1.5, 1.5, 1.0, 0.5],
         bounds=[(0.1, 5), (0.2, 5), (0.1, 5), (0.05, 2)]),
    dict(key="decay_offset", domain="thermal / sensor (Newton cooling)",
         shape="decay-to-baseline", expr="c + a*exp(-b*t)", names=["a", "b", "c"],
         func=_f_decay_offset, true={"a": 3.0, "b": 0.8, "c": 1.0}, t=(0, 8),
         p0=[2.0, 1.0, 0.5], bounds=[(0.5, 6), (0.1, 3), (0.0, 3)]),
    dict(key="expgrow", domain="growth / finance", shape="monotone",
         expr="a*exp(b*t)", names=["a", "b"], func=_f_expgrow,
         true={"a": 1.5, "b": 0.6}, t=(0, 4), p0=[1.0, 1.0],
         bounds=[(0.2, 5), (0.05, 2)]),
    dict(key="power", domain="physics / scaling law", shape="monotone",
         expr="a*(t+1)**b", names=["a", "b"], func=_f_power,
         true={"a": 2.0, "b": 1.4}, t=(0, 6), p0=[1.0, 1.0],
         bounds=[(0.2, 5), (0.3, 3)]),
    dict(key="stretched", domain="disordered relaxation (KWW)", shape="multi-exp",
         expr="A*exp(-(t/tau)**q)", names=["A", "q", "tau"], func=_f_stretched,
         true={"A": 3.0, "q": 1.6, "tau": 2.0}, t=(0.02, 8), p0=[2.0, 1.0, 1.5],
         bounds=[(0.5, 6), (0.3, 3), (0.3, 5)]),
    dict(key="gauss", domain="spectroscopy", shape="peak",
         expr="A*exp(-(t-mu)**2/(2*s**2))", names=["A", "mu", "s"],
         func=_f_gauss, true={"A": 4.0, "mu": 3.0, "s": 0.8}, t=(0, 6),
         p0=[2.0, 2.5, 1.0], bounds=[(0.5, 8), (1, 5), (0.2, 2)]),
    dict(key="lorentz", domain="spectroscopy (resonance)", shape="peak",
         expr="A/(1 + ((t-mu)/g)**2)", names=["A", "g", "mu"], func=_f_lorentz,
         true={"A": 4.0, "g": 0.7, "mu": 3.0}, t=(0, 6), p0=[2.0, 0.5, 2.5],
         bounds=[(0.5, 8), (0.2, 2), (1, 5)]),
    dict(key="double_gauss", domain="chromatography", shape="peak",
         expr="A1*exp(-(t-m1)**2/(2*s1**2)) + A2*exp(-(t-m2)**2/(2*s2**2))",
         names=["A1", "A2", "m1", "m2", "s1", "s2"], func=_f_double_gauss,
         true={"A1": 4.0, "A2": 2.5, "m1": 2.0, "m2": 4.0, "s1": 0.5, "s2": 0.7},
         t=(0, 6), p0=[3.0, 2.0, 1.5, 3.5, 0.6, 0.6],
         bounds=[(1, 8), (1, 6), (0.5, 3), (3, 5.5), (0.2, 1.5), (0.2, 1.5)]),
    dict(key="logistic", domain="epidemiology", shape="sigmoid",
         expr="K/(1+exp(-r*(t-t0)))", names=["K", "r", "t0"], func=_f_logistic,
         true={"K": 5.0, "r": 1.5, "t0": 3.0}, t=(0, 6), p0=[3.0, 1.0, 2.5],
         bounds=[(1, 10), (0.3, 4), (1, 5)]),
    dict(key="gompertz", domain="tumour / population growth", shape="sigmoid",
         expr="A*exp(-b*exp(-c*t))", names=["A", "b", "c"], func=_f_gompertz,
         true={"A": 5.0, "b": 3.0, "c": 0.8}, t=(0, 8), p0=[3.0, 2.0, 0.5],
         bounds=[(1, 10), (0.5, 8), (0.1, 3)]),
    dict(key="weibull", domain="reliability (failure CDF)", shape="sigmoid",
         expr="K*(1-exp(-(t/lam)**k))", names=["K", "k", "lam"], func=_f_weibull,
         true={"K": 4.0, "k": 2.2, "lam": 3.0}, t=(0.02, 8), p0=[3.0, 1.5, 2.0],
         bounds=[(1, 8), (0.5, 4), (0.5, 6)]),
    dict(key="mm", domain="enzyme kinetics", shape="rational-saturating",
         expr="Vmax*t/(Km+t)", names=["Km", "Vmax"], func=_f_mm,
         true={"Km": 1.5, "Vmax": 5.0}, t=(0.05, 8), p0=[1.0, 3.0],
         bounds=[(0.2, 5), (1, 10)]),
    dict(key="hill", domain="pharmacology (dose-response)",
         shape="rational-saturating",
         expr="Vmax*t**nh/(K**nh + t**nh)", names=["K", "Vmax", "nh"],
         func=_f_hill, true={"K": 2.0, "Vmax": 5.0, "nh": 2.5}, t=(0.02, 8),
         p0=[1.5, 3.0, 2.0], bounds=[(0.3, 5), (1, 10), (1, 5)]),
]


def gen(model, rng, *, n=220, noise=0.05, outliers=0.0, sparse=False):
    """Simulate one noisy response of ``model``: a clean curve over the
    family's t-range plus Gaussian noise scaled to the signal. ``outliers``
    injects that fraction of spikes at about 8 sigma; ``sparse`` keeps an
    irregular subset of the samples. Returns ``(t, y, clean)``."""
    t = np.linspace(model["t"][0], model["t"][1], n)
    clean = model["func"](t, *[model["true"][k] for k in model["names"]])
    scale = clean.std() + 1e-9
    y = clean + rng.normal(0, noise * scale, n)
    if outliers > 0:
        m = rng.random(n) < outliers
        y[m] += rng.normal(0, 8 * scale, int(m.sum()))
    if sparse:
        idx = np.sort(rng.choice(n, size=max(12, n // 8), replace=False))
        t, y, clean = t[idx], y[idx], clean[idx]
    return t, y, clean


def _scipy_bounds(b):
    """Pair list to scipy's ``(lo_list, hi_list)``, the shape the
    curve_fit-style baseline fitters want. dtfit's own fitters take the pair
    list directly, so do not route their bounds through this: a 2-parameter
    family's tuple is ambiguous and reads there as per-parameter pairs."""
    return ([x[0] for x in b], [x[1] for x in b]) if b else None


# The dtfit estimators, behind a uniform (label, est_dict) interface. LSI
# coeffs come back in sympy name-sorted order, so each zips against
# ``sorted(names)``.
def est_lsi(m, t, y):
    # Oscillatory families take the oscillatory recipe built into ``fit_lsi``:
    # passing ``freq_param`` turns smoothing off, seeds the frequency from an
    # FFT and raises the spectral order. ``m["osc"]`` names the angular
    # frequency.
    p0 = list(m["p0"]) if m.get("p0") else [(lo + hi) / 2 for lo, hi in m["bounds"]]
    r = dt.fit_lsi(t, y, m["expr"], "t", p0=p0, bounds=m["bounds"],
                   freq_param=m.get("osc") or None)
    return dict(zip(sorted(m["names"]), r.coeffs))


def est_eac(m, t, y, loss="linear"):
    p0 = list(m["p0"]) if m.get("p0") else None
    r = dt.fit_eac(t, y, m["expr"], "t", p0=p0, bounds=m["bounds"],
                   loss=loss)
    return dict(zip(sorted(m["names"]), r.coeffs))


def est_adaptive(m, t, y):
    p0 = list(m["p0"]) if m.get("p0") else None
    r = fit_eac(t, y, m["expr"], "t", p0=p0, window_mode="curvature")
    return dict(zip(sorted(m["names"]), r.coeffs))


def est_ensemble(m, t, y):
    r = ensemble_fit(t, y, m["expr"], "t", method="eac", n_windows=8,
                     overlap=0.5, aggregate="median", p0=m["p0"],
                     bounds=m["bounds"])
    return dict(zip(sorted(m["names"]), r.coeffs))


def est_merged(m, t, y):
    """The merged selector, delegated to :func:`dtfit.auto_estimate`: it routes
    an oscillatory family to the LSI oscillatory recipe through ``freq_param``
    and otherwise keeps whichever of LSI and EAC fits better in-sample. Part C
    exercises the dedicated regime variants #3, #4 and #6."""
    r = dt.auto_estimate(t, y, m["expr"], "t", shape="auto",
                         freq_param=m.get("osc") or None,
                         p0=m.get("p0"), bounds=m.get("bounds"))
    return dict(zip(sorted(m["names"]), r.coeffs))


def est_nlls(m, t, y):
    p = bl.scipy_curve_fit(t, y, m["func"], m["p0"], bounds=_scipy_bounds(m["bounds"]))
    return dict(zip(m["names"], p))


def est_robust_nlls(m, t, y):
    p = bl.robust_curve_fit(t, y, m["func"], m["p0"],
                            bounds=_scipy_bounds(m["bounds"]), loss="soft_l1")
    return dict(zip(m["names"], p))


def est_moment(m, t, y):
    """Method of moments / GMM on monomial integral moments: the unconditioned
    integral-projection baseline. It matches the model's integral moments to
    the data's exactly as EAC/LSI match areas and spectra, but its test
    functions are monomials rather than orthogonal, so it carries the
    Hilbert-matrix ill-conditioning that LSI's Legendre reconditioning removes.
    That makes it the fair foil, across every family, for what the
    reconditioning buys."""
    p = bl.moment_match_fit(t, y, m["func"], m["p0"],
                            bounds=_scipy_bounds(m["bounds"]))
    return dict(zip(m["names"], p))


def safe(fn, m, t, y):
    """Run estimator ``fn`` and return its mean relative parameter-recovery error
    in percent, or NaN if it raises: the uniform cell the tables and sweeps
    fill."""
    try:
        return param_err(fn(m, t, y), m["true"])
    except Exception:
        return np.nan


# Part A, recovery across families: methods and applicability verdict
A_METHODS = [("dtfit LSI", est_lsi), ("dtfit EAC", est_eac),
             ("dtfit adaptive-EAC (#6)", est_adaptive),
             ("dtfit merged", est_merged), ("SciPy NLLS (gold)", est_nlls),
             # The unconditioned integral-moment ancestor of LSI, in as the
             # honest foil for what the Legendre reconditioning buys. It is
             # expected to trail on the higher-parameter families.
             ("Method of moments", est_moment)]
# The per-method dtfit diagnostic breakdown: the estimators whose minimum is
# the honest best explicit dtfit method. ``est_merged`` is a selector over
# these two of them, keeping whichever of LSI and EAC fits better in-sample, so
# pooling it into the same minimum would double-count its own inputs and
# flatter the "best dtfit" number, a min-over-4 containing the min-over-2 of
# two of the four. ``merged`` is reported separately instead, as the single
# deployable answer; see :func:`family_recovery_row`.
DT_DIAGNOSTIC_LABELS = ["dtfit LSI", "dtfit EAC", "dtfit adaptive-EAC (#6)"]
# Back-compat alias: the min-pool for "best dtfit".
DT_LABELS = DT_DIAGNOSTIC_LABELS


def applicability_verdict(best_dt_err, nlls_err):
    """Categorical verdict comparing the best dtfit error against NLLS.

    The tie band is deliberately tight, 1.15x the NLLS error, so that a tie
    means genuinely comparable rather than merely within half again. A dtfit
    error between 1.15x and 2x NLLS is reported as the mild loss it is, and
    beyond 2x is a clear NLLS win. :func:`family_recovery_row` surfaces the
    quantitative version, the ratio ``dtfit_err/nlls_err``, alongside this.
    """
    if nlls_err <= 0:
        return "dtfit ties/beats NLLS" if best_dt_err <= 1e-9 else "NLLS wins"
    ratio = best_dt_err / nlls_err
    if ratio <= 1.15:
        return "dtfit ties/beats NLLS"
    if ratio <= 2.0:
        return "NLLS slightly better"
    return "NLLS wins"


def family_recovery_row(err_row, method_names):
    """Per-family recovery summary from one row of the ``err`` matrix, aligned
    to ``method_names``, the labels of :data:`A_METHODS`.

    The returned dict reports each explicit dtfit method's error, so a reader
    sees which single method wins rather than only the minimum over them; the
    ``merged`` selector's error separately, as the single deployable answer and
    not pooled into the diagnostic minimum (see :data:`DT_DIAGNOSTIC_LABELS`);
    and the numeric ``dtfit/NLLS`` ratio beside the categorical verdict.
    NaN-safe.
    """
    idx = {nm: j for j, nm in enumerate(method_names)}

    def _get(label):
        j = idx.get(label)
        return float(err_row[j]) if j is not None else float("nan")

    per_method = {lbl: _get(lbl) for lbl in DT_DIAGNOSTIC_LABELS}
    diag_vals = [v for v in per_method.values() if np.isfinite(v)]
    best_dt = float(np.min(diag_vals)) if diag_vals else float("nan")
    best_label = (min(per_method, key=lambda k: (np.inf if not np.isfinite(per_method[k])
                                                 else per_method[k]))
                  if diag_vals else None)
    merged = _get("dtfit merged")
    nlls = _get("SciPy NLLS (gold)")
    ratio = (best_dt / nlls) if (np.isfinite(best_dt) and np.isfinite(nlls)
                                 and nlls > 0) else float("nan")
    row = {
        "best dtfit method": best_label,
        "best dtfit err %": best_dt,
        "merged (deployable) err %": merged,
        "NLLS err %": nlls,
        "dtfit/NLLS ratio": ratio,
        "verdict": applicability_verdict(best_dt, nlls),
    }
    for lbl in DT_DIAGNOSTIC_LABELS:
        row[f"{lbl} err %"] = per_method[lbl]
    return row


# Part B: the robustness sweeps and the no-parameter learners
def noise_sweep(model, fn, noise_levels, *, n=240, seeds=3):
    """Mean parameter-recovery error of estimator ``fn`` on ``model`` across the
    given Gaussian-noise levels (averaged over ``seeds``). Returns a list aligned
    with ``noise_levels``."""
    curve = []
    for nz in noise_levels:
        errs = [safe(fn, model, *gen(model, np.random.default_rng(s), n=n,
                                     noise=nz)[:2]) for s in range(seeds)]
        curve.append(float(np.nanmean(errs)))
    return curve


def outlier_sweep(model, fn, fracs, *, n=300, noise=0.05, seeds=3):
    """Mean parameter-recovery error across outlier fractions (averaged over
    ``seeds``). Returns a list aligned with ``fracs``."""
    curve = []
    for fr in fracs:
        errs = [safe(fn, model, *gen(model, np.random.default_rng(100 + s), n=n,
                                     noise=noise, outliers=fr)[:2])
                for s in range(seeds)]
        curve.append(float(np.nanmean(errs)))
    return curve


def learner_curve_fit(model, rng, *, n=300, noise=0.30):
    """Compare curve accuracy, not parameter accuracy, at one heavy-noise
    condition: dtfit EAC and SciPy NLLS, which recover parameters, against the
    black-box sklearn MLP and Gaussian process, which recover none. A missing
    sklearn skips the learner and leaves its row carrying NaN scores. Returns
    ``(t, y, clean, rows)``, each row being ``{"method", "R2", "RMSE"}``."""
    t, y, clean = gen(model, rng, n=n, noise=noise)
    rows = []

    def _pred_params(est_fn):
        est = est_fn(model, t, y)
        return model["func"](t, *[est[k] for k in model["names"]])

    def _pred_learner(curve_fn):
        try:
            return curve_fn(t, y, t)
        except Exception:
            return None

    for label, pred in [
            ("dtfit EAC", _pred_params(est_eac)),
            ("SciPy NLLS", _pred_params(est_nlls)),
            ("sklearn MLP (no params)", _pred_learner(mlp_curve)),
            ("Gaussian process (no params)", _pred_learner(gp_curve))]:
        if pred is None:
            rows.append({"method": label, "R2": np.nan, "RMSE": np.nan})
        else:
            sc = metrics(clean, pred)
            rows.append({"method": label, "R2": sc["R2"], "RMSE": sc["RMSE"]})
    return t, y, clean, rows


# Part C: the special regimes the merged selector routes
def _model(key):
    return next(mm for mm in MODELS if mm["key"] == key)


def regime_rows(rng):
    """The C1-C3 single-channel regimes: a concentrated transient, sparse
    sampling and a short record. Returns a list of
    ``{"regime", "adaptive_EAC", "EAC", "NLLS", "note"}`` rows, the errors in
    percent."""
    rows = []

    # C1, a concentrated transient: fast rise, long flat tail
    fo = _model("firstorder")
    fo_t = dict(fo, t=(0, 8), true={"K": 3.0, "tau": 0.4})
    t, y, _ = gen(fo_t, rng, n=400, noise=0.04)
    rows.append({"regime": "concentrated transient (fast tau, long tail)",
                 "adaptive_EAC": safe(est_adaptive, fo_t, t, y),
                 "EAC": safe(est_eac, fo_t, t, y),
                 "NLLS": safe(est_nlls, fo_t, t, y),
                 "note": "adaptive-EAC (#6) -- curvature windows on the transient"})

    # C2, sparse and irregular sampling
    dm = _model("damped")
    t, y, _ = gen(dm, rng, n=300, noise=0.05, sparse=True)
    rows.append({"regime": f"sparse sampling ({t.size} pts)",
                 "adaptive_EAC": safe(est_adaptive, dm, t, y),
                 "EAC": safe(est_eac, dm, t, y),
                 "NLLS": safe(est_nlls, dm, t, y),
                 "note": "EAC -- area criterion tolerant of irregular spacing"})

    # C3, a short record of very few points
    gm = _model("gauss")
    t, y, _ = gen(gm, rng, n=18, noise=0.05)
    rows.append({"regime": "short record (18 pts, gaussian)",
                 "adaptive_EAC": safe(est_adaptive, gm, t, y),
                 "EAC": safe(est_eac, gm, t, y),
                 "NLLS": safe(est_nlls, gm, t, y),
                 "note": "all comparable -- few points, no clear edge"})
    return rows


def joint_channels(rng):
    """C4, the multi-channel shared parameter that the joint fit (#4) exists
    for: one decay rate shared across short, noisy channels, none of which
    constrains tau well alone. Returns
    ``{"joint_err", "indep_err", "indep_scatter"}``, the errors in percent."""
    tau_true, ks = 1.2, [3.0, 2.0, 4.0, 2.5]
    tt = np.linspace(0, 6, 30)
    chans = [(tt, K * (1 - np.exp(-tt / tau_true)) + rng.normal(0, 0.18 * K, tt.size))
             for K in ks]
    j = fit_joint(chans, "K*(1-exp(-t/tau))", "t", shared=["tau"], n_windows=5,
                  p0_shared=[1.0], p0_private=[1.0])
    joint_err = abs(j.shared["tau"] - tau_true) / tau_true * 100
    indep = []
    for (tx, yx) in chans:
        try:
            r = dt.fit_eac(tx, yx, "K*(1-exp(-t/tau))", "t", p0=[1.0, 1.0],
                           bounds=[(0.1, 10), (0.05, 5)])
            indep.append(float(r.coeffs[1]))     # sorted names [K, tau] -> tau
        except Exception:
            pass
    indep_err = float(np.mean([abs(v - tau_true) / tau_true * 100 for v in indep])) \
        if indep else float("nan")
    indep_scatter = float(np.std(indep)) if indep else float("nan")
    return {"joint_err": float(joint_err), "indep_err": indep_err,
            "indep_scatter": indep_scatter}


# Part C2: the head-to-head against the Western signal-parameter lineage,
# Prony, Matrix Pencil and ESPRIT, which is the comparison a signal-processing
# reviewer outside the Pukhov school asks for. Scoped to the two textbook,
# offset-clean tasks where a subspace mode maps unambiguously onto one physical
# quantity.
def _dominant_mode_rate(model) -> float:
    """Real part, the growth or decay rate, of the highest-amplitude recovered
    mode."""
    i = int(np.argmax(np.abs(model.amp)))
    return float(model.rate[i].real)


def _dominant_mode_frequency(model) -> float:
    """Angular frequency ``|Im(rate)|`` of the highest-amplitude oscillatory mode."""
    osc = model.frequency > 1e-9
    if not np.any(osc):
        return 0.0
    amps = np.where(osc, np.abs(model.amp), -np.inf)
    return float(model.frequency[int(np.argmax(amps))])


def subspace_rate_recovery(rng, *, n=400, noise=0.03):
    """dtfit against the Western signal-parameter lineage on the tasks the
    subspace methods were built for.

    Compares dtfit LSI, gold-standard SciPy NLLS, classical Prony and the
    SVD-robust Matrix Pencil / ESPRIT on recovering:

    * a single exponential's growth rate ``b`` (``expgrow``);
    * a sinusoid's angular frequency ``w`` (``sine``, mean-removed so the
      subspace sees one clean conjugate pair).

    These two families are where a Prony-family mode maps onto exactly one
    physical quantity. Only there is the head-to-head apples for apples.
    Returns a list of ``{"task", "quantity", "true", "<method>": err% ...}``
    rows.
    """
    def err(est_val, true_val):
        return float(abs(est_val - true_val) / abs(true_val) * 100)

    rows = []

    eg = _model("expgrow")
    t, y, _ = gen(eg, rng, n=n, noise=noise)
    b_true = eg["true"]["b"]
    rows.append({
        "task": "exp growth rate (a*exp(b*t))", "quantity": "b", "true": b_true,
        "dtfit LSI": err(est_lsi(eg, t, y)["b"], b_true),
        "SciPy NLLS": err(est_nlls(eg, t, y)["b"], b_true),
        "Prony": err(_dominant_mode_rate(prony_fit(t, y, 1)), b_true),
        "Matrix Pencil/ESPRIT":
            err(_dominant_mode_rate(matrix_pencil_fit(t, y, 1)), b_true),
    })

    si = _model("sine")
    t, y, _ = gen(si, rng, n=n, noise=noise)
    w_true = si["true"]["w"]
    yc = y - float(np.mean(y))
    rows.append({
        "task": "sinusoid frequency (A*sin(w*t+p))", "quantity": "w",
        "true": w_true,
        "dtfit LSI": err(est_lsi(si, t, y)["w"], w_true),
        "SciPy NLLS": err(est_nlls(si, t, y)["w"], w_true),
        "Prony": err(_dominant_mode_frequency(prony_fit(t, yc, 2)), w_true),
        "Matrix Pencil/ESPRIT":
            err(_dominant_mode_frequency(matrix_pencil_fit(t, yc, 2)), w_true),
    })
    return rows


# Part D: real-data recovery, where there is no ground truth, so validity has
# to come from methods agreeing with each other and fitting well
def load_data(name, col=1):
    """Load one column of a bundled CSV under ``experiments/data``."""
    import csv
    rows = list(csv.reader((EXPERIMENTS_DIR / "data" / name).open()))[1:]
    return np.array([float(r[col]) for r in rows])


def load_puromycin():
    """Load the canonical Puromycin enzyme-kinetics dataset (Bates & Watts
    1988, the treated group), bundled as ``experiments/data/puromycin.csv``.

    Twelve measurements, six substrate concentrations in replicate, of an
    enzyme's initial reaction velocity with and without puromycin. It is a
    textbook Michaelis-Menten dataset, real, sparse and replicated, and native
    to the ``mm`` and ``hill`` families, where the COVID and FX growth-rate
    curves of D1/D2 only exercise the exponential form. Returns
    ``(conc, velocity)`` as float arrays, from a ``conc,velocity`` header."""
    import csv
    rows = list(csv.reader((EXPERIMENTS_DIR / "data" / "puromycin.csv").open()))[1:]
    conc = np.array([float(r[0]) for r in rows])
    velocity = np.array([float(r[1]) for r in rows])
    return conc, velocity


def real_puromycin():
    """Fit the Michaelis-Menten law ``Vmax*t/(Km+t)`` (the ``mm`` family) to the
    real Puromycin data with dtfit LSI, dtfit EAC and SciPy NLLS.

    Real data carries no ground-truth parameter vector, so validity has to show
    up as the methods agreeing on ``{Vmax, Km}`` and fitting well. The reported
    ``RMSE`` is the in-sample curve residual in velocity units and ``R2`` its
    normalised twin. The wide bounds and seed below suit the actual scale of
    this data, concentration in [0.02, 1.1] mM and velocity up to about 210,
    where the synthetic ``mm`` entry in :data:`MODELS` does not. Returns a list
    of ``{"method", "Vmax", "Km", "RMSE", "R2"}`` rows."""
    conc, velocity = load_puromycin()
    # A Puromycin-scaled copy of the ``mm`` family: the same expr, func and
    # name layout, on a seed and bounds this data's scale supports, with Vmax
    # near the maximum velocity and Km near a mid-range concentration.
    mmr = dict(_model("mm"),
               true={"Km": 0.1, "Vmax": 210.0}, t=(conc.min(), conc.max()),
               p0=[0.1, 200.0], bounds=[(1e-3, 5.0), (50.0, 500.0)])
    rows = []
    for label, fn in [("dtfit LSI", est_lsi), ("dtfit EAC", est_eac),
                      ("SciPy NLLS", est_nlls)]:
        try:
            est = fn(mmr, conc, velocity)
            pred = _f_mm(conc, est["Km"], est["Vmax"])
            sc = metrics(velocity, pred)
            rows.append({"method": label, "Vmax": float(est["Vmax"]),
                         "Km": float(est["Km"]), "RMSE": sc["RMSE"], "R2": sc["R2"]})
        except Exception:
            rows.append({"method": label, "Vmax": float("nan"), "Km": float("nan"),
                         "RMSE": float("nan"), "R2": float("nan")})
    return rows


# Part E, the model-mismatch negative control. Fit the wrong structural model
# to data from a known generator, and two things follow: no estimator rescues a
# mis-specified model, both dtfit and NLLS degrading sharply, and an in-sample
# goodness statistic (R2) flags the mismatch, which is what a blind selector
# keys on.
def _fit_rmse_r2(model, fn, t, y):
    """Fit ``model`` to ``(t, y)`` with estimator ``fn`` and return the in-sample
    curve ``(RMSE, R2)``, NaN on failure. Curve accuracy rather than parameter
    accuracy: it is the only comparable currency once the fitted model has the
    wrong structure."""
    try:
        est = fn(model, t, y)
        pred = model["func"](t, *[est[k] for k in model["names"]])
        sc = metrics(y, pred)
        return float(sc["RMSE"]), float(sc["R2"])
    except Exception:
        return float("nan"), float("nan")


# The negative-control cases: data generated from ``true``, fitted with both
# the correct structure and a structurally wrong one. Each wrong model must
# accept the same 1-D ``t`` support as the truth. ``param`` names a physical
# quantity present and comparable in both models, for a parameter-error column,
# or None where no such quantity exists.
_MISMATCH_CASES = [
    dict(name="biexp truth, fitted as single-exp decay",
         true="biexp", correct="biexp", wrong="decay_offset", param=None,
         note="two decay rates collapsed onto one -- the fast/slow split is lost"),
    dict(name="damped truth, fitted as exp growth",
         true="damped", correct="damped", wrong="expgrow", param=None,
         note="an oscillation forced onto a monotone exponential -- no cycle"),
    dict(name="logistic truth, fitted as exp growth",
         true="logistic", correct="logistic", wrong="expgrow", param=None,
         note="a saturating sigmoid forced onto unbounded growth -- no plateau"),
]


def exp_model_mismatch(seeds=5):
    """The model-mismatch negative control: fit the wrong structural model and
    show that no estimator rescues it.

    For each case in :data:`_MISMATCH_CASES`, the same data from a known
    generator is fitted with the correct structure and with a structurally
    wrong one, by both dtfit-LSI and SciPy NLLS. It reports each estimator's
    in-sample curve ``RMSE`` and ``R2`` for correct against wrong, and the R2
    gap, correct minus wrong, that a blind selector would flag the mismatch by.
    Averaged over ``seeds``. Returns a list of one dict per case::

        {"case", "correct model", "wrong model",
         "LSI correct RMSE", "LSI wrong RMSE",
         "NLLS correct RMSE", "NLLS wrong RMSE",
         "wrong/correct RMSE ratio",              # >> 1 == mismatch (worst est)
         "correct R2", "wrong R2", "R2 gap (flags mismatch)", "note"}
    """
    rows = []
    for case in _MISMATCH_CASES:
        true_m = _model(case["true"])
        correct_m = _model(case["correct"])
        wrong_m = _model(case["wrong"])
        acc: dict[str, list[float]] = {k: [] for k in (
            "lsi_c_rmse", "lsi_w_rmse", "nlls_c_rmse", "nlls_w_rmse",
            "lsi_c_r2", "lsi_w_r2", "nlls_c_r2", "nlls_w_r2")}
        for s in range(seeds):
            t, y, _ = gen(true_m, np.random.default_rng(1000 + s), n=200, noise=0.05)
            lc_rmse, lc_r2 = _fit_rmse_r2(correct_m, est_lsi, t, y)
            lw_rmse, lw_r2 = _fit_rmse_r2(wrong_m, est_lsi, t, y)
            nc_rmse, nc_r2 = _fit_rmse_r2(correct_m, est_nlls, t, y)
            nw_rmse, nw_r2 = _fit_rmse_r2(wrong_m, est_nlls, t, y)
            acc["lsi_c_rmse"].append(lc_rmse); acc["lsi_w_rmse"].append(lw_rmse)
            acc["nlls_c_rmse"].append(nc_rmse); acc["nlls_w_rmse"].append(nw_rmse)
            acc["lsi_c_r2"].append(lc_r2); acc["lsi_w_r2"].append(lw_r2)
            acc["nlls_c_r2"].append(nc_r2); acc["nlls_w_r2"].append(nw_r2)

        def _m(key):
            return float(np.nanmean(acc[key]))

        # Take the correct and wrong RMSE from whichever estimator fits the
        # wrong model best: the honest worst case for a claim that a good
        # estimator rescues a mis-specified model.
        wrong_rmse = min(_m("lsi_w_rmse"), _m("nlls_w_rmse"))
        correct_rmse = min(_m("lsi_c_rmse"), _m("nlls_c_rmse"))
        ratio = (wrong_rmse / correct_rmse
                 if correct_rmse > 0 else float("inf"))
        correct_r2 = max(_m("lsi_c_r2"), _m("nlls_c_r2"))
        wrong_r2 = max(_m("lsi_w_r2"), _m("nlls_w_r2"))
        rows.append({
            "case": case["name"],
            "correct model": case["correct"],
            "wrong model": case["wrong"],
            "LSI correct RMSE": _m("lsi_c_rmse"),
            "LSI wrong RMSE": _m("lsi_w_rmse"),
            "NLLS correct RMSE": _m("nlls_c_rmse"),
            "NLLS wrong RMSE": _m("nlls_w_rmse"),
            "wrong/correct RMSE ratio": ratio,
            "correct R2": correct_r2,
            "wrong R2": wrong_r2,
            "R2 gap (flags mismatch)": correct_r2 - wrong_r2,
            "note": case["note"],
        })
    return rows


# The applicability map, per family: the best dtfit method and the reasoning.
# It lives here so the notebook can render it as a table.
FAMILY_REASON = {
    "damped": ("EAC / LSI",
               "Oscillation -- the frequency lives in the spectrum/area; fitted "
               "with smoothing off, high order and an FFT seed (the forecasting "
               "recipe), it ties NLLS."),
    "sine": ("LSI",
             "Pure harmonic -- LSI's home turf once the cycle is not smoothed "
             "away; a default-smoothed low-order fit gives ~50% error, the osc "
             "recipe gives <1%."),
    "firstorder": ("EAC / LSI",
                   "A smooth saturating-exponential bulk; the area criterion "
                   "pins K and tau; ties NLLS."),
    "biexp": ("EAC",
              "Two decay rates read from the integrated curve; ties NLLS (the "
              "rate pair is mildly ill-conditioned for everyone)."),
    "decay_offset": ("LSI / EAC",
                     "Exponential decay to a non-zero baseline (Newton cooling / "
                     "RC discharge to a floor); a smooth bulk shape -- the rate and "
                     "the offset come straight out of the integral; ties NLLS."),
    "expgrow": ("LSI / EAC",
                "A monotone bulk shape; the rate sets the whole spectrum; ties "
                "NLLS."),
    "power": ("LSI",
              "A monotone scaling law; the exponent shapes the bulk; ties NLLS."),
    "stretched": ("LSI",
                  "KWW relaxation; LSI recovers it moderately -- the stretch "
                  "exponent beta trades off with tau for every method, so error is "
                  "larger than a plain exponential."),
    "gauss": ("EAC / adaptive-EAC (#6)",
              "A single peak -- the area / curvature criteria concentrate on the "
              "bend where mu and sigma are determined; ties NLLS."),
    "lorentz": ("EAC",
                "A heavy-tailed resonance -- the one family where NLLS keeps a "
                "slight edge: the tails dominate any global integral, so the width "
                "gamma is a touch harder for the area criterion. Even so dtfit is "
                "within ~0.1% of NLLS (both well under 0.5%)."),
    "double_gauss": ("EAC / adaptive-EAC (#6)",
                     "Two overlapping peaks: the **area / curvature** criteria "
                     "separate the components and tie NLLS, but the **LSI "
                     "spectrum** struggles (overlapping peaks blur the spectral "
                     "signature, ~2-3% error) -- use EAC, not LSI, for multi-peak "
                     "shapes."),
    "logistic": ("LSI / EAC",
                 "Sigmoid -- the inflection shapes the integral; ties NLLS."),
    "gompertz": ("EAC / LSI",
                 "Asymmetric sigmoid (growth); the bulk determines all three "
                 "parameters; ties NLLS."),
    "weibull": ("LSI",
                "Reliability CDF (sigmoid); ties NLLS (slightly looser than the "
                "logistic -- the shape exponent k and scale lambda partly trade "
                "off)."),
    "mm": ("EAC / adaptive-EAC (#6)",
           "Rational saturation. **The old report's 151% 'Michaelis-Menten "
           "exception' was a parameter-ordering bug** (the spectral coefficients "
           "were zipped to the names in the wrong order); with the order fixed the "
           "rational saturation is recovered to ~0.3% -- adaptive/curvature windows "
           "put resolution on the early rise where Km is set. It is *not* a "
           "boundary family."),
    "hill": ("adaptive-EAC (#6) / LSI",
             "Rational saturation with a cooperativity exponent; the curvature "
             "windows concentrate on the rise that sets K and nh -- ties NLLS "
             "(~0.3%), not a failure."),
}
