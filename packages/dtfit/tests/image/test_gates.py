"""Monte-Carlo gates on the estimator's statistical claims. CI runs the
small replicate counts; DTFIT_NIGHTLY=1 runs the full ones."""
import os

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import curve_fit, least_squares

from dtfit.image import Original, fit, order_for

from accuracy.scenarios import SCENARIOS_BY_NAME

NIGHTLY = bool(os.environ.get("DTFIT_NIGHTLY"))
REPS = 60 if NIGHTLY else 10
FAMILIES = [
    "exp_decay_offset", "logistic", "michaelis_menten", "gompertz",
    "gaussian", "damped_oscillation", "weibull_cdf", "hill",
    "lorentzian", "sine",
]


def _case(name):
    scn = SCENARIOS_BY_NAME[name]
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    names = sorted(str(s) for s in f.free_symbols if s != t)
    fn = sp.lambdify((t, *[sp.Symbol(n) for n in names]), f, "numpy")
    return m.expr, m.var, fn, np.array([scn.true[n] for n in names]), scn


@pytest.mark.parametrize("name", FAMILIES)
def test_efficiency_at_order_for(name):
    expr, var, fn, pt, scn = _case(name)
    x = np.linspace(scn.x0, scn.x1, scn.n)
    yc = fn(x, *pt)
    sig = 0.05 * np.std(yc)
    rng = np.random.default_rng(11)
    k = order_for(expr, pt, (scn.x0, scn.x1), var=var)
    e_img, e_cf = [], []
    for _ in range(REPS):
        y = yc + sig * rng.standard_normal(x.size)
        e_img.append(
            fit(expr, Original(x, y), var, order=k, p0=pt).coeffs - pt
        )
        e_cf.append(curve_fit(fn, x, y, p0=pt, maxfev=4000)[0] - pt)
    ratio = np.sqrt(np.mean(np.square(e_img), axis=0)) / (
        np.sqrt(np.mean(np.square(e_cf), axis=0)) + 1e-300
    )
    limit = 1.05 if NIGHTLY else 1.25
    assert np.mean(ratio) < limit, f"{name}: efficiency ratio {ratio}"


@pytest.mark.parametrize(
    "name", ["exp_decay_offset", "logistic", "michaelis_menten", "gompertz"]
)
def test_coverage_of_95_percent_intervals(name):
    expr, var, fn, pt, scn = _case(name)
    x = np.linspace(scn.x0, scn.x1, scn.n)
    yc = fn(x, *pt)
    sig = 0.05 * np.std(yc)
    rng = np.random.default_rng(12)
    reps = 200 if NIGHTLY else 60
    hits = []
    for _ in range(reps):
        y = yc + sig * rng.standard_normal(x.size)
        r = fit(expr, Original(x, y), var, order=12, p0=pt)
        hits.append(np.abs(r.coeffs - pt) <= 1.96 * np.sqrt(np.diag(r.cov)))
    cov = float(np.mean(hits))
    lo, hi = (0.92, 0.98) if NIGHTLY else (0.86, 0.995)
    assert lo <= cov <= hi, f"{name}: coverage {cov:.3f}"


@pytest.mark.parametrize(
    "name", ["exp_decay_offset", "logistic", "michaelis_menten", "gompertz"]
)
def test_robust_image_matches_scipy_soft_l1(name):
    expr, var, fn, pt, scn = _case(name)
    x = np.linspace(scn.x0, scn.x1, scn.n)
    yc = fn(x, *pt)
    sig = 0.05 * np.std(yc)
    rng = np.random.default_rng(13)
    e_img, e_sl = [], []
    for _ in range(REPS):
        y = yc + sig * rng.standard_normal(x.size)
        idx = rng.choice(x.size, x.size // 10, replace=False)
        y[idx] += 10 * sig * rng.choice([-1.0, 1.0], idx.size)
        e_img.append(
            fit(
                expr, Original(x, y), var, order=12, p0=pt, robust=True
            ).coeffs - pt
        )
        e_sl.append(
            least_squares(
                lambda p: fn(x, *p) - y, pt, loss="soft_l1", f_scale=sig
            ).x - pt
        )
    ratio = np.sqrt(np.mean(np.square(e_img), axis=0)) / (
        np.sqrt(np.mean(np.square(e_sl), axis=0)) + 1e-300
    )
    limit = 1.15 if NIGHTLY else 1.4
    assert np.mean(ratio) < limit, f"{name}: robust ratio {ratio}"
