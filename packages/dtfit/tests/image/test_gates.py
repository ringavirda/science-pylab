"""Monte-Carlo gates on the estimator's statistical claims. CI runs the
small replicate counts; DTFIT_NIGHTLY=1 runs the full ones."""
import os
import warnings

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import curve_fit, least_squares
from scipy.stats import laplace, t as t_dist

from dtfit.image import Original, fit, order_for

from accuracy.scenarios import SCENARIOS_BY_NAME

NIGHTLY = os.environ.get("DTFIT_NIGHTLY", "") not in ("", "0")
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
    assert 0.5 < np.mean(ratio) < limit, f"{name}: efficiency ratio {ratio}"


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
    order = 20 if name == "logistic" else 12
    hits = []
    for _ in range(reps):
        y = yc + sig * rng.standard_normal(x.size)
        r = fit(expr, Original(x, y), var, order=order, p0=pt)
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
    order = 20 if name == "logistic" else 12
    e_img, e_sl = [], []
    for _ in range(REPS):
        y = yc + sig * rng.standard_normal(x.size)
        idx = rng.choice(x.size, x.size // 10, replace=False)
        y[idx] += 10 * sig * rng.choice([-1.0, 1.0], idx.size)
        e_img.append(
            fit(
                expr, Original(x, y), var, order=order, p0=pt, robust=True
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
    assert 0.3 < np.mean(ratio) < limit, f"{name}: robust ratio {ratio}"


NOISE_FAMILIES = [
    "exp_decay_offset", "logistic", "michaelis_menten", "gompertz",
    "gaussian", "damped_oscillation",
]


@pytest.mark.parametrize("name", NOISE_FAMILIES)
def test_noise_sigma_recovers_the_noise_level(name):
    """The tail-order noise estimate on a smooth signal, at an order well
    above the signal's own: within 15 percent nightly, 20 to 25 percent in
    CI, of the true sigma on average over the replicates, and available
    in at least 80 percent of them."""
    expr, var, fn, pt, scn = _case(name)
    x = np.linspace(scn.x0, scn.x1, scn.n)
    yc = fn(x, *pt)
    sig = 0.05 * np.std(yc)
    got = []
    for s in range(REPS):
        rng = np.random.default_rng(4242 + s)
        img = Original(x, yc + sig * rng.standard_normal(x.size)).image(
            "legendre", 40
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = img.noise_sigma()
        if v is not None:
            got.append(v / sig)
    assert len(got) >= 0.8 * REPS, f"{name}: {len(got)}/{REPS} had a tail"
    ratio = float(np.mean(got))
    lo, hi = (0.85, 1.15) if NIGHTLY else (0.80, 1.25)
    assert lo <= ratio <= hi, f"{name}: noise_sigma ratio {ratio:.3f}"


# A proportion needs more replicates than a mean: the band is meaningless
# below about 200 pairs.
FALSE_ALARM_REPS = 600 if NIGHTLY else 200
NOISE_DRAWS = {
    "gaussian": lambda r, m: r.standard_normal(m),
    "student_t3": lambda r, m: t_dist.rvs(3, size=m, random_state=r)
    / np.sqrt(3.0),
    "laplace": lambda r, m: laplace.rvs(size=m, random_state=r) / np.sqrt(2.0),
}


@pytest.mark.parametrize("noise", sorted(NOISE_DRAWS))
def test_equality_test_false_alarm_rate(noise):
    """Two images of the same signal under Gaussian, Student-t(3) and
    Laplace noise: the chi-square equality test flags them at its nominal
    rate whatever the tail weight."""
    draw = NOISE_DRAWS[noise]
    x = np.linspace(0.0, 10.0, 400)
    yc = 5.0 / (1.0 + np.exp(-1.5 * (x - 5.0)))
    sig = 0.05 * float(np.std(yc))
    flags = []
    for s in range(FALSE_ALARM_REPS):
        rng = np.random.default_rng(4242 + s)
        a = Original(x, yc + sig * draw(rng, x.size)).image("legendre", 12)
        b = Original(x, yc + sig * draw(rng, x.size)).image("legendre", 12)
        flags.append(a.test_equal(b).reject)
    rate = float(np.mean(flags))
    lo, hi = (0.03, 0.08) if NIGHTLY else (0.02, 0.10)
    assert lo <= rate <= hi, f"{noise}: false-alarm rate {rate:.4f}"
