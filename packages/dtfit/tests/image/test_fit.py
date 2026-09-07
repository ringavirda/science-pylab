import numpy as np
import pytest
from scipy.optimize import curve_fit

from dtfit.image import (
    Original, Image, fit, order_for, coverage, fft_frequency_seed,
)
from dtfit.types import FittingResult

from accuracy.scenarios import SCENARIOS_BY_NAME


def _case(name):
    scn = SCENARIOS_BY_NAME[name]
    import sympy as sp
    m = scn.model()
    t = sp.Symbol(m.var)
    f = sp.sympify(m.expr)
    names = sorted(str(s) for s in f.free_symbols if s != t)
    fn = sp.lambdify((t, *[sp.Symbol(n) for n in names]), f, "numpy")
    return m.expr, m.var, fn, np.array([scn.true[n] for n in names]), scn


def _grids(scn, kind, seed=1):
    n = scn.n
    if kind == "uniform":
        return np.linspace(scn.x0, scn.x1, n)
    if kind == "clustered":
        a = int(0.8 * n)
        third = scn.x0 + (scn.x1 - scn.x0) / 3
        return np.sort(np.concatenate([
            np.linspace(scn.x0, third, a),
            np.linspace(third, scn.x1, n - a + 1)[1:],
        ]))
    x = np.sort(np.random.default_rng(seed).uniform(scn.x0, scn.x1, n))
    x[0], x[-1] = scn.x0, scn.x1
    return x


@pytest.mark.parametrize(
    "name",
    [
        "exp_decay_offset", "logistic", "michaelis_menten", "gompertz",
        "gaussian", "damped_oscillation",
    ],
)
@pytest.mark.parametrize("kind", ["uniform", "clustered", "random"])
def test_noise_free_fit_is_exact_on_any_grid(name, kind):
    expr, var, fn, pt, scn = _case(name)
    x = _grids(scn, kind)
    order = 24 if name == "damped_oscillation" else 12
    res = fit(expr, Original(x, fn(x, *pt)), var, order=order, p0=pt)
    assert np.max(np.abs(res.coeffs / pt - 1.0)) < 1e-7
    assert res.converged and res.rss_source == "samples"
    assert res.n_obs == x.size


def test_fit_on_image_matches_fit_on_original_and_uses_image_rss():
    expr, var, fn, pt, scn = _case("logistic")
    x = _grids(scn, "uniform")
    y = fn(x, *pt) + 0.05 * np.random.default_rng(0).standard_normal(x.size)
    o = Original(x, y)
    a = fit(expr, o, var, order=12, p0=pt)
    b = fit(expr, Image.of(o, "legendre", 12), var, p0=pt)
    assert np.allclose(a.coeffs, b.coeffs, atol=1e-8)
    assert b.rss_source == "image"
    assert abs(b.rss - a.rss) / a.rss < 0.02
    assert np.allclose(
        np.sqrt(np.diag(a.cov)), np.sqrt(np.diag(b.cov)), rtol=0.02
    )


def test_image_rss_is_exact_for_a_model_in_the_span():
    x = np.linspace(0, 1, 100)
    y = 1.0 + 2.0 * x + 0.1 * np.random.default_rng(1).standard_normal(100)
    o = Original(x, y)
    a = fit("a + b*x", o, "x", order=4, p0=[1.0, 2.0])
    b = fit("a + b*x", Image.of(o, "legendre", 4), "x", p0=[1.0, 2.0])
    assert abs(a.rss - b.rss) < 1e-9


def test_covariance_is_calibrated_and_tracks_sigma_semantics():
    expr, var, fn, pt, scn = _case("exp_decay_offset")
    x = _grids(scn, "uniform")
    rng = np.random.default_rng(5)
    sig = 0.05 * np.std(fn(x, *pt))
    hits = []
    for _ in range(100):
        y = fn(x, *pt) + sig * rng.standard_normal(x.size)
        r = fit(expr, Original(x, y), var, order=8, p0=pt)
        se = np.sqrt(np.diag(r.cov))
        hits.append(np.abs(r.coeffs - pt) <= 1.96 * se)
    assert 0.88 <= np.mean(hits) <= 0.99
    y = fn(x, *pt) + sig * rng.standard_normal(x.size)
    rel = fit(
        expr, Original(x, y), var, order=8, p0=pt,
        sigma=np.full(x.size, 3.0),
    )
    plain = fit(expr, Original(x, y), var, order=8, p0=pt)
    assert np.allclose(
        np.sqrt(np.diag(rel.cov)), np.sqrt(np.diag(plain.cov)), rtol=1e-6
    )
    absl = fit(
        expr, Original(x, y), var, order=8, p0=pt,
        sigma=np.full(x.size, 3.0), absolute_sigma=True,
    )
    absl2 = fit(
        expr, Original(x, y), var, order=8, p0=pt,
        sigma=np.full(x.size, 6.0), absolute_sigma=True,
    )
    assert np.allclose(
        np.sqrt(np.diag(absl2.cov)), 2.0 * np.sqrt(np.diag(absl.cov)),
        rtol=1e-6,
    )


def test_efficiency_matches_scipy_on_a_clustered_grid():
    expr, var, fn, pt, scn = _case("logistic")
    x = _grids(scn, "clustered")
    rng = np.random.default_rng(7)
    sig = 0.05 * np.std(fn(x, *pt))
    e_img, e_cf = [], []
    for _ in range(20):
        y = fn(x, *pt) + sig * rng.standard_normal(x.size)
        e_img.append(
            fit(expr, Original(x, y), var, order=12, p0=pt).coeffs - pt
        )
        e_cf.append(curve_fit(fn, x, y, p0=pt)[0] - pt)
    ratio = (
        np.sqrt(np.mean(np.square(e_img), axis=0))
        / np.sqrt(np.mean(np.square(e_cf), axis=0))
    )
    assert np.all(ratio < 1.25)


def test_bounds_and_global_fallback_are_reproducible():
    x = np.linspace(0, 10, 300)
    y = (
        2.0 * np.sin(1.5 * x + 0.3)
        + 0.1 * np.random.default_rng(2).standard_normal(300)
    )
    kw = dict(
        order=24,
        bounds={"A": (0.1, 5.0), "w": (0.5, 3.0), "p": (-3.2, 3.2)},
    )
    a = fit(
        "A*sin(w*x + p)", Original(x, y), "x", p0=[1.0, 3.0, 0.0], **kw
    )
    b = fit(
        "A*sin(w*x + p)", Original(x, y), "x", p0=[1.0, 3.0, 0.0], **kw
    )
    assert np.allclose(a.coeffs, b.coeffs)
    assert abs(a.params["w"] - 1.5) < 0.02


def test_robust_flag_needs_an_original_and_helps_under_outliers():
    expr, var, fn, pt, scn = _case("exp_decay_offset")
    x = _grids(scn, "uniform")
    rng = np.random.default_rng(9)
    sig = 0.05 * np.std(fn(x, *pt))
    y = fn(x, *pt) + sig * rng.standard_normal(x.size)
    idx = rng.choice(x.size, 20, replace=False)
    y[idx] += 20 * sig
    plain = fit(expr, Original(x, y), var, order=12, p0=pt)
    rob = fit(expr, Original(x, y), var, order=12, p0=pt, robust=True)
    assert (
        np.max(np.abs(rob.coeffs / pt - 1))
        < 0.5 * np.max(np.abs(plain.coeffs / pt - 1))
    )
    with pytest.raises(TypeError):
        fit(
            expr, Image.of(Original(x, y), "legendre", 12), var, p0=pt,
            robust=True,
        )


def test_input_type_errors():
    x = np.linspace(0, 3, 60)
    y = 2.0 * np.exp(-1.1 * x)
    o = Original(x, y)
    img = Image.of(o, "legendre", 8)
    with pytest.raises(TypeError):
        fit("a*exp(-b*x)", img, "x", sigma=np.ones(x.size), p0=[1.0, 1.0])
    with pytest.raises(TypeError):
        fit("a*exp(-b*x)", img, "x", basis="auto", p0=[1.0, 1.0])
    with pytest.raises(TypeError):
        fit("a*exp(-b*x)", (x, y), "x", order=8, p0=[1.0, 1.0])


def test_unidentified_parameter_reports_infinite_stderr():
    x = np.linspace(0, 1, 100)
    y = 6.0 * x + 1e-6 * np.random.default_rng(3).standard_normal(100)
    r = fit("a*b*x", Original(x, y), "x", order=4, p0=[2.0, 3.0])
    assert r.converged
    assert abs(r.params["a"] * r.params["b"] - 6.0) < 0.1
    assert np.isinf(np.diag(r.cov)).all()


def test_non_finite_model_at_p0_raises():
    t = np.linspace(0, 1000, 400)
    y = np.exp(-0.01 * t)
    with pytest.raises(ValueError, match="not finite"):
        fit(
            "a*exp(b*t)", Original(t, y), "t", order=12,
            p0=[1.0, 1.0],
        )


def test_no_global_fallback_from_a_good_start():
    x = np.linspace(0, 10, 300)
    y = (
        2.0 * np.sin(1.5 * x + 0.3)
        + 0.1 * np.random.default_rng(2).standard_normal(300)
    )
    kw = dict(
        order=24,
        bounds={"A": (0.1, 5.0), "w": (0.5, 3.0), "p": (-3.2, 3.2)},
    )
    r = fit(
        "A*sin(w*x + p)", Original(x, y), "x",
        p0={"A": 2.0, "p": 0.3, "w": 1.5}, **kw,
    )
    assert r.nfev < 60
    assert abs(r.params["w"] - 1.5) < 0.02


@pytest.mark.parametrize(
    "name",
    [
        "exp_decay_offset", "logistic", "michaelis_menten", "gompertz",
        "gaussian",
    ],
)
def test_noise_free_fit_converges_from_a_perturbed_start(name):
    expr, var, fn, pt, scn = _case(name)
    x = _grids(scn, "uniform")
    res = fit(
        expr, Original(x, fn(x, *pt)), var, order=12, p0=0.9 * pt
    )
    assert np.max(np.abs(res.coeffs / pt - 1.0)) < 1e-7


def test_callable_model_and_input_errors():
    x = np.linspace(0, 3, 120)
    y = (
        2.0 * np.exp(-1.1 * x)
        + 0.02 * np.random.default_rng(4).standard_normal(120)
    )
    r = fit(
        lambda x, a, b: a * np.exp(-b * x), Original(x, y),
        order=8, p0=[1.0, 1.0],
    )
    assert r.names == ("a", "b") and abs(r.params["b"] - 1.1) < 0.05
    with pytest.raises(ValueError):
        fit("a*exp(-b*x)", Original(x, y), "x", order=8, p0=[1.0])
    with pytest.raises(ValueError):
        fit("a*exp(-b*x)", Original(x, y), "x", order=0, p0=[1.0, 1.0])
    with pytest.raises(ValueError):
        fit(
            "a + b*x + c*x**2 + d*x**3", Original(x, y), "x", order=2
        )


def test_order_for_matches_measured_orders():
    assert order_for(
        "a0 + a1*x", [1.0, 2.0], (0.0, 5.0), var="x"
    ) == 1
    assert order_for(
        "a0 + a1*x + a2*x**2", [1.0, 0.5, 0.3], (0.0, 5.0), var="x"
    ) == 2
    k = order_for(
        "L/(1 + exp(-k*(x - x0)))", [5.0, 1.5, 5.0], (0.0, 10.0), var="x"
    )
    assert 12 <= k <= 20
    k = order_for(
        "A*exp(-z*w*x)*sin(w*sqrt(1 - z**2)*x)", [2.0, 2.0, 0.12],
        (0.0, 12.0), var="x",
    )
    assert 12 <= k <= 24


def test_default_order_and_coverage_warning():
    expr, var, fn, pt, scn = _case("logistic")
    x = _grids(scn, "uniform")
    y = fn(x, *pt)
    r = fit(expr, Original(x, y), var, p0=pt)
    assert r.image_order >= 12
    assert np.max(np.abs(r.coeffs / pt - 1)) < 1e-6
    low = Image.of(Original(x, y), "legendre", 4)
    with pytest.warns(UserWarning, match="coverage"):
        fit(expr, low, var, p0=pt)
    assert coverage(expr, pt, low, var=var) > 0.02


def test_oscillatory_recipe_seeds_frequency():
    x = np.linspace(0, 12, 400)
    y = (
        1.0 + 2.0 * np.sin(1.5 * x + 0.5)
        + 0.05 * np.random.default_rng(3).standard_normal(400)
    )
    assert abs(fft_frequency_seed(x, y) - 1.5) < 0.3
    r = fit(
        "c + A*sin(w*x + p)", Original(x, y), "x",
        p0={"c": 0.0, "A": 1.0, "w": 0.3, "p": 0.0}, freq_param="w",
    )
    assert abs(r.params["w"] - 1.5) < 0.01
    assert r.image_order >= 12


def test_auto_basis_routes_by_shape():
    x = np.linspace(0, 12, 400)
    y = (
        1.0 + 2.0 * np.sin(1.5 * x + 0.5)
        + 0.05 * np.random.default_rng(3).standard_normal(400)
    )
    r = fit(
        "c + A*sin(w*x + p)", Original(x, y), "x", basis="auto",
        p0={"c": 0.0, "A": 1.0, "w": 0.3, "p": 0.0}, freq_param="w",
    )
    assert r.basis_name == "legendre"
    assert abs(r.params["w"] - 1.5) < 0.01
    expr, var, fn, pt, scn = _case("exp_decay_offset")
    xg = _grids(scn, "uniform")
    yg = fn(xg, *pt) + 0.02 * np.random.default_rng(1).standard_normal(
        xg.size
    )
    r = fit(expr, Original(xg, yg), var, basis="auto", p0=pt)
    assert r.basis_name in ("legendre", "block")
    assert np.max(np.abs(r.coeffs / pt - 1)) < 0.05
    with pytest.raises(TypeError):
        fit(
            expr, Image.of(Original(xg, yg), "legendre", 8), var,
            basis="auto", p0=pt,
        )


def test_result_round_trips_through_dict():
    x = np.linspace(0, 5, 60)
    y = 1.0 + 2.0 * x + 0.01 * np.random.default_rng(0).standard_normal(60)
    r = fit("a + b*x", Original(x, y), "x", p0=[1.0, 1.0])
    r2 = FittingResult.from_dict(r.to_dict())
    assert r2.rss_source == r.rss_source
    assert r2.image_order == r.image_order
    assert r2.basis_name == r.basis_name
