import warnings

import numpy as np
import pytest
from scipy.optimize import curve_fit

from dtfit.image import (
    Original, Image, fit, fit_lsi, order_for, coverage, fft_frequency_seed,
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
    if name == "damped_oscillation":
        order = 24
    elif name in ("logistic", "gaussian"):
        order = 20
    else:
        order = 12
    res = fit(expr, Original(x, fn(x, *pt)), var, order=order, p0=pt)
    assert np.max(np.abs(res.coeffs / pt - 1.0)) < 1e-7
    assert res.converged and res.rss_source == "samples"
    assert res.n_obs == x.size


def test_fit_on_image_matches_fit_on_original_and_uses_image_rss():
    expr, var, fn, pt, scn = _case("logistic")
    x = _grids(scn, "uniform")
    y = fn(x, *pt) + 0.05 * np.random.default_rng(0).standard_normal(x.size)
    o = Original(x, y)
    a = fit(expr, o, var, order=24, p0=pt)
    b = fit(expr, Image.of(o, "legendre", 24), var, p0=pt)
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
            fit(expr, Original(x, y), var, order=20, p0=pt).coeffs - pt
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
    with pytest.warns(UserWarning, match="differential-evolution"):
        a = fit(
            "A*sin(w*x + p)", Original(x, y), "x", p0=[1.0, 3.0, 0.0], **kw
        )
    with pytest.warns(UserWarning, match="differential-evolution"):
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
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match="not finite"):
            fit(
                "a*exp(b*t)", Original(t, y), "t", order=12,
                p0=[1.0, 1.0],
            )


def test_nan_sensitivity_at_one_sample_does_not_zero_the_column():
    """d/dn of x**n is x**n*log(x), NaN at x=0 with limit 0 there. That one
    non-finite sensitivity must not poison the whole projected column for
    n and stall it at p0."""
    K, Vmax, n = 2.0, 3.0, 2.0
    x = np.linspace(0, 10, 200)
    y = Vmax * x**n / (K**n + x**n)
    r = fit(
        "Vmax*x**n/(K**n + x**n)", Original(x, y), "x", order=12,
        p0=[1.5, 2.5, 1.5],
    )
    assert abs(r.params["K"] - K) < 1e-6 * K
    assert abs(r.params["Vmax"] - Vmax) < 1e-6 * Vmax
    assert abs(r.params["n"] - n) < 1e-6 * n


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
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
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
    order = 20 if name in ("logistic", "gaussian") else 12
    res = fit(
        expr, Original(x, fn(x, *pt)), var, order=order, p0=0.9 * pt
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
    with pytest.warns(UserWarning, match="coverage"):
        fit(expr, Original(x, y), var, order=4, p0=pt)
    with pytest.warns(UserWarning, match="coverage"):
        fit_lsi(x, y, expr, var, k_star=4, p0=pt)


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


def test_auto_basis_returns_the_best_candidate():
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
    o = Original(xg, yg)
    r = fit(expr, o, var, basis="auto", p0=pt)
    assert np.max(np.abs(r.coeffs / pt - 1)) < 0.05
    a = fit(expr, o, var, p0=pt)
    b = fit(expr, o, var, basis="block", p0=pt)
    assert r.rss <= min(a.rss, b.rss) * (1 + 1e-3)
    with pytest.raises(TypeError):
        fit(
            expr, Image.of(o, "legendre", 8), var,
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


def test_from_dict_without_image_keys_defaults_to_none():
    d = {
        "expr": "a + b*x", "var": "x", "names": ["a", "b"],
        "coeffs": [1.0, 2.0], "cov": None, "x_range": [0.0, 1.0],
    }
    r = FittingResult.from_dict(d)
    assert r.rss_source is None
    assert r.image_order is None
    assert r.basis_name is None


def test_frequency_seed_on_a_non_uniform_grid():
    class _Scn:
        x0, x1, n = 0.0, 12.0, 400

    x = _grids(_Scn(), "random")
    y = (
        1.0 + 2.0 * np.sin(1.5 * x + 0.5)
        + 0.05 * np.random.default_rng(3).standard_normal(x.size)
    )
    assert abs(fft_frequency_seed(x, y) - 1.5) < 0.3


def test_frequency_seed_sees_a_cycle_under_a_trend():
    """The straight line is removed before the FFT; without that its leakage
    owns the lowest non-zero bin and the peak lands there."""
    x = np.linspace(0.0, 20.0, 400)
    y = 1.0 + 0.5 * x + 2.0 * np.sin(1.3 * x + 0.4)
    assert fft_frequency_seed(x, y) == pytest.approx(1.3, abs=0.06)
    assert fft_frequency_seed(x, y - np.polyval(np.polyfit(x, y, 1), x)) == (
        pytest.approx(fft_frequency_seed(x, y))
    )


def test_auto_recovers_a_cycle_riding_on_a_trend():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 20.0, 400)
    y = (1.0 + 0.5 * x + 2.0 * np.sin(1.3 * x + 0.4)
         + rng.normal(0.0, 0.1, x.size))
    r = fit(
        "a0 + a1*x + A*sin(w*x + p)", Original(x, y), "x", basis="auto",
        p0={"a0": 1.0, "a1": 0.5, "A": 2.0, "w": 1.0, "p": 0.0},
        freq_param="w",
    )
    assert r.params["w"] == pytest.approx(1.3, abs=0.01)
    assert r.params["A"] == pytest.approx(2.0, abs=0.05)


def test_auto_with_a_frequency_parameter_runs_two_candidates(monkeypatch):
    """``freq_param`` turns the oscillatory recipe on inside every candidate,
    so a plain Legendre candidate would repeat the oscillatory one."""
    import sys

    # dtfit.image re-exports the fit function, which shadows the module of
    # the same name, so the module is reached through sys.modules.
    module = sys.modules["dtfit.image.fit"]
    real = module.fit
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs.get("basis"))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "fit", counting)
    x = np.linspace(0, 12, 400)
    y = (1.0 + 2.0 * np.sin(1.5 * x + 0.5)
         + 0.05 * np.random.default_rng(3).standard_normal(400))
    o = Original(x, y)
    p0 = {"c": 0.0, "A": 1.0, "w": 0.3, "p": 0.0}
    real("c + A*sin(w*x + p)", o, "x", basis="auto", p0=p0, freq_param="w")
    assert calls == ["legendre", "block"]
    calls.clear()
    real("c + A*sin(w*x + p)", o, "x", basis="auto", p0=p0)
    assert calls == ["legendre", "legendre", "block"]
