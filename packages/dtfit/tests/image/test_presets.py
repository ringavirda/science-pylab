import numpy as np
import pytest
import sympy as sp

import dtfit
from dtfit import fit_lsi, fit_eac
from dtfit.image import fit, Original


def _exp(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 3.0, n)
    y = 3.0 * np.exp(-1.0 * x) + 0.05 * rng.standard_normal(n)
    return x, y


def test_presets_are_the_same_call_as_fit():
    x, y = _exp()
    a = fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], k_star=8)
    b = fit("a*exp(-b*x)", Original(x, y), "x", order=8, p0=[1.0, 1.0])
    assert np.allclose(a.coeffs, b.coeffs)
    c = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], n_windows=8)
    d = fit("a*exp(-b*x)", Original(x, y), "x", basis="block", order=8,
            p0=[1.0, 1.0])
    assert np.allclose(c.coeffs, d.coeffs)
    assert dtfit.fit_lsi is fit_lsi and dtfit.fit_eac is fit_eac


def test_presets_recover_parameters_with_defaults():
    x, y = _exp()
    for fn in (fit_lsi, fit_eac):
        r = fn(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0])
        assert (abs(r.params["a"] - 3.0) < 0.05
                and abs(r.params["b"] - 1.0) < 0.05)
        assert (r.cov is not None and r.converged and r.n_obs == 200
                and r.rsquared > 0.99)


def test_legacy_keywords_warn_and_are_ignored():
    x, y = _exp()
    with pytest.warns(DeprecationWarning, match="filter_data"):
        fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], filter_data=True)
    with pytest.warns(DeprecationWarning, match="window_mode"):
        fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                window_mode="curvature")
    with pytest.warns(DeprecationWarning, match="loss"):
        r = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], loss="soft_l1")
    assert abs(r.params["b"] - 1.0) < 0.05
    with pytest.raises(TypeError):
        fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], no_such_option=1)


def test_k_star_auto_and_dict_inputs():
    x, y = _exp()
    r = fit_lsi(x, y, "a*exp(-b*x)", "x", k_star="auto",
                p0={"a": 1.0, "b": 1.0}, bounds={"b": (0.0, 5.0)})
    assert abs(r.params["b"] - 1.0) < 0.05
    with pytest.raises(ValueError):
        fit_lsi(x, y, "a*exp(-b*x)", "x", p0={"a": 1.0})
    with pytest.raises(ValueError):
        fit_eac(x, y, "a*exp(-b*x)", "x", bounds=[(0.0, 1.0), (2.0, 1.0)])
    for fn in (fit_lsi, fit_eac):
        bounds = {"b": (0.0, 5.0)}
        r_dict = fn(x, y, "a*exp(-b*x)", "x", p0={"a": 1.0, "b": 1.0},
                    bounds=bounds)
        r_pos = fn(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], bounds=bounds)
        assert np.allclose(r_dict.coeffs, r_pos.coeffs)


def test_callable_models_keep_signature_order_and_predict():
    x, y = _exp()
    r = fit_eac(x, y, lambda x, a, b: a * np.exp(-b * x), p0=[1.0, 1.0])
    assert r.names == ("a", "b") and r.var == "x"
    yhat, std = r.predict(x, return_std=True)
    assert yhat.shape == std.shape == x.shape
    with pytest.raises(ValueError):
        r.to_dict()
    r2 = fit_lsi(x, y, lambda x, *p: p[0] * np.exp(-p[1] * x),
                 param_names=["a", "b"], p0=[1.0, 1.0])
    assert r2.names == ("a", "b")
    e = fit_lsi(x, y, sp.sympify("a*exp(-b*x)"), "x", p0=[1.0, 1.0])
    assert e.to_dict()["expr"] == "a*exp(-b*x)"


def test_sigma_semantics_and_validation():
    x, y = _exp()
    base = fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0])
    rel = fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                  sigma=np.full(x.size, 2.0))
    assert np.allclose(rel.stderr()["b"], base.stderr()["b"], rtol=1e-6)
    a1 = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                 sigma=np.full(x.size, 1.0), absolute_sigma=True)
    a2 = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                 sigma=np.full(x.size, 2.0), absolute_sigma=True)
    assert np.isclose(a2.stderr()["b"], 2.0 * a1.stderr()["b"], rtol=1e-6)
    with pytest.raises(ValueError):
        fit_lsi(x, y, "a*exp(-b*x)", "x", sigma=np.full(x.size - 1, 1.0))
    with pytest.raises(ValueError):
        fit_eac(x, y, "a*exp(-b*x)", "x", sigma=np.full(x.size, 0.0))
    y2 = y.copy()
    y2[10] = np.nan
    r = fit_lsi(x, y2, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                sigma=np.full(x.size, 1.0), nan_policy="omit")
    assert r.n_obs == 199
    with pytest.raises(ValueError):
        fit_lsi(x, y2, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                sigma=np.full(199, 1.0), nan_policy="omit")


def test_nan_policy_and_multivariate_rejection():
    x, y = _exp()
    y2 = y.copy()
    y2[10] = np.nan
    with pytest.raises(ValueError):
        fit_lsi(x, y2, "a*exp(-b*x)", "x", p0=[1.0, 1.0])
    r = fit_lsi(x, y2, "a*exp(-b*x)", "x", p0=[1.0, 1.0], nan_policy="omit")
    assert r.n_obs == 199
    with pytest.raises(ValueError, match="1-D"):
        fit_eac(np.column_stack([x, x]), y, "a*exp(-b*x)", "x")


def test_oscillatory_recipe_through_preset():
    x = np.linspace(0, 12, 400)
    y = (2.0 * np.sin(1.5 * x + 0.5)
         + 0.05 * np.random.default_rng(3).standard_normal(400))
    r = fit_lsi(x, y, "A*sin(w*x + p)", "x", freq_param="w",
                p0=[1.0, 0.0, 0.3])
    assert abs(r.params["w"] - 1.5) < 0.01


def test_converged_flag_propagates_solver_failure():
    x, y = _exp()
    r = fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0],
                solver_options={"max_nfev": 1})
    assert r.converged is False and "converged=False" in repr(r)


def test_pandas_inputs_match_ndarray():
    pd = pytest.importorskip("pandas")
    x, y = _exp()
    a = fit_lsi(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0])
    b = fit_lsi(pd.Series(x), pd.Series(y), "a*exp(-b*x)", "x", p0=[1.0, 1.0])
    assert np.allclose(a.coeffs, b.coeffs)
    with pytest.raises(ValueError):
        fit_eac(x, pd.DataFrame({"a": y, "b": y}), "a*exp(-b*x)", "x")
