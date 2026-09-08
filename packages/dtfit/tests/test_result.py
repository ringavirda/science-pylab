"""FittingResult: named params, uncertainty, prediction bands, serialization."""

import warnings

import numpy as np
import pytest

from dtfit import fit_lsi, fit_eac, FittingResult
from dtfit.models import resolve_model, result_kwargs
from dtfit._stats import information_criteria, nlls_covariance


@pytest.fixture
def fit():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300)
    y = 2.0 * np.exp(0.8 * t) + rng.normal(0, 0.05, t.size)
    return fit_lsi(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0]), t, y


def test_named_params_and_back_compat(fit):
    r, t, y = fit
    assert set(r.params) == {"a", "b"}
    assert r.params["a"] == pytest.approx(2.0, abs=0.1)
    assert r.coeffs.shape == (2,)
    assert np.asarray(r.model(t)).shape == t.shape


def test_stderr_and_confidence_intervals(fit):
    r, t, y = fit
    se = r.stderr()
    assert set(se) == {"a", "b"} and all(v >= 0 for v in se.values())
    ci = r.confidence_intervals(level=0.95)
    lo, hi = ci["b"]
    assert lo < r.params["b"] < hi


def test_predict_with_std(fit):
    r, t, y = fit
    yhat, std = r.predict(t, return_std=True)
    assert yhat.shape == t.shape and std.shape == t.shape
    assert np.all(std >= 0)


def test_serialization_roundtrip(fit):
    r, t, y = fit
    d = r.to_dict()
    assert set(d) >= {"expr", "var", "names", "coeffs", "cov"}
    r2 = FittingResult.from_dict(d)
    assert np.allclose(r2.coeffs, r.coeffs)
    assert np.allclose(r2.predict(t), r.predict(t))
    assert r2.x_range == r.x_range


def test_convergence_flag_is_reported(fit):
    r, t, y = fit
    assert r.converged is True
    assert isinstance(r.message, str) and r.message
    # eac populates it too
    re = fit_eac(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0])
    assert re.converged is True


def test_predict_warns_only_on_extrapolation(fit):
    r, t, y = fit
    assert r.x_range is not None and r.x_range[0] <= t[0] and r.x_range[1] >= t[-1]
    # inside the fitted range: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.predict(t, warn_extrapolation=True)
    # past the fitted range: a UserWarning fires
    with pytest.warns(UserWarning, match="extrapolat"):
        r.predict(np.array([t[-1] + 10.0]), warn_extrapolation=True)
    # opt-in only: the default call stays silent even past the range
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.predict(np.array([t[-1] + 10.0]))


def test_summary_is_str(fit):
    r, _, _ = fit
    s = r.summary()
    assert "a =" in s and "b =" in s


def test_no_expr_result_degrades_gracefully():
    # callable-only result: model() still works, to_dict() and UQ do not
    x = np.linspace(0, 1, 20)
    r = FittingResult(coeffs=np.array([0.0, 0.0, 1.0]), model=np.poly1d([1.0, 0.0, 0.0]))
    assert np.asarray(r.model(x)).shape == x.shape
    with pytest.raises(ValueError):
        r.to_dict()


def test_param_model_std_band_matches_expr_band(fit):
    # a callable-only result must reproduce the expr-based std band by
    # finite-differencing param_model, to ~1e-6.
    r, t, y = fit
    coeffs = r.coeffs

    def f(x, a, b):
        return a * np.exp(b * x)

    spec = resolve_model(f)  # callable form of the same model
    r_call = FittingResult(coeffs=coeffs, cov=r.cov, **result_kwargs(spec, coeffs))
    assert r_call.expr is None and r_call.param_model is not None

    y_ex, std_ex = r.predict(t, return_std=True)
    y_ca, std_ca = r_call.predict(t, return_std=True)
    assert np.all(np.isfinite(std_ca)) and np.all(std_ca >= 0)
    np.testing.assert_allclose(y_ca, y_ex, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(std_ca, std_ex, rtol=1e-6, atol=1e-9)
    # .model falls back to param_model for a callable-only result
    np.testing.assert_allclose(np.asarray(r_call.model(t)), y_ex, rtol=1e-9)


def test_rsquared_aic_bic_math():
    rss, tss, n = 4.0, 20.0, 50
    names = ("a", "b", "c")
    r = FittingResult(coeffs=np.array([1.0, 2.0, 3.0]), names=names,
                      rss=rss, tss=tss, n_obs=n)
    assert r.rsquared == pytest.approx(1.0 - rss / tss)
    aic, bic = information_criteria(rss, n, len(names))
    assert r.aic == pytest.approx(aic)
    assert r.bic == pytest.approx(bic)
    # missing pieces -> None (not an error)
    bare = FittingResult(coeffs=np.array([1.0]), names=("a",))
    assert bare.rsquared is None and bare.aic is None and bare.bic is None


def test_stats_roundtrip_and_summary(fit):
    r, t, y = fit
    yhat = r.predict(t)
    rss = float(np.sum((y - yhat) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = FittingResult(coeffs=r.coeffs, cov=r.cov, expr=r.expr, var=r.var,
                       names=r.names, rss=rss, tss=tss, n_obs=y.size, nfev=17)
    d = r2.to_dict()
    assert d["rss"] == rss and d["tss"] == tss and d["n_obs"] == y.size
    assert d["nfev"] == 17
    back = FittingResult.from_dict(d)
    assert back.rss == pytest.approx(rss) and back.nfev == 17
    assert back.rsquared == pytest.approx(r2.rsquared)
    assert "R^2" in r2.summary()


def test_residuals_helper(fit):
    r, t, y = fit
    res = r.residuals(t, y)
    assert res.shape == t.shape
    np.testing.assert_allclose(res, y - r.predict(t))


def test_predict_pandas_in_pandas_out(fit):
    pd = pytest.importorskip("pandas")
    r, t, _ = fit
    idx = pd.date_range("2024-01-01", periods=t.size, freq="D")
    ts = pd.Series(t, index=idx)

    y_arr = r.predict(t)
    y_ser = r.predict(ts)
    assert isinstance(y_arr, np.ndarray)
    assert isinstance(y_ser, pd.Series)
    assert list(y_ser.index) == list(idx)
    np.testing.assert_allclose(y_ser.to_numpy(), y_arr)

    y2, std2 = r.predict(ts, return_std=True)
    assert isinstance(y2, pd.Series) and isinstance(std2, pd.Series)
    assert list(std2.index) == list(idx)


def test_covariance_absolute_sigma_invariants():
    rng = np.random.default_rng(0)
    m, n = 40, 3
    jac = rng.normal(size=(m, n))
    res = rng.normal(size=m)
    k = 3.7

    cov_def = nlls_covariance(jac, res, n)
    cov_abs = nlls_covariance(jac, res, n, absolute_sigma=True)
    assert cov_def is not None and cov_abs is not None

    # default cov = absolute cov * reduced chi-square
    sigma2 = float(res @ res) / (m - n)
    np.testing.assert_allclose(cov_def, sigma2 * cov_abs, rtol=1e-10)

    # absolute-sigma cov ignores the residual magnitude (sigma^2 = 1)
    cov_abs_scaled = nlls_covariance(jac, k * res, n, absolute_sigma=True)
    np.testing.assert_allclose(cov_abs_scaled, cov_abs, rtol=1e-10)

    # default cov scales by k^2 when the residual is scaled by k (jac fixed)
    cov_def_res = nlls_covariance(jac, k * res, n)
    np.testing.assert_allclose(cov_def_res, k**2 * cov_def, rtol=1e-10)

    # scaling jac and res together rescales the assumed sigma, not the fit,
    # so the default cov comes out unchanged
    cov_def_global = nlls_covariance(k * jac, k * res, n)
    np.testing.assert_allclose(cov_def_global, cov_def, rtol=1e-10)
