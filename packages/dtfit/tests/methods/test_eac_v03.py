""":func:`dtfit.fit_eac` beyond a plain string model:

* the three model-input forms (string, :class:`sympy.Expr`, callable) that
  :func:`dtfit.methods.resolve_model` accepts, with a forward-difference
  Jacobian standing in for the analytic one on the callable path;
* per-sample ``sigma`` turning the area system into weighted least squares,
  and the ``absolute_sigma`` covariance semantics of ``scipy.curve_fit``;
* ``solver_options`` reaching the underlying solver;
* the ``rss`` / ``tss`` / ``n_obs`` / ``nfev`` / ``cost`` diagnostics and the
  ``.rsquared`` / ``.aic`` / ``.bic`` they feed.
"""

import numpy as np
import pytest
import sympy as sp

from dtfit import fit_eac


# callable models and the finite-difference Jacobian
def test_eac_callable_matches_string_expr(arctan_data):
    """A plain ``f(x, a, w)`` callable recovers the truth and lands on the same
    optimum as the equivalent sympy string: the objective is identical, and
    only the Jacobian differs, analytic against forward-difference."""
    x, y, true = arctan_data

    def f(x, a, w):
        return a * np.arctan(w * x)

    c_call = fit_eac(x, y, f, p0=[1.0, 1.0]).coeffs
    c_str = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0]).coeffs
    assert abs(c_call[0] - true["a"]) < 0.5
    assert abs(c_call[1] - true["w"]) < 0.5
    # Same least-squares minimum to solver tolerance.
    np.testing.assert_allclose(c_call, c_str, rtol=1e-5, atol=1e-5)


def test_eac_callable_fd_jacobian_saturating():
    """The forward-difference Jacobian converges on a saturating model given no
    symbolic form at all, and the overdetermined area system still yields a
    covariance."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 300)
    y = 3.0 * (1.0 - np.exp(-0.9 * x)) + rng.normal(0, 0.02, x.size)

    def sat(x, a, k):
        return a * (1.0 - np.exp(-k * x))

    res = fit_eac(x, y, sat, p0=[1.0, 1.0])
    assert res.converged
    assert abs(res.params["a"] - 3.0) < 0.2
    assert abs(res.params["k"] - 0.9) < 0.2
    assert res.cov is not None and res.cov.shape == (2, 2)
    assert np.all(np.isfinite(np.sqrt(np.diag(res.cov))))


def test_eac_callable_uses_signature_parameter_order(arctan_data):
    """A callable's parameters follow signature order, not the sorted-name
    order a symbolic model uses: here ``(w, a)`` where sorting would give
    ``(a, w)``."""
    x, y, true = arctan_data

    def f(x, w, a):  # deliberately not alphabetical
        return a * np.arctan(w * x)

    res = fit_eac(x, y, f, p0=[1.0, 1.0])
    assert res.names == ("w", "a")
    assert abs(res.params["a"] - true["a"]) < 0.5
    assert abs(res.params["w"] - true["w"]) < 0.5


def test_eac_callable_result_predicts_but_does_not_serialize(arctan_data):
    """A callable-only result carries a bound model and ``param_model`` but no
    ``expr``, so ``predict`` works, std band included, while ``to_dict`` raises
    as it does for any expression-less fit."""
    x, y, _ = arctan_data

    def f(x, a, w):
        return a * np.arctan(w * x)

    res = fit_eac(x, y, f, p0=[1.0, 1.0])
    assert res.expr is None
    assert res.param_model is not None
    yp = res.predict(x)
    assert yp.shape == x.shape and np.all(np.isfinite(yp))
    yp2, std = res.predict(x, return_std=True)
    assert std.shape == x.shape and np.all(np.isfinite(std)) and np.all(std >= 0)
    with pytest.raises(ValueError, match="expr"):
        res.to_dict()


def test_eac_callable_without_analytic_derivative_fits():
    """A piecewise ``np.where`` model has no closed-form sympy derivative and
    still fits through the forward-difference Jacobian."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 5, 200)
    y = 3.0 * (1.0 - np.exp(-0.8 * x)) + rng.normal(0, 0.02, x.size)

    def piecewise_sat(x, a, k):
        return a * np.where(x > 0, 1.0 - np.exp(-k * x), 0.0)

    res = fit_eac(x, y, piecewise_sat, p0=[1.0, 1.0])
    assert res.converged
    assert abs(res.params["a"] - 3.0) < 0.2
    assert abs(res.params["k"] - 0.8) < 0.2


def test_eac_callable_param_names_for_varargs(arctan_data):
    """A ``*args`` callable has no introspectable parameter names, so
    ``param_names`` must be supplied; it then drives the canonical order."""
    x, y, true = arctan_data

    def f(x, *p):
        a, w = p
        return a * np.arctan(w * x)

    with pytest.raises(ValueError, match="param_names"):
        fit_eac(x, y, f, p0=[1.0, 1.0])
    res = fit_eac(x, y, f, p0=[1.0, 1.0], param_names=["a", "w"])
    assert res.names == ("a", "w")
    assert abs(res.params["a"] - true["a"]) < 0.5


def test_eac_sympy_expr_input_serializes(arctan_data):
    """A :class:`sympy.Expr` model (not a string) fits like the string form and
    still serializes (its ``expr`` is stored as a string)."""
    x, y, true = arctan_data
    xt, a, w = sp.symbols("x a w")
    res = fit_eac(x, y, a * sp.atan(w * xt), "x", p0=[1.0, 1.0])
    assert isinstance(res.expr, str)
    assert abs(res.params["a"] - true["a"]) < 0.5
    d = res.to_dict()
    assert d["expr"] == res.expr


# sigma weighting and absolute_sigma
def test_eac_sigma_weighting_improves_heteroscedastic_fit():
    """Weighting each window's area residual by ``1/sigma_area`` down-weights a
    noisy tail and beats the unweighted system. One noise draw can go either
    way, so the claim is the median over 24 seeds and a win in most of them."""
    a_true, b_true = 2.5, 1.0
    x = np.linspace(0, 4, 240)
    clean = a_true * np.exp(-b_true * x)
    sigma = np.where(x < 2.0, 0.01, 0.6)  # tail 60x noisier

    err_w, err_u = [], []
    for seed in range(24):
        rng = np.random.default_rng(seed)
        y = clean + rng.normal(0.0, sigma)
        cu = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0]).coeffs
        cw = fit_eac(x, y, "a*exp(-b*x)", "x", p0=[1.0, 1.0], sigma=sigma).coeffs
        err_u.append(np.hypot(cu[0] - a_true, cu[1] - b_true))
        err_w.append(np.hypot(cw[0] - a_true, cw[1] - b_true))
    assert np.median(err_w) < np.median(err_u)
    assert np.mean(np.asarray(err_w) < np.asarray(err_u)) > 0.7


def test_eac_sigma_validation(arctan_data):
    x, y, _ = arctan_data
    with pytest.raises(ValueError, match="same length"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], sigma=np.ones(x.size - 1))
    bad = np.ones(x.size)
    bad[3] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], sigma=bad)
    nan_sig = np.ones(x.size)
    nan_sig[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], sigma=nan_sig)


def test_eac_absolute_sigma_covariance_semantics(arctan_data):
    """``absolute_sigma`` matches ``scipy.curve_fit``: with ``False`` the
    covariance is invariant to a global rescale of ``sigma``; with ``True`` it
    scales by that factor squared. Coefficients are unchanged either way."""
    x, y, _ = arctan_data
    sigma = 0.05 * (1.0 + x / 5.0)

    rf1 = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                  sigma=sigma, absolute_sigma=False)
    rf2 = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                  sigma=10.0 * sigma, absolute_sigma=False)
    # Relative errors: only the sigma shape matters, so cov is scale-invariant.
    np.testing.assert_allclose(rf2.cov, rf1.cov, rtol=1e-6)

    rt1 = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                  sigma=sigma, absolute_sigma=True)
    rt2 = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                  sigma=10.0 * sigma, absolute_sigma=True)
    # Absolute errors: a 10x larger sigma is a 100x larger covariance.
    np.testing.assert_allclose(rt2.cov, 100.0 * rt1.cov, rtol=1e-6)
    np.testing.assert_allclose(rt1.coeffs, rf1.coeffs, rtol=1e-8, atol=1e-8)


# solver_options and diagnostics
def test_eac_solver_options_threads_through(arctan_data):
    x, y, _ = arctan_data
    # An option that actually bites: capping the evaluations stops the solver
    # short of convergence, and both facts are recorded.
    capped = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                     solver_options={"max_nfev": 3})
    full = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    # Assert the invariant, not a count: scipy's LM nfev bookkeeping under
    # max_nfev varies by version.
    assert capped.nfev is not None and capped.nfev < full.nfev
    assert capped.converged is False
    # An unknown option reaches least_squares and raises: proof it forwards.
    with pytest.raises(TypeError, match="unexpected keyword"):
        fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0],
                solver_options={"definitely_not_an_option": 1})


def test_eac_records_fit_diagnostics(arctan_data):
    x, y, _ = arctan_data
    res = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    assert res.n_obs == x.size
    assert res.rss is not None and res.rss > 0
    assert res.tss is not None and res.tss > res.rss
    assert res.nfev is not None and res.nfev > 0
    assert res.cost is not None and res.cost >= 0
    # rss/tss/n_obs feed the derived quality metrics.
    assert res.rsquared is not None and 0.9 < res.rsquared <= 1.0
    assert res.aic is not None and res.bic is not None
    # Round-trip preserves the diagnostics.
    from dtfit.types import FittingResult
    d = res.to_dict()
    back = FittingResult.from_dict(d)
    assert back.n_obs == res.n_obs
    np.testing.assert_allclose(back.rss, res.rss)


def test_eac_robust_nfev_sums_over_inner_solves(arctan_data):
    """The robust IRLS path reports the sum of its inner re-solves' nfev, never
    less than the single-solve count."""
    x, y, _ = arctan_data
    plain = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    robust = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0], robust=True)
    assert plain.nfev is not None and robust.nfev is not None
    assert robust.nfev >= plain.nfev
