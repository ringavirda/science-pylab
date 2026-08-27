""":func:`dtfit.fit_lsi` beyond a plain string model: callable models,
per-sample sigma weights, absolute_sigma covariance scaling, solver_options
reaching the solver, and the fit-quality statistics on the result."""

import numpy as np
import pytest

from dtfit import fit_lsi


# callable models
def test_lsi_callable_matches_string_expr(exp_data):
    """A Python callable model fits the same as the equivalent string
    expression, both resolving to the ``[a, b]`` parameter layout, and
    recovers the true parameters."""
    x, y, true = exp_data
    r_str = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_call = fit_lsi(x, y, lambda t, a, b: a * np.exp(b * t), "t", p0=[1.0, -0.5])

    np.testing.assert_allclose(r_call.coeffs, r_str.coeffs, atol=1e-6, rtol=1e-6)
    a, b = r_call.coeffs
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3
    # A callable carries no expression, so the result cannot serialize; the
    # numeric model and predict path still work.
    assert r_call.expr is None
    assert callable(r_call.model)
    np.testing.assert_allclose(r_call.model(x), r_str.model(x), atol=1e-6)


def test_lsi_callable_introspects_param_names(exp_data):
    """Signature order (after the leading variable) names a callable's params."""
    x, y, _ = exp_data
    r = fit_lsi(x, y, lambda t, a, b: a * np.exp(b * t), "t", p0=[1.0, -0.5])
    assert r.names == ("a", "b")
    assert set(r.params) == {"a", "b"}


def test_lsi_callable_predict_return_std(exp_data):
    """A callable-only result finite-differences ``param_model`` for the
    prediction std band. ``to_dict`` still raises: there is no expression to
    serialize."""
    x, y, _ = exp_data
    r = fit_lsi(x, y, lambda t, a, b: a * np.exp(b * t), "t", p0=[1.0, -0.5])
    xp = np.linspace(0.0, 2.0, 12)
    yp, std = r.predict(xp, return_std=True)
    assert yp.shape == xp.shape
    assert std.shape == xp.shape
    assert np.all(np.isfinite(yp)) and np.all(np.isfinite(std))
    assert np.all(std >= 0.0)
    with pytest.raises(ValueError):
        r.to_dict()


def test_lsi_callable_oscillatory_freq_param():
    """freq_param seeds the angular-frequency parameter from the FFT peak even
    for a callable: the recipe is data-driven, with no symbolic order-raise."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 500)
    a_t, w_t, p_t = 2.0, 3.0, 0.5
    y = a_t * np.sin(w_t * x + p_t) + rng.normal(0.0, 0.02, x.size)

    def f(t, a, w, phi):
        return a * np.sin(w * t + phi)

    r = fit_lsi(x, y, f, "t", param_names=["a", "w", "phi"],
                freq_param="w", p0=[1.0, 1.0, 0.0])
    assert abs(r.params["w"] - w_t) < 0.2
    assert abs(abs(r.params["a"]) - a_t) < 0.2


# per-sample sigma weights
def test_lsi_sigma_downweights_corrupted_tail():
    """A large sigma over a corrupted region down-weights it: the weighted fit
    recovers the clean-region parameters where the unweighted fit is pulled
    off by the corruption."""
    x = np.linspace(0.0, 2.0, 120)
    a_true, b_true = 2.5, -1.2
    y = a_true * np.exp(b_true * x)
    tail = x > 1.4
    y = y.copy()
    y[tail] += 3.0  # gross systematic corruption of the tail
    sigma = np.where(tail, 100.0, 0.01)  # trust the clean head, distrust the tail

    r_w = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], sigma=sigma)
    r_u = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])

    # The weighted fit beats the unweighted one on both parameters.
    assert abs(r_w.coeffs[0] - a_true) < abs(r_u.coeffs[0] - a_true)
    assert abs(r_w.coeffs[1] - b_true) < abs(r_u.coeffs[1] - b_true)
    # It also lands within 0.1 of the truth, not merely closer.
    assert abs(r_w.coeffs[0] - a_true) < 0.1
    assert abs(r_w.coeffs[1] - b_true) < 0.1


def test_lsi_sigma_none_is_bit_identical(exp_data):
    """Passing ``sigma=None`` is bit-identical to omitting sigma entirely."""
    x, y, _ = exp_data
    r_default = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_none = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], sigma=None)
    np.testing.assert_array_equal(r_default.coeffs, r_none.coeffs)
    np.testing.assert_array_equal(
        np.asarray(r_default.cov), np.asarray(r_none.cov)
    )


def test_lsi_sigma_validation(exp_data):
    """sigma must match the raw sample count and be finite and strictly
    positive. The messages come from the shared ``_resolve_sigma``, so
    fit_eac reports these failures the same way."""
    x, y, _ = exp_data
    with pytest.raises(ValueError, match="same length as data_y"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                sigma=np.ones(x.size - 1))
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                sigma=np.zeros(x.size))
    bad = np.ones(x.size)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], sigma=bad)
    bad_nan = np.ones(x.size)
    bad_nan[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], sigma=bad_nan)


# absolute_sigma covariance scaling
def test_lsi_absolute_sigma_scales_se_by_k(exp_data):
    """With ``absolute_sigma=True`` the standard errors scale with sigma:
    multiplying every sigma by k multiplies each SE by k (scipy semantics),
    while the fitted coefficients are unchanged."""
    x, y, _ = exp_data
    sig = np.full(x.size, 0.05)
    r1 = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                 sigma=sig, absolute_sigma=True)
    r10 = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                  sigma=10.0 * sig, absolute_sigma=True)
    se1 = np.sqrt(np.diag(r1.cov))
    se10 = np.sqrt(np.diag(r10.cov))
    np.testing.assert_allclose(se10, 10.0 * se1, rtol=1e-6)
    np.testing.assert_allclose(r1.coeffs, r10.coeffs, atol=1e-8)


def test_lsi_relative_sigma_leaves_se_invariant(exp_data):
    """With ``absolute_sigma=False`` (default) sigma is relative: a uniform
    scaling of sigma leaves the standard errors unchanged."""
    x, y, _ = exp_data
    sig = np.full(x.size, 0.05)
    r1 = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                 sigma=sig, absolute_sigma=False)
    r10 = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                  sigma=10.0 * sig, absolute_sigma=False)
    se1 = np.sqrt(np.diag(r1.cov))
    se10 = np.sqrt(np.diag(r10.cov))
    np.testing.assert_allclose(se10, se1, rtol=1e-6)


# solver_options
def test_lsi_solver_options_max_nfev(exp_data):
    """solver_options forward tolerances to the driver; a tight ``max_nfev``
    caps the evaluation count and prevents convergence. Of the three
    assertions after the guard, two catch a dropped solver_options: the capped
    fit would then converge at the full nfev. The third only establishes the
    baseline, since the uncapped fit never receives solver_options."""
    x, y, _ = exp_data
    r_full = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_lim = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                    solver_options={"max_nfev": 1})
    assert r_full.nfev is not None and r_lim.nfev is not None
    # The capped count is scipy-version dependent (1 to 4). Compare against
    # the full fit, never an absolute number.
    assert r_lim.nfev < r_full.nfev
    assert r_lim.converged is False   # a 1-eval fit cannot converge
    assert r_full.converged is True   # the uncapped full fit does


def test_lsi_solver_options_bounded_maps_maxfun(exp_data, monkeypatch):
    """On the bounded DE + L-BFGS-B path, solver_options reach the refine step
    with max_nfev mapped onto L-BFGS-B's ``maxfun``. The spy on ``minimize``
    pins that mapping, so a rename shows up as a failure here."""
    import dtfit._core._spectral as _spectral

    seen = {}
    orig = _spectral.minimize

    def spy(*a, **k):
        seen["options"] = k.get("options")
        return orig(*a, **k)

    monkeypatch.setattr(_spectral, "minimize", spy)
    x, y, _ = exp_data
    # all-finite bounds and no p0: DE global stage, then L-BFGS-B refine
    r = fit_lsi(x, y, "a*exp(b*x)", "x",
                bounds=[(0.0, 10.0), (-5.0, 0.0)],
                solver_options={"max_nfev": 4, "ftol": 1e-3, "gtol": 1e-3})
    assert r.converged
    assert seen["options"] == {"ftol": 1e-3, "gtol": 1e-3, "maxfun": 4}


def test_lsi_robust_threads_solver_options_to_inner_solves(exp_data, monkeypatch):
    """The robust IRLS re-solves honour solver_options, not only the main
    solve. The spy on the inner ``least_squares`` shows a tight ``max_nfev``
    reaching every re-solve, matching fit_eac's robust threading."""
    import dtfit.methods._lsi as _lsi

    seen: list = []
    orig = _lsi.least_squares

    def spy(*a, **k):
        seen.append(k.get("max_nfev"))
        return orig(*a, **k)

    monkeypatch.setattr(_lsi, "least_squares", spy)
    x, y, _ = exp_data
    fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], robust=True,
            solver_options={"max_nfev": 5})
    assert seen and all(v == 5 for v in seen)


# optional var, matching fit_eac
def test_lsi_var_optional_symbolic_requires_var(exp_data):
    """``var`` is optional in the signature, but a symbolic model still needs
    it: resolve_model raises the same error fit_eac does."""
    x, y, _ = exp_data
    with pytest.raises(ValueError, match="var is required"):
        fit_lsi(x, y, "a*exp(b*x)")


def test_lsi_var_optional_callable_defaults_to_x(exp_data):
    """A callable needs no ``var``: it defaults to ``'x'``, and omitting the
    argument fits the same coefficients as the string twin."""
    x, y, _ = exp_data
    r_novar = fit_lsi(x, y, lambda t, a, b: a * np.exp(b * t), p0=[1.0, -0.5])
    r_str = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert r_novar.var == "x"
    np.testing.assert_allclose(r_novar.coeffs, r_str.coeffs, atol=1e-6, rtol=1e-6)


# fit-quality statistics
def test_lsi_reports_fit_quality_stats(exp_data):
    """The result carries R^2, AIC/BIC, n_obs, cost and nfev, all present and
    sane for a clean exponential."""
    x, y, _ = exp_data
    r = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert r.n_obs == x.size
    assert r.rss is not None and r.rss >= 0.0
    assert r.tss is not None and r.tss > 0.0
    assert r.rsquared is not None and 0.9 < r.rsquared <= 1.0
    assert r.aic is not None and np.isfinite(r.aic)
    assert r.bic is not None and np.isfinite(r.bic)
    assert r.cost is not None and r.cost >= 0.0
    assert r.nfev is not None and r.nfev >= 1
    # R^2 surfaces in the human-readable summary.
    assert "R^2" in r.summary()


def test_lsi_stats_survive_callable_model(exp_data):
    """The diagnostics are computed over the raw samples regardless of model
    form, so a callable reports the same stats as its string twin."""
    x, y, _ = exp_data
    r_str = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_call = fit_lsi(x, y, lambda t, a, b: a * np.exp(b * t), "t", p0=[1.0, -0.5])
    assert r_call.rsquared is not None
    np.testing.assert_allclose(r_call.rsquared, r_str.rsquared, rtol=1e-6)
    np.testing.assert_allclose(r_call.rss, r_str.rss, rtol=1e-6)
    assert r_call.n_obs == r_str.n_obs


# sigma under nan_policy='omit', a contract shared with fit_eac
def test_lsi_and_eac_agree_on_sigma_omit_length(exp_data):
    """sigma is indexed against the raw samples, one entry each, and stays that
    way under nan_policy='omit'. Both fitters have to read it the same way;
    the shared ``_resolve_sigma`` is what keeps them in step."""
    from dtfit import fit_eac

    x, y, true = exp_data
    y = y.copy()
    y[5] = np.nan  # one dropped pair under omit
    sigma = np.full(x.size, 0.05)  # raw length, not the post-drop length
    r_l = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                  sigma=sigma, nan_policy="omit")
    r_e = fit_eac(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                  sigma=sigma, nan_policy="omit")
    assert r_l.n_obs == x.size - 1 and r_e.n_obs == x.size - 1
    assert abs(r_l.params["a"] - true["a"]) < 0.4
    assert abs(r_e.params["a"] - true["a"]) < 0.4
