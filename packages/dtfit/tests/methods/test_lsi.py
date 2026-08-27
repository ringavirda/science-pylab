"""LSI (least-squares integral) batch method."""

import numpy as np
import pytest

from dtfit import fit_lsi


def test_fit_lsi_recovers_exponential(exp_data):
    x, y, true = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    a, b = result.coeffs  # params sorted by name: a, b
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3
    assert callable(result.model)


def test_lsi_returns_covariance(exp_data):
    x, y, _ = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert result.cov is not None
    assert result.cov.shape == (2, 2)
    # Standard errors are small and finite for this clean signal.
    se = np.sqrt(np.diag(result.cov))
    assert np.all(np.isfinite(se)) and np.all(se < 0.5)


def test_lsi_auto_order(exp_data):
    x, y, true = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", k_star="auto", p0=[1.0, -0.5])
    a, b = result.coeffs
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3


# p0 / bounds normalization: dict and positional forms
def test_lsi_dict_p0_matches_positional(exp_data):
    x, y, _ = exp_data
    pos = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5]).coeffs
    named = fit_lsi(x, y, "a*exp(b*x)", "x", p0={"a": 1.0, "b": -0.5}).coeffs
    np.testing.assert_allclose(named, pos)


def test_lsi_partial_dict_bounds(exp_data):
    x, y, true = exp_data
    # Only 'a' is bounded; 'b' stays unbounded. A bounds dict may be partial,
    # and the resulting mixed box must still constrain the local solve.
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5],
                     bounds={"a": (0.0, 10.0)})
    a, b = result.coeffs
    assert 0.0 <= a <= 10.0
    assert abs(a - true["a"]) < 0.3
    assert abs(b - true["b"]) < 0.3


def test_lsi_p0_dict_missing_and_unknown_names_raise(exp_data):
    x, y, _ = exp_data
    with pytest.raises(ValueError, match=r"missing \['b'\]"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0={"a": 1.0})
    with pytest.raises(ValueError, match=r"unknown \['c'\]"):
        fit_lsi(x, y, "a*exp(b*x)", "x", p0={"a": 1.0, "b": -0.5, "c": 0.0})


def test_lsi_bounds_lo_above_hi_raises(exp_data):
    x, y, _ = exp_data
    with pytest.raises(ValueError, match="'a'"):
        fit_lsi(x, y, "a*exp(b*x)", "x", bounds=[(3.0, 1.0), (-2.0, 0.0)])


def test_lsi_filter_data_defaults_off(exp_data, monkeypatch):
    """A fitter must not quietly modify the caller's data, so the
    Savitzky-Golay pre-filter is opt-in and the default path never calls it."""
    import dtfit.methods._lsi as _lsi_mod
    x, y, _ = exp_data
    called = []
    orig = _lsi_mod._savgol_prefilter
    monkeypatch.setattr(_lsi_mod, "_savgol_prefilter",
                        lambda yy: called.append(True) or orig(yy))
    fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert not called, "default fit_lsi must not pre-filter the data"
    fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], filter_data=True)
    assert called, "filter_data=True must opt into the pre-filter"


def test_lsi_robust_message_reflects_inner_solver(exp_data):
    x, y, _ = exp_data
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], robust=True)
    # The message carries the last inner solver's status rather than a fixed
    # 'robust IRLS' constant.
    assert result.message != "robust IRLS"
    assert result.message.startswith("robust IRLS (")


def test_lsi_robust_converged_flag_propagates_inner_failure(
    monkeypatch, exp_data
):
    """The ``converged`` flag, not only the message, reflects the last inner
    IRLS solve. With ``least_squares`` wrapped to report failure, the result
    must not claim success."""
    import types

    import dtfit.methods._lsi as _lsi_mod

    x, y, _ = exp_data
    # noise so the IRLS residual scale is non-zero
    y = y + 0.05 * np.random.default_rng(0).standard_normal(y.size)

    real_ls = _lsi_mod.least_squares

    def failing_ls(*args, **kwargs):
        sol = real_ls(*args, **kwargs)
        return types.SimpleNamespace(x=sol.x, jac=sol.jac, fun=sol.fun,
                                     success=False, message="inner failed")

    monkeypatch.setattr(_lsi_mod, "least_squares", failing_ls)
    result = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5], robust=True)
    assert result.converged is False
    assert "inner failed" in result.message
