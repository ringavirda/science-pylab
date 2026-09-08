"""DSB (symbolic differential spectra balance) reference method."""

from typing import cast

import numpy as np
import pytest

from dtfit.reference import find_degree, fit_dsb
from dtfit._symbolic import taylor_coeffs
import sympy as sp


def _balance(x, y, expr, var, n_params, degree=None):
    """Fit ``expr`` by DSB from data: pick the polynomial degree by BIC but
    never below ``n_params - 1``, where the balance would be underdefined,
    fit that polynomial, then balance its Maclaurin spectrum."""
    deg = max(find_degree(x, y, method="bic"), n_params - 1, 1)
    deg = deg if degree is None else degree
    return fit_dsb(np.polyfit(x, y, deg)[::-1], expr, var)


def test_dsb_fits_an_additive_exponential(lint_exp_data):
    x, y = lint_exp_data
    res = _balance(x, y, "a0 + a1*x + a2*exp(a3*x)", "x", 4)
    assert len(res.coeffs) == 4
    pred = np.asarray(res.model(x), float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.8


def test_taylor_coeffs_match_known_series():
    t = sp.Symbol("x")
    coeffs = taylor_coeffs(cast(sp.Expr, sp.sympify("exp(x)")), t, 4)
    assert [sp.nsimplify(c) for c in coeffs] == [
        sp.Rational(1, sp.factorial(k)) for k in range(5)
    ]
    s = [float(c) for c in taylor_coeffs(cast(sp.Expr, sp.sympify("sin(x)")), t, 4)]
    assert np.allclose(s, [0, 1, 0, -1 / 6, 0])


def test_dsb_fits_models_without_handwritten_discretes():
    # No hand-written discrete rule exists for log or rational forms; the
    # generic Taylor balance covers them anyway.
    x = np.linspace(0.0, 1.2, 300)
    y = 0.2 + 1.5 * np.log(1 + 0.9 * x)
    res = _balance(x, y, "a0 + a1*log(1 + a2*x)", "x", 3, degree=6)
    pred = np.asarray(res.model(x), float)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert 1.0 - ss_res / ss_tot > 0.95


def test_dsb_underdetermined_balance_raises():
    # atan's even Maclaurin orders vanish, so a degree-2 polynomial cannot
    # identify three parameters. DSB has to say so instead of returning junk.
    coeffs_poly = np.array([1.0, 2.0, 0.0])  # only orders 0,1 constrain params
    with pytest.raises(ValueError, match="constrain"):
        fit_dsb(coeffs_poly, "a0 + a1*atan(a2*x)", "x")


def test_dsb_user_input_errors_are_value_errors():
    with pytest.raises(ValueError, match="no free parameters"):
        fit_dsb(np.array([1.0, 2.0]), "2*x + 1", "x")
    # fewer polynomial coefficients than parameters
    with pytest.raises(ValueError, match="underdefined"):
        fit_dsb(np.array([1.0]), "a0 + a1*x", "x")
    # explicit rank below the parameter count
    with pytest.raises(ValueError, match="underdefined"):
        fit_dsb(np.array([1.0, 2.0, 3.0]), "a0 + a1*x + a2*x**2", "x", rank=2)


def test_dsb_keeps_roots_with_a_zero_component():
    # The true offset is exactly 0, so the symbolic square balance solves to
    # (a0, a1) = (0, 2). A zero component is a legitimate root: filtering roots
    # on it would drop the fit to numeric least squares, and the message
    # asserted below would change.
    result = fit_dsb(np.array([0.0, 2.0]), "a0 + a1*x", "x")
    np.testing.assert_allclose(result.coeffs, [0.0, 2.0], atol=1e-12)
    assert result.message == "symbolic balance solved"
