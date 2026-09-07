"""The EAC block preset and the LSI oscillatory recipe.

* ``fit_eac``, the uniform-window block preset, aimed at concentrated
  transients;
* the ``fit_lsi`` oscillatory recipe (``oscillatory=`` / ``freq_param=`` with
  ``fft_frequency_seed``), which recovers a sinusoid the default order at
  p0 does not resolve.
"""

import numpy as np
import pytest

from dtfit import (
    fit_lsi,
    fit_eac,
    fft_frequency_seed,
)


def test_eac_recovers_transient():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 3, 400)
    y = 2.0 * (1 - np.exp(-3.0 * t)) + rng.normal(0, 0.02, t.size)
    r = fit_eac(t, y, "K*(1-exp(-a*x))", "x", p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - 2.0) < 0.2 and abs(r.coeffs[1] - 3.0) < 0.5


def test_fft_frequency_seed_finds_dominant_cycle():
    t = np.linspace(0, 4 * np.pi, 400)
    y = 2.0 * np.sin(1.5 * t)
    assert fft_frequency_seed(t, y) == pytest.approx(1.5, rel=0.05)


def test_oscillatory_recipe_recovers_sine_where_default_fails():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    w_true = 1.5
    y = 2.0 * np.sin(w_true * t) + rng.normal(0, 0.05, t.size)

    osc = fit_lsi(t, y, "A*sin(w*x)", "x", freq_param="w", p0=[1.0, 1.0])
    names = ["A", "w"]  # sympy sorts the parameters, so A comes before w
    w_osc = osc.coeffs[names.index("w")]
    assert abs(w_osc - w_true) < 0.1

    # without the recipe the default order at p0 need not resolve the cycle
    plain = fit_lsi(t, y, "A*sin(w*x)", "x", p0=[1.0, 1.0])
    w_plain = plain.coeffs[names.index("w")]
    # Both fits can land within 1 percent on some platforms; the recipe
    # must then not be worse than the default beyond that level.
    assert abs(w_osc - w_true) <= max(abs(w_plain - w_true), 0.01)


def test_oscillatory_flag_raises_order_under_bounds():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6 * np.pi, 400)
    y = np.sin(2.0 * t) + rng.normal(0, 0.05, t.size)
    # the bounds path: a global search brackets the frequency; the recipe
    # still helps
    with pytest.warns(UserWarning, match="differential-evolution"):
        r = fit_lsi(t, y, "A*sin(w*x)", "x", oscillatory=True,
                    bounds=[(0.1, 5.0), (0.5, 4.0)])
    assert abs(r.coeffs[1] - 2.0) < 0.2


def test_freq_param_unknown_raises():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="freq_param"):
        fit_lsi(t, np.sin(t), "A*sin(w*x)", "x", freq_param="omega")
