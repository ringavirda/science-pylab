"""Compiled-kernel parity: the C backend must match the pure-Python fallback.

The kernels in ``dtfit._core._kernels`` are checked against
``scipy.integrate.simpson`` and NumPy. ``fit_eac`` and ``fit_lsi`` are then
run twice, once with the native backend forced off. Both paths must agree to
within roundoff; that is the contract of the optional extension.
"""

import os

import numpy as np
import pytest
from scipy.integrate import simpson

import dtfit._core._kernels as K
from dtfit import fit_eac, fit_lsi

# The extension is optional and only built on the Linux CI job; tests that
# need it skip elsewhere. Where the build is expected, CI sets
# ``DTFIT_REQUIRE_NATIVE`` and a missing or broken build is a hard failure.
requires_native = pytest.mark.skipif(
    not K.HAVE_NATIVE, reason="compiled dtfit._core._native not built (optional on this platform)"
)


@pytest.fixture
def force_fallback(monkeypatch):
    """Run the body with the compiled backend disabled (pure-Python path)."""
    monkeypatch.setattr(K, "HAVE_NATIVE", False)
    yield


def test_native_is_built():
    if not os.environ.get("DTFIT_REQUIRE_NATIVE"):
        pytest.skip("native extension optional here; set DTFIT_REQUIRE_NATIVE to enforce")
    assert K.HAVE_NATIVE, "dtfit._core._native not built -- run python build_native.py"


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 16, 17, 64, 65])
def test_simpson_windows_matches_scipy(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(0.0, 5.0, n))
    y = rng.normal(size=n)
    got = K.simpson_windows(y, x, np.array([0]), np.array([n]))[0]
    assert abs(got - simpson(y=y, x=x)) < 1e-12


def test_simpson_windows_multi_and_rows_match_scipy():
    x = np.linspace(0.0, 10.0, 101)
    Y = np.vstack([np.sin(x), np.cos(2 * x), x**2, np.exp(-x)])
    starts = np.array([0, 30, 60])
    stops = np.array([30, 60, 101])

    ref = np.array(
        [[simpson(y=row[s:e], x=x[s:e]) for s, e in zip(starts, stops)] for row in Y]
    )
    rows = K.simpson_windows_rows(Y, x, starts, stops)
    assert np.allclose(rows, ref, atol=1e-12)
    single = K.simpson_windows(Y[0], x, starts, stops)
    assert np.allclose(single, ref[0], atol=1e-12)


def test_legendre_project_matches_numpy():
    nodes, w = np.polynomial.legendre.leggauss(16)
    V = np.polynomial.legendre.legvander(nodes, 5)
    norm = (2.0 * np.arange(6) + 1.0) / 2.0
    fv = np.exp(0.7 * nodes) + 0.3 * nodes**2
    got = K.legendre_project(fv, w, V, norm)
    assert np.allclose(got, norm * ((w * fv) @ V), atol=1e-13)


@requires_native
def test_fallback_kernels_match_native():
    x = np.linspace(0.0, 8.0, 97)
    Y = np.vstack([np.sin(x), x**2])
    starts, stops = np.array([0, 40]), np.array([40, 97])
    nodes, w = np.polynomial.legendre.leggauss(16)
    V = np.polynomial.legendre.legvander(nodes, 5)
    norm = (2.0 * np.arange(6) + 1.0) / 2.0
    fv = np.cos(nodes)

    native = (
        K.simpson_windows(Y[0], x, starts, stops),
        K.simpson_windows_rows(Y, x, starts, stops),
        K.legendre_project(fv, w, V, norm),
    )
    K.HAVE_NATIVE = False
    try:
        fb = (
            K.simpson_windows(Y[0], x, starts, stops),
            K.simpson_windows_rows(Y, x, starts, stops),
            K.legendre_project(fv, w, V, norm),
        )
    finally:
        K.HAVE_NATIVE = True
    for a, b in zip(native, fb):
        assert np.allclose(a, b, atol=1e-12)


def test_fit_eac_backend_agnostic(arctan_data, force_fallback):
    x, y, _ = arctan_data
    fb = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0]).coeffs
    K.HAVE_NATIVE = True  # monkeypatch puts the original back afterwards
    nat = fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0]).coeffs
    assert np.allclose(fb, nat, rtol=1e-9, atol=1e-9)


def test_fit_lsi_backend_agnostic(exp_data, force_fallback):
    x, y, _ = exp_data
    fb = fit_lsi(x, y, "a*exp(b*x)", "x").coeffs
    K.HAVE_NATIVE = True
    nat = fit_lsi(x, y, "a*exp(b*x)", "x").coeffs
    assert np.allclose(fb, nat, rtol=1e-9, atol=1e-9)
