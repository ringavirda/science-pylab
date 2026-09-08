"""The numpy window-projection kernels match scipy/numpy to roundoff.

These are the retained bodies after the optional compiled backend was
removed; nothing in dtfit core exercises ``simpson_windows`` directly, so
this pins all three against their references.
"""

import numpy as np
import pytest
from scipy.integrate import simpson

from dtfit._core import _kernels as K


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 16, 17, 64, 65])
def test_simpson_windows_matches_scipy(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(0.0, 5.0, n))
    y = rng.normal(size=n)
    got = K.simpson_windows(y, x, np.array([0]), np.array([n]))[0]
    assert abs(got - simpson(y=y, x=x)) < 1e-12


def test_simpson_windows_rows_and_single_match_scipy():
    x = np.linspace(0.0, 10.0, 101)
    Y = np.vstack([np.sin(x), np.cos(2 * x), x**2, np.exp(-x)])
    starts = np.array([0, 30, 60])
    stops = np.array([30, 60, 101])
    ref = np.array(
        [[simpson(y=row[s:e], x=x[s:e]) for s, e in zip(starts, stops)]
         for row in Y]
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
