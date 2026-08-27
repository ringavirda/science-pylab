"""Native kernels stay numerically identical to the NumPy/SciPy fallback.

Guards the GIL-release refactor of dtfit._core._native: dropping the GIL around the
pure compute loops must not change any result. Also exercises concurrent calls
from multiple threads to catch a botched GIL handshake (which would deadlock or
corrupt output).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from scipy.integrate import simpson

from dtfit._core import _kernels

pytestmark = pytest.mark.skipif(
    not _kernels.HAVE_NATIVE, reason="compiled kernels not built"
)


def _data(seed=0, n=201):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 5, n))
    y = np.sin(x) + 0.1 * x
    return x, y


def test_simpson_windows_matches_scipy():
    from dtfit._core import _native

    x, y = _data()
    starts = np.array([0, 50, 100, 150], dtype=np.intp)
    stops = np.array([50, 100, 150, 201], dtype=np.intp)
    got = _native.simpson_windows(np.ascontiguousarray(y),
                                  np.ascontiguousarray(x), starts, stops)
    ref = np.array([simpson(y=y[s:e], x=x[s:e]) for s, e in zip(starts, stops)])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_simpson_windows_rows_matches_scipy():
    from dtfit._core import _native

    x, y = _data()
    Y = np.ascontiguousarray(np.vstack([y, np.cos(x), x ** 2]))
    starts = np.array([0, 70, 140], dtype=np.intp)
    stops = np.array([70, 140, 201], dtype=np.intp)
    got = _native.simpson_windows_rows(Y, np.ascontiguousarray(x), starts, stops)
    ref = np.array([[simpson(y=row[s:e], x=x[s:e])
                     for s, e in zip(starts, stops)] for row in Y])
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-11)


def test_legendre_project_matches_numpy():
    from dtfit._core import _native

    nodes, w = leggauss(16)
    V = legvander(nodes, 5)
    norm = (2 * np.arange(6) + 1) / 2.0
    fv = np.exp(0.3 * nodes)
    got = _native.legendre_project(np.ascontiguousarray(fv),
                                   np.ascontiguousarray(w),
                                   np.ascontiguousarray(V),
                                   np.ascontiguousarray(norm))
    ref = norm * ((w * fv) @ V)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


def test_kernels_are_thread_safe_under_concurrency():
    from dtfit._core import _native

    x, y = _data()
    starts = np.array([0, 50, 100, 150], dtype=np.intp)
    stops = np.array([50, 100, 150, 201], dtype=np.intp)
    ref = _native.simpson_windows(np.ascontiguousarray(y),
                                  np.ascontiguousarray(x), starts, stops)

    def call(_):
        return _native.simpson_windows(np.ascontiguousarray(y),
                                       np.ascontiguousarray(x), starts, stops)

    with ThreadPoolExecutor(max_workers=8) as ex:
        outs = list(ex.map(call, range(64)))
    for o in outs:
        np.testing.assert_allclose(o, ref, rtol=0, atol=1e-12)
