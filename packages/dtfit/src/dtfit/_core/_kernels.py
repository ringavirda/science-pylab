"""Numpy/scipy kernels for the window-projection inner loops.

The optimizers and the streaming filters run two small inner loops many
times: composite-Simpson window integrals and Gauss-Legendre spectral
projections. These are their numpy/scipy implementations.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson


def _as_idx(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.intp)


def simpson_windows(
    y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Composite-Simpson integral of ``y(x)`` over each index window.

    Window ``k`` spans the half-open index range ``[starts[k], stops[k])`` of
    the shared samples ``(x, y)``. Returns ``len(starts)`` areas.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    starts = _as_idx(starts)
    stops = _as_idx(stops)
    return np.array(
        [simpson(y=y[s:e], x=x[s:e]) for s, e in zip(starts, stops)],
        dtype=np.float64,
    )


def simpson_windows_rows(
    Y: np.ndarray, x: np.ndarray, starts: np.ndarray, stops: np.ndarray
) -> np.ndarray:
    """Per-row composite-Simpson integral of a 2-D ``Y`` over the windows.

    ``Y`` has shape ``(nrows, len(x))``; returns ``(nrows, len(starts))``
    areas. Integrates a stack of parameter sensitivities (Jacobian rows) in
    one call.
    """
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    starts = _as_idx(starts)
    stops = _as_idx(stops)
    return np.array(
        [
            [simpson(y=row[s:e], x=x[s:e]) for s, e in zip(starts, stops)]
            for row in Y
        ],
        dtype=np.float64,
    )


def legendre_project(
    fv: np.ndarray, qw: np.ndarray, legvander: np.ndarray, norm: np.ndarray
) -> np.ndarray:
    """Fused Gauss-Legendre spectral projection.

    Computes ``norm * ((qw * fv) @ legvander)``, the order-``k`` Legendre
    coefficients of a function sampled at the quadrature nodes. ``legvander``
    is ``(nq, k)``; ``fv`` and ``qw`` are ``(nq,)``; ``norm`` is ``(k,)``.
    """
    fv = np.ascontiguousarray(fv, dtype=np.float64)
    qw = np.ascontiguousarray(qw, dtype=np.float64)
    legvander = np.ascontiguousarray(legvander, dtype=np.float64)
    norm = np.ascontiguousarray(norm, dtype=np.float64)
    return norm * ((qw * fv) @ legvander)
