"""Weak-form ODE identification on the image.

A rate law ``y' = f(y; theta)`` that is nonlinear in its constants ``theta``
becomes LINEAR in ``theta`` in the weak form: multiply the ODE by a test
function ``phi`` that vanishes with its derivatives at the window endpoints,
integrate over the window, and move each derivative of ``y`` onto ``phi`` by
parts (``int y' phi = -int y phi'``). No ODE is solved and no starting guess is
needed; the derivatives of the noisy ``y`` are never taken pointwise, only
projected. Two structural tricks widen the reach:

* a rational rate law linearizes by clearing its denominator (Michaelis-Menten:
  ``(Km + y) y' = -Vm y``);
* a two-state system observed in one state linearizes by eliminating the
  unobserved state (Lotka-Volterra from the prey alone).

The projection is exactly the image's derivative-basis machinery: ``I0``, ``I1``
and ``I2`` project ``g``, ``g'`` and ``g''`` onto the same test-function family.
This is the batch weak form, the first experimental cut. The GLS weighting of
the weak residuals and the instrumental-variable step that close the remaining
1.1-2x accuracy gap to NLLS (measured on the design prototype) are follow-ups;
``beta`` in the prey-only Lotka-Volterra is structurally unidentifiable and is
not returned.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.polynomial import legendre as _L

WeakOps = tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]


def weak_operators(
    t: np.ndarray, n_test: int = 12, order: int = 2
) -> WeakOps:
    """The weak-form projection operators ``(I0, I1, I2)`` on the grid ``t``.

    Each maps a sampled function ``g(t)`` to its ``n_test``-vector of
    projections against the test functions
    ``phi_k(u) = (1 - u**2)**order * P_k(u)``, with ``P_k`` the ``k``-th
    Legendre polynomial and ``u`` the domain mapped to ``[-1, 1]``. The window
    factor vanishes with ``order`` derivatives at ``u = +/-1``, so the boundary
    terms of the integration by parts drop:

    * ``I0(g) = integral g phi`` -- ``g`` against ``phi``;
    * ``I1(g) = integral g' phi = -integral g phi'`` -- one derivative of ``g``,
      read from the data without differentiating it;
    * ``I2(g) = integral g'' phi = integral g phi''`` -- two derivatives.

    Args:
        t: sample times, shape ``(n,)``, strictly increasing; may be
            non-uniform (trapezoid weights come from :func:`numpy.gradient`).
        n_test: number of test functions (Legendre degrees ``0..n_test-1``).
            More test functions give more weak equations; ``12`` suffices for
            the demonstrated laws, ``24`` for the second-order elimination.
        order: vanishing order of the window factor at the endpoints; ``2``
            lets ``I2`` integrate a second derivative by parts cleanly.

    Returns:
        ``(I0, I1, I2)``, three callables each taking ``g`` of shape ``(n,)``
        and returning a vector of shape ``(n_test,)``.
    """
    t = np.asarray(t, dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    u = 2.0 * (t - t0) / (t1 - t0) - 1.0
    du = 2.0 / (t1 - t0)
    w = np.gradient(t)
    eye = np.eye(n_test)
    P = _L.legvander(u, n_test - 1)
    dP = np.column_stack(
        [_L.legval(u, _L.legder(eye[k])) for k in range(n_test)]
    )
    d2P = np.column_stack(
        [_L.legval(u, _L.legder(eye[k], 2)) for k in range(n_test)]
    )
    b = (1 - u ** 2) ** order
    db = -2 * order * u * (1 - u ** 2) ** (order - 1)
    d2b = (
        -2 * order * (1 - u ** 2) ** (order - 1)
        + 4 * order * (order - 1) * u ** 2 * (1 - u ** 2) ** (order - 2)
    )
    phi = b[:, None] * P
    dphi = (db[:, None] * P + b[:, None] * dP) * du
    d2phi = (
        d2b[:, None] * P + 2 * db[:, None] * dP + b[:, None] * d2P
    ) * du * du

    def i0(g: np.ndarray) -> np.ndarray:
        return (w * np.asarray(g, dtype=float)) @ phi

    def i1(g: np.ndarray) -> np.ndarray:
        return -(w * np.asarray(g, dtype=float)) @ dphi

    def i2(g: np.ndarray) -> np.ndarray:
        return (w * np.asarray(g, dtype=float)) @ d2phi

    return i0, i1, i2


def fit_logistic(
    t: np.ndarray, y: np.ndarray, n_test: int = 12
) -> dict[str, float]:
    """Fit ``y' = r y (1 - y / K)`` in the weak form.

    Expanded, ``y' = r y - (r/K) y**2`` is linear in ``[r, r/K]``, so
    ``I1(y) = r I0(y) - (r/K) I0(y**2)``.

    Returns:
        ``{"r": growth rate, "K": carrying capacity}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, _ = weak_operators(t, n_test=n_test)
    a = np.column_stack([i0(y), -i0(y * y)])
    r, r_over_k = np.linalg.lstsq(a, i1(y), rcond=None)[0]
    return {"r": float(r), "K": float(r / r_over_k)}


def fit_michaelis_menten(
    t: np.ndarray, y: np.ndarray, n_test: int = 12
) -> dict[str, float]:
    """Fit the Michaelis-Menten decay ``y' = -Vm y / (Km + y)`` in the weak
    form.

    Clearing the denominator gives ``Km y' + y y' = -Vm y``; with
    ``y y' = (y**2)'/2`` the weak form is
    ``Km I1(y) + Vm I0(y) = -I1(y**2)/2``, linear in ``[Km, Vm]``.

    Returns:
        ``{"Vm": max rate, "Km": half-saturation constant}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, _ = weak_operators(t, n_test=n_test)
    a = np.column_stack([i1(y), i0(y)])
    rhs = -0.5 * i1(y * y)
    km, vm = np.linalg.lstsq(a, rhs, rcond=None)[0]
    return {"Vm": float(vm), "Km": float(km)}


def fit_damped_oscillator(
    t: np.ndarray, y: np.ndarray, n_test: int = 12
) -> dict[str, float]:
    """Fit a damped oscillation ``y'' + 2 zeta omega y' + omega**2 y = 0``.

    The homogeneous second-order law is already linear in its constants, so
    the weak form is ``I2(y) + c1 I1(y) + c2 I0(y) = 0`` with ``c1 = 2 zeta
    omega`` and ``c2 = omega**2``; solving ``[I1(y), I0(y)] [c1, c2] = -I2(y)``
    gives the natural frequency and damping ratio without differentiating the
    noisy ``y`` and without a starting guess.

    Returns:
        ``{"omega": natural frequency, "zeta": damping ratio}``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    i0, i1, i2 = weak_operators(t, n_test=n_test)
    a = np.column_stack([i1(y), i0(y)])
    c1, c2 = np.linalg.lstsq(a, -i2(y), rcond=None)[0]
    omega = float(np.sqrt(abs(c2)))
    return {"omega": omega, "zeta": float(c1 / (2 * omega)) if omega else float("nan")}


def fit_lotka_volterra_prey(
    t: np.ndarray, x: np.ndarray, n_test: int = 24
) -> dict[str, float]:
    """Recover the Lotka-Volterra rates from the PREY series ``x`` alone.

    The predator is eliminated through ``y = (alpha - (ln x)') / beta``, which
    turns the prey equation into the second-order law
    ``-(ln x)'' = delta alpha x - delta x' - gamma alpha + gamma (ln x)'``.
    Its weak form
    ``-I2(ln x) = (delta alpha) I0(x) - delta I1(x)
                  - (gamma alpha) I0(1) + gamma I1(ln x)``
    is linear in ``[delta alpha, delta, gamma alpha, gamma]``, from which
    ``alpha``, ``gamma`` and ``delta`` follow. ``beta`` scales the unobserved
    predator and is structurally unidentifiable from the prey alone, so it is
    not returned.

    Args:
        x: prey series, shape ``(n,)``, strictly positive.

    Returns:
        ``{"alpha": prey growth, "gamma": predator death,
        "delta": predator gain}``.
    """
    t = np.asarray(t, dtype=float)
    x = np.clip(np.asarray(x, dtype=float), 1e-12, None)
    i0, i1, i2 = weak_operators(t, n_test=n_test)
    lx = np.log(x)
    a = np.column_stack(
        [i0(x), -i1(x), -i0(np.ones_like(x)), i1(lx)]
    )
    c = np.linalg.lstsq(a, -i2(lx), rcond=None)[0]
    delta_alpha, delta, _gamma_alpha, gamma = c
    return {
        "alpha": float(delta_alpha / delta),
        "gamma": float(gamma),
        "delta": float(delta),
    }
