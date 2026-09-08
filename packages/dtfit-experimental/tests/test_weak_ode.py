"""Weak-form ODE identification: the operators read derivatives from the data,
and each linearized law recovers its constants without an ODE solve or a p0."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from dtfit_experimental.weak_ode import (
    fit_logistic,
    fit_lotka_volterra_prey,
    fit_michaelis_menten,
    weak_operators,
)


def test_weak_operators_read_the_first_derivative():
    # I1(g) projects g' against the test functions; check it against a
    # function with a known derivative (g = sin, g' = cos) to a tight
    # tolerance on a fine grid.
    t = np.linspace(0.0, 2.0 * np.pi, 2000)
    i0, i1, _ = weak_operators(t, n_test=8)
    got = i1(np.sin(t))
    want = i0(np.cos(t))
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_weak_operators_read_the_second_derivative():
    t = np.linspace(0.0, 2.0 * np.pi, 2000)
    i0, _, i2 = weak_operators(t, n_test=8)
    got = i2(np.sin(t))          # projects g''
    want = i0(-np.sin(t))        # g'' = -sin
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_logistic_recovers_r_and_k():
    r, k, y0 = 1.4, 5.0, 0.4
    t = np.linspace(0.0, 8.0, 500)
    y = solve_ivp(lambda _t, yy: r * yy * (1 - yy / k), (0, 8), [y0],
                  t_eval=t, rtol=1e-9).y[0]
    y = y + 0.01 * y0 * np.random.default_rng(0).standard_normal(t.size)
    p = fit_logistic(t, y)
    assert p["r"] == pytest.approx(r, rel=0.06)
    assert p["K"] == pytest.approx(k, rel=0.06)


def test_michaelis_menten_recovers_vm_and_km():
    vm, km, y0 = 1.2, 0.8, 4.0
    t = np.linspace(0.0, 8.0, 400)
    y = solve_ivp(lambda _t, yy: -vm * yy / (km + yy), (0, 8), [y0],
                  t_eval=t, rtol=1e-9).y[0]
    y = y + 0.01 * y0 * np.random.default_rng(1).standard_normal(t.size)
    p = fit_michaelis_menten(t, y)
    assert p["Vm"] == pytest.approx(vm, rel=0.08)
    assert p["Km"] == pytest.approx(km, rel=0.15)   # Km is the softer constant


def test_lotka_volterra_recovers_rates_from_prey_alone():
    al, be, de, ga = 1.1, 0.4, 0.1, 0.4
    t = np.linspace(0.0, 30.0, 1500)
    s = solve_ivp(
        lambda _t, z: [al * z[0] - be * z[0] * z[1],
                       de * z[0] * z[1] - ga * z[1]],
        (0, 30), [10.0, 5.0], t_eval=t, rtol=1e-10, atol=1e-12,
    )
    x = s.y[0] * (1 + 0.005 * np.random.default_rng(2).standard_normal(t.size))
    p = fit_lotka_volterra_prey(t, x)
    assert p["alpha"] == pytest.approx(al, rel=0.10)
    assert p["gamma"] == pytest.approx(ga, rel=0.10)
    assert p["delta"] == pytest.approx(de, rel=0.15)
    assert "beta" not in p   # structurally unidentifiable from the prey alone
