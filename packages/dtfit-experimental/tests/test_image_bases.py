"""The experimental bases on dtfit's image interface: contract and recovery.

Each is a dtfit.image.Basis subclass fit through dtfit.fit(basis=instance);
the recovery smokes pin that the image path works, the natural-target
efficiency numbers live in the module docstrings (measurement, not a gate).
"""

import numpy as np
import pytest

from dtfit.image import Original, fit
from dtfit.image.bases import Basis
from dtfit_experimental import ChebyshevBasis, FourierBasis, LaguerreBasis


def test_n_coef_and_evaluate_shape():
    u = np.linspace(-1.0, 1.0, 50)
    assert FourierBasis(5).n_coef == 11
    assert ChebyshevBasis(8).n_coef == 9
    assert LaguerreBasis(8).n_coef == 9
    assert FourierBasis(5).evaluate(u).shape == (50, 11)
    assert ChebyshevBasis(8).evaluate(u).shape == (50, 9)
    assert LaguerreBasis(8).evaluate(u).shape == (50, 9)


def test_subclass_and_equality_roundtrip():
    for b in (FourierBasis(5), ChebyshevBasis(8), LaguerreBasis(8)):
        assert isinstance(b, Basis)
        assert b.to_dict()["order"] == b.order
        assert b == type(b)(b.order)


def test_fourier_recovers_a_periodic_model():
    x = np.linspace(0.0, 8.0, 400)
    truth = dict(a=0.5, A=2.0, w=2 * np.pi * 4 / 8.0, p=0.4)
    y = (truth["a"] + truth["A"] * np.sin(truth["w"] * x + truth["p"])
         + 0.05 * np.random.default_rng(0).standard_normal(x.size))
    r = fit("a + A*sin(w*t + p)", Original(x, y), "t",
            basis=FourierBasis(5),
            p0={"a": 0.0, "A": 1.0, "w": truth["w"], "p": 0.0})
    assert r.params["A"] == pytest.approx(2.0, abs=0.1)
    assert r.params["w"] == pytest.approx(truth["w"], abs=0.05)


def test_chebyshev_recovers_a_smooth_model():
    x = np.linspace(0.0, 1.0, 400)
    y = 2.0 * np.exp(-1.5 * x) + 0.3 + \
        0.02 * np.random.default_rng(0).standard_normal(x.size)
    r = fit("a*exp(-b*t) + c", Original(x, y), "t",
            basis=ChebyshevBasis(16),
            p0={"a": 1.0, "b": 1.0, "c": 0.0})
    assert r.params["b"] == pytest.approx(1.5, abs=0.1)


def test_laguerre_recovers_a_decaying_model():
    x = np.linspace(0.0, 6.0, 400)
    y = 3.0 * np.exp(-1.2 * x) + 0.2 + \
        0.03 * np.random.default_rng(0).standard_normal(x.size)
    r = fit("a*exp(-b*t) + c", Original(x, y), "t",
            basis=LaguerreBasis(8),
            p0={"a": 1.0, "b": 1.0, "c": 0.0})
    assert r.params["b"] == pytest.approx(1.2, abs=0.1)
