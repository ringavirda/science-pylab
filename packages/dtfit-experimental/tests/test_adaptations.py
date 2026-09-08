"""Experimental EAC/LSI adaptations, each on a case where the truth is known.

The ``#n`` labels are the adaptation index from the ``dtfit_experimental``
docstring. The map-reduce estimators PartitionedLSI and PartitionedEAC live
in ``dtfit_experimental.scale``; covered in ``test_scale_partitioned.py``.
"""

import numpy as np
import pytest

from dtfit_experimental import (
    fit_lsi_basis,
    fit_joint,
    boosted_fit,
)
from sklearn.metrics import r2_score


@pytest.fixture
def exp_stream():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 600)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    return t, y, (1.0, 0.9)


@pytest.fixture
def sine():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    return t, y, (2.0, 1.5)


# #2 pluggable basis
def test_fourier_basis_lsi_recovers_sine(sine):
    t, y, (A, w) = sine
    r = fit_lsi_basis(t, y, "A*sin(w*x)", "x", basis="fourier", order=8,
                      bounds=[(0.1, 5), (0.2, 3)])
    assert abs(r.coeffs[0] - A) < 0.2
    assert abs(r.coeffs[1] - w) < 0.1


def test_basis_dispatch_legendre_matches_exp(exp_stream):
    t, y, (a, b) = exp_stream
    r = fit_lsi_basis(t, y, "a*exp(b*t)", "t", basis="legendre", order=6,
                      p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - a) < 0.2 and abs(r.coeffs[1] - b) < 0.2


def test_unknown_basis_raises():
    with pytest.raises(ValueError):
        fit_lsi_basis(np.linspace(0, 1, 20), np.ones(20), "a*x", "x",
                      basis="nope")


# #3, the ensemble, is retired: the robust image covers its case, gated by
# tests/validation/test_outlier_robustness.py in the stable suite.


# #4 joint multi-channel
def test_joint_shares_frequency_across_channels():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 300)
    chans = [(t, A * np.sin(1.2 * t) + rng.normal(0, 0.05, t.size))
             for A in (1.0, 2.0, 3.0)]
    j = fit_joint(chans, "A*sin(w*x)", "x", shared=["w"], n_windows=6,
                  p0_shared=[1.0], p0_private=[1.0])
    assert abs(j.shared["w"] - 1.2) < 0.1
    amps = sorted(p["A"] for p in j.private)
    np.testing.assert_allclose(amps, [1.0, 2.0, 3.0], atol=0.2)


# #5 boosting
def test_boosted_fit_composite_improves_fit():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 3, 300)
    y = 0.5 * t + 0.3 * np.exp(0.5 * t) + rng.normal(0, 0.03, t.size)
    bm = boosted_fit(t, y, [
        dict(expr="a*x", var="x", method="lsi", p0=[1.0]),
        dict(expr="c*exp(d*x)", var="x", method="lsi", p0=[0.3, 0.5]),
    ])
    assert r2_score(y, bm.predict(t)) > 0.95
