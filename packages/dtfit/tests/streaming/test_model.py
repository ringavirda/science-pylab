"""The compiled model behind the streaming filter, and the coasting
functions."""

import numpy as np
import pytest

from dtfit.streaming import coast
from dtfit.streaming._model import CompiledModel


def test_symbolic_model_compiles_names_and_derivatives():
    m = CompiledModel("A*sin(w*t)", "t")
    assert m.names == ["A", "w"] and m.symbolic and not m.has_regressors
    t = np.linspace(0, 1, 5)
    p = np.array([2.0, 3.0])
    np.testing.assert_allclose(m.eval(t, None, p), 2.0 * np.sin(3.0 * t))
    J = m.jacobian(t, None, p)
    assert J.shape == (5, 2)
    np.testing.assert_allclose(J[:, 0], np.sin(3.0 * t))
    np.testing.assert_allclose(J[:, 1], 2.0 * t * np.cos(3.0 * t))
    np.testing.assert_allclose(m.dfdt(t, *p), 6.0 * np.cos(3.0 * t))


def test_constant_model_broadcasts_to_the_window():
    m = CompiledModel("c0", "t")
    out = m.eval(np.linspace(0, 1, 4), None, np.array([1.5]))
    assert out.shape == (4,) and np.all(out == 1.5)
    assert m.jacobian(
        np.linspace(0, 1, 4), None, np.array([1.5])
    ).shape == (4, 1)


def test_regressor_model_splits_drift_and_regressor_parts():
    m = CompiledModel("c0 + c1*t + S", "t", regressors="S")
    assert m.names == ["c0", "c1"] and m.regressors == ["S"]
    t = np.array([0.0, 1.0])
    cols = [np.array([5.0, 7.0])]
    np.testing.assert_allclose(
        m.eval(t, cols, np.array([1.0, 2.0])), [6.0, 10.0]
    )
    assert m.reg_tuple({"S": 3.0}) == (3.0,)
    assert m.reg_tuple([3.0]) == (3.0,)
    with pytest.raises(ValueError):
        m.reg_tuple(None)
    with pytest.raises(ValueError):
        m.reg_tuple([1.0, 2.0])


def test_callable_model_has_no_time_derivatives():
    def f(t, w, A):
        return A * np.sin(w * t)

    m = CompiledModel(f, "t")
    assert m.names == ["w", "A"] and not m.symbolic
    assert m.dfdt is None and m.f_reg is None
    J = m.jacobian(np.linspace(0, 1, 6), None, np.array([1.5, 2.0]))
    assert J.shape == (6, 2)
    with pytest.raises(ValueError, match="regressors"):
        CompiledModel(f, "t", regressors="S")


def test_coast_reduces_to_predict_in_support_and_is_linear_past_it():
    m = CompiledModel("c0 + c1*t + c2*t**2 + c3*t**3", "t")
    p = np.array([2.0, 3.0, 0.0, -0.1])
    anchor = 4.0
    xin = np.array([3.0, 4.0])
    np.testing.assert_allclose(
        coast.coast(m, p, xin, anchor), coast.predict(m, p, xin)
    )
    gap = anchor + np.arange(1, 21) * 0.1
    c1 = coast.coast(m, p, gap, anchor, order=1)
    assert np.allclose(np.diff(c1, 2), 0.0, atol=1e-9)
    assert not np.allclose(
        np.diff(coast.predict(m, p, gap), 2), 0.0, atol=1e-9
    )
    assert coast.coast(m, p, gap, None).shape == gap.shape


def test_coast_cov_grows_with_the_gap_and_matches_predict_cov_in_support():
    m = CompiledModel("c0 + c1*t", "t")
    p = np.array([1.0, 0.5])
    P = np.diag([0.01, 0.04])
    anchor = 2.0
    xin = np.array([1.0, 2.0])
    np.testing.assert_allclose(
        coast.coast_cov(m, p, P, xin, anchor),
        coast.predict_cov(m, p, P, xin),
    )
    gap = np.array([3.0, 5.0, 9.0])
    v = coast.coast_cov(m, p, P, gap, anchor)
    assert np.all(np.diff(v) > 0)
    np.testing.assert_allclose(
        coast.predict_cov(m, p, P, np.array([3.0])),
        [0.01 + 0.04 * 9.0],
    )
