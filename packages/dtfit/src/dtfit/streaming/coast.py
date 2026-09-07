"""Prediction and dead-reckoning from a streaming filter's state."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._model import CompiledModel


def predict(
    model: CompiledModel, p: np.ndarray, x: Any, regressors: Any = None
) -> np.ndarray:
    """The model at ``p`` on ``x``, shaped like ``x``. With external
    regressors, ``regressors`` supplies their value(s) at ``x`` (a
    ``{name: array-or-scalar}`` mapping broadcast to ``x``'s shape, or an
    ``(len(x), n_reg)`` array).

    Raises:
        ValueError: a regressor model without ``regressors``.
    """
    xa = np.asarray(x, dtype=float)
    if not model.has_regressors:
        return np.broadcast_to(np.asarray(model.f(xa, *p), float), xa.shape)
    cols = model.predict_cols(xa, regressors)
    return np.asarray(model.f(xa, *cols, *p), dtype=float)


def predict_cov(
    model: CompiledModel,
    p: np.ndarray,
    P: np.ndarray,
    x: Any,
    regressors: Any = None,
) -> np.ndarray:
    """``J(x)^T P J(x)`` with ``J = d f / d p``: the variance the state
    covariance ``P`` implies for the model output at ``x``, shaped like
    ``x`` and clipped at zero. ``P`` is the filter's gain state; a
    calibrated parameter covariance is ``ImageFilter.result().cov``.

    Raises:
        ValueError: a regressor model without ``regressors``.
    """
    xa = np.asarray(x, dtype=float)
    if model.has_regressors:
        cols = model.predict_cols(xa, regressors)
        jac_cols = [
            np.broadcast_to(np.asarray(j(xa, *cols, *p), float), xa.shape)
            for j in model.jac
        ]
    else:
        jac_cols = [
            np.broadcast_to(np.asarray(j(xa, *p), float), xa.shape)
            for j in model.jac
        ]
    jac = np.stack(jac_cols, axis=-1)
    var = np.einsum("...i,ij,...j->...", jac, P, jac)
    return np.clip(var, 0.0, None)


def coast(
    model: CompiledModel,
    p: np.ndarray,
    x: Any,
    anchor: float | None,
    *,
    order: int = 1,
    regressors: Any = None,
) -> np.ndarray:
    """Dead-reckon past the window from ``anchor``, the last in-window time:
    ``f(a) + f'(a) (x - a)`` at ``order=1`` (constant velocity), plus
    ``f''(a) (x - a)^2 / 2`` at ``order=2``. At and before the anchor, and
    with ``anchor=None``, it is :func:`predict`. A regressor model is
    split: the regressor part is evaluated at the supplied future
    ``regressors``, the time-only drift is dead-reckoned at a frozen rate.

    Raises:
        NotImplementedError: a callable model (no time derivatives), or a
            regressor model without ``regressors``.
    """
    if not model.symbolic:
        raise NotImplementedError(
            "coast() needs a symbolic model for its time-derivatives; "
            "construct the filter from an expression string to use "
            "coasting"
        )
    xa = np.asarray(x, dtype=float)
    if model.has_regressors:
        if regressors is None or model.f_reg is None:
            raise NotImplementedError(
                "coast() on a model with external regressors needs the "
                "future regressor value(s): pass regressors=<value(s) "
                "at x> (e.g. IMU-propagated), or use predict()."
            )
        cols = model.predict_cols(xa, regressors)
        in_support = np.asarray(model.f(xa, *cols, *p), dtype=float)
        if anchor is None:
            return in_support
        a_arr = np.asarray(float(anchor), dtype=float)
        dt = xa - float(a_arr)
        reg_part = np.asarray(model.f_reg(xa, *cols, *p), dtype=float)
        assert model.f_drift is not None and model.f_drift_dt is not None
        fd = float(np.asarray(model.f_drift(a_arr, *p)))
        vd = float(np.asarray(model.f_drift_dt(a_arr, *p)))
        drift = fd + vd * dt
        if order >= 2:
            assert model.f_drift_d2t is not None
            ad = float(np.asarray(model.f_drift_d2t(a_arr, *p)))
            drift = drift + 0.5 * ad * dt * dt
        return np.where(dt > 0.0, reg_part + drift, in_support)
    in_support = np.broadcast_to(np.asarray(model.f(xa, *p), float), xa.shape)
    if anchor is None:
        return np.asarray(in_support, dtype=float)
    a = float(anchor)
    dt = xa - a
    a_arr = np.asarray(a, dtype=float)
    assert model.dfdt is not None and model.d2fdt2 is not None
    f_a = float(np.asarray(model.f(a_arr, *p)))
    v_a = float(np.asarray(model.dfdt(a_arr, *p)))
    coasted = f_a + v_a * dt
    if order >= 2:
        acc_a = float(np.asarray(model.d2fdt2(a_arr, *p)))
        coasted = coasted + 0.5 * acc_a * dt * dt
    return np.where(dt > 0.0, coasted, in_support)


def coast_cov(
    model: CompiledModel,
    p: np.ndarray,
    P: np.ndarray,
    x: Any,
    anchor: float | None,
    *,
    order: int = 1,
) -> np.ndarray:
    """The variance :func:`coast` implies from ``P`` through the coast
    Jacobian ``df/dp(a) + d f'/dp(a) dt [+ d f''/dp(a) dt^2 / 2]``, so the
    band widens with the gap; :func:`predict_cov` at and before the
    anchor. ``P`` is the filter's gain state; a calibrated covariance is
    ``ImageFilter.result().cov``.

    Raises:
        NotImplementedError: a callable model, or a model with external
            regressors.
    """
    if not model.symbolic:
        raise NotImplementedError(
            "coast_cov() needs a symbolic model for its time-derivatives; "
            "construct the filter from an expression string to use "
            "coasting"
        )
    if model.has_regressors:
        raise NotImplementedError(
            "coast_cov() is undefined for models with external "
            "regressors; use predict_cov()."
        )
    xa = np.asarray(x, dtype=float)
    base_cov = predict_cov(model, p, P, xa)
    if anchor is None:
        return base_cov
    a = float(anchor)
    dt = xa - a
    a_arr = np.asarray(a, dtype=float)
    jcols = []
    for k in range(len(p)):
        jp = float(np.asarray(model.jac[k](a_arr, *p)))
        jpt = float(np.asarray(model.dfdt_jac[k](a_arr, *p)))
        col = jp + jpt * dt
        if order >= 2:
            jptt = float(np.asarray(model.d2fdt2_jac[k](a_arr, *p)))
            col = col + 0.5 * jptt * dt * dt
        jcols.append(np.broadcast_to(np.asarray(col, dtype=float), xa.shape))
    jac = np.stack(jcols, axis=-1)
    var = np.einsum("...i,ij,...j->...", jac, P, jac)
    return np.where(dt > 0.0, np.clip(var, 0.0, None), base_cov)
