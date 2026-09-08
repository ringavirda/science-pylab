"""Adaptation #4: joint shared-parameter multi-channel fit.

Channels often share structure: one frequency across the x/y/z axes of a
trajectory, one growth rate across regions, one time constant across a MIMO
plant's outputs. Fitting them independently throws that coupling away.
:func:`fit_joint` stacks every channel's EAC area equations into a single
system carrying shared parameters, estimated from all channels at once, plus
per-channel private ones, then solves the lot in a single least-squares pass.
Each shared unknown ends up with more equations behind it: that is the
observability gain over independent fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
import sympy as sp
from scipy.optimize import least_squares, differential_evolution, minimize

from dtfit._symbolic import model_params


@dataclass
class JointResult:
    """Result of a joint multi-channel fit."""

    shared: dict[str, float]
    private: list[dict[str, float]]
    expr: str
    var: str

    def predict(self, channel: int, x: np.ndarray) -> np.ndarray:
        t = sp.Symbol(self.var)
        f_sym = sp.sympify(self.expr)
        subs = {sp.Symbol(k): v for k, v in self.shared.items()}
        subs.update({sp.Symbol(k): v for k, v in self.private[channel].items()})
        # sympy's stub types subs() as an iterable of pairs; a {sym: val} dict
        # is accepted at runtime.
        model = sp.lambdify(t, f_sym.subs(subs), "numpy")  # pyright: ignore[reportCallIssue, reportArgumentType]
        v = model(np.asarray(x, dtype=float))
        return np.full(np.shape(x), float(v)) if np.ndim(v) == 0 else np.asarray(v, float)


def fit_joint(
    channels: Sequence[tuple[np.ndarray, np.ndarray]],
    expr: str,
    var: str,
    shared: Sequence[str],
    *,
    n_windows: int = 6,
    active_ratio: float = 1.0,
    p0_shared: Sequence[float] | None = None,
    p0_private: Sequence[float] | None = None,
    bounds_shared: Sequence[tuple[float, float]] | None = None,
    bounds_private: Sequence[tuple[float, float]] | None = None,
) -> JointResult:
    """Jointly fit ``expr`` across channels with shared + private parameters.

    Args:
        channels: List of ``(x, y)`` arrays, one per channel.
        expr, var: Model expression and main variable.
        shared: Names of parameters tied across all channels; the rest are
            estimated per channel.
        n_windows: Area windows per channel (equal-areas equations).
        active_ratio: Leading fraction of each channel used for windows.
        p0_shared / p0_private: Optional initial guesses, ones by default.
        bounds_shared / bounds_private: Optional ``(min, max)`` per shared and
            per private parameter. Pass both to put a global search
            (differential evolution) ahead of the local refine, as a multimodal
            shared parameter such as a free frequency requires; pass only one
            and the plain local solve runs instead.

    Returns:
        JointResult with the shared and per-channel private estimates.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = [str(p) for p in model_params(f_sym, t)]
    shared = list(shared)
    private = [p for p in params if p not in shared]
    n_ch = len(channels)
    ns, npv = len(shared), len(private)

    f_func = sp.lambdify((t, *[sp.Symbol(p) for p in params]), f_sym, "numpy")

    # Precompute per-channel windows + data areas.
    ch_data = []
    for x, y in channels:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        idx_max = max(int(x.size * active_ratio), n_windows + 1)
        edges = np.linspace(0, idx_max, n_windows + 1).astype(int)
        areas = np.array([
            np.trapezoid(y[edges[k]:edges[k + 1]], x[edges[k]:edges[k + 1]])
            if edges[k + 1] - edges[k] >= 2 else 0.0
            for k in range(n_windows)
        ])
        ch_data.append((x, edges, areas))

    def unpack(theta: np.ndarray):
        s = theta[:ns]
        priv = theta[ns:].reshape(n_ch, npv)
        return s, priv

    def channel_full_params(s: np.ndarray, pv: np.ndarray) -> list[float]:
        mp = dict(zip(shared, s))
        mp.update(dict(zip(private, pv)))
        return [mp[p] for p in params]

    def residual(theta: np.ndarray) -> np.ndarray:
        s, priv = unpack(theta)
        res = []
        for c, (x, edges, areas) in enumerate(ch_data):
            full = channel_full_params(s, priv[c])
            mv = np.asarray(f_func(x, *full), dtype=float)
            if mv.ndim == 0:
                mv = np.full_like(x, float(mv))
            m_areas = np.array([
                np.trapezoid(mv[edges[k]:edges[k + 1]], x[edges[k]:edges[k + 1]])
                if edges[k + 1] - edges[k] >= 2 else 0.0
                for k in range(n_windows)
            ])
            res.append(m_areas - areas)
        return np.concatenate(res)

    g_shared = np.ones(ns) if p0_shared is None else np.asarray(p0_shared, float)
    g_priv = (np.ones(npv) if p0_private is None else np.asarray(p0_private, float))
    theta0 = np.concatenate([g_shared, np.tile(g_priv, n_ch)])

    if bounds_shared is not None and bounds_private is not None:
        all_bounds = list(bounds_shared) + list(bounds_private) * n_ch

        def cost(theta: np.ndarray) -> float:
            r = residual(theta)
            return float(r @ r)

        res_g = cast(Any, differential_evolution)(
            cost, all_bounds, strategy="best1bin", popsize=12, seed=0,
            maxiter=200, tol=1e-7)
        res = minimize(cost, res_g.x, method="L-BFGS-B", bounds=all_bounds)
        s, priv = unpack(np.asarray(res.x, dtype=float))
    else:
        sol = least_squares(residual, theta0, method="lm")
        s, priv = unpack(np.asarray(sol.x, dtype=float))
    return JointResult(
        shared=dict(zip(shared, s)),
        private=[dict(zip(private, priv[c])) for c in range(n_ch)],
        expr=expr,
        var=var,
    )
