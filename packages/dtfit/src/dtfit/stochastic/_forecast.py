"""Regime-appropriate forecasters and rolling-origin backtest model selection
for :func:`dtfit.stochastic.fit_stochastic`. Each ``(train, h) -> array``
candidate (random walk, drift, mean reversion, local-slope or curvature trend,
multi-harmonic seasonal continuation) is backtested, and the RMSE-optimal one
wins; the random walk is the default when nothing beats it."""

from __future__ import annotations

import warnings
from typing import Any, Callable, cast

import numpy as np

from dtfit.methods import fit_lsi
from ._stats import _seasonal_design, _fit_seasonal

__all__ = ["FORECASTERS"]


def _bind_forecaster(
    fn: Callable[..., Any], y: np.ndarray
) -> Callable[[int], np.ndarray]:
    """Bind a ``(train, h) -> array`` forecaster to the fitted series, yielding
    the ``h -> array`` closure stored on the model."""
    def _f(h: int) -> np.ndarray:
        return np.asarray(fn(y, h), dtype=float)
    return _f


def _fc_rw(tr: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, float(tr[-1]))


def _fc_drift(tr: np.ndarray, h: int) -> np.ndarray:
    d = float(np.mean(np.diff(tr))) if tr.size > 1 else 0.0
    return float(tr[-1]) + d * np.arange(1, h + 1, dtype=float)


def _fc_meanrev(tr: np.ndarray, h: int) -> np.ndarray:
    """AR(1) mean reversion toward the sample mean: the level-series model for
    a persistent but stationary series such as interest rates or spreads."""
    mu = float(tr.mean())
    x = tr - mu
    if x.size < 3:
        return np.full(h, float(tr[-1]))
    phi = float((x[:-1] @ x[1:]) / (x[:-1] @ x[:-1] + 1e-12))
    phi = float(np.clip(phi, -0.999, 0.999))
    return mu + phi ** np.arange(1, h + 1, dtype=float) * (float(tr[-1]) - mu)


def _local_slope(tr: np.ndarray, period: float | None = None) -> float:
    """Slope of a line through the recent portion, so that an accelerating
    trend like CO2 is extrapolated at its current rate instead of the flatter
    global average."""
    n = tr.size
    k = n if period is None else min(n, max(2 * int(period), n // 3, 10))
    k = int(max(10, min(k, n)))
    seg = tr[-k:]
    return float(np.polyfit(np.arange(seg.size, dtype=float), seg, 1)[0])


def _trend_fit_extrap(series: np.ndarray, h: int,
                      period: float | None = None) -> np.ndarray:
    """Curvature-aware trend extrapolation from an LSI quadratic fit over the
    recent window, evaluated at the future indices and anchored at the fitted
    last value.

    A local straight line mis-extrapolates a curved trend badly, as on an
    accelerating series like CO2; a quadratic LSI trend captures the
    curvature. Anchoring at the fitted value keeps the forecast unbiased,
    because pinning it to the noisy last observation would carry that
    residual forward as a constant offset and leave the forecast sitting
    stably above or below the actual. If the quadratic would run away past a
    sane multiple of the in-sample range, a fitted straight line is used
    instead.
    """
    steps = np.arange(1, h + 1, dtype=float)
    n = series.size
    k = n if period is None else min(n, max(2 * int(period) + 2, n // 3, 60))
    k = int(max(30, min(k, n)))
    seg = series[-k:]
    z = np.arange(seg.size, dtype=float)
    zf = (seg.size - 1) + steps
    if seg.size >= 12:
        try:
            r = fit_lsi(z, seg, "a0 + a1*z + a2*z**2", "z",
                        k_star=2, filter_data=True)
            a0, a1, a2 = (float(r.coeffs[0]), float(r.coeffs[1]),
                          float(r.coeffs[2]))
            pred = a0 + a1 * zf + a2 * zf ** 2
            base_last = a0 + a1 * (seg.size - 1) + a2 * (seg.size - 1) ** 2
            span = float(seg.max() - seg.min()) + 1e-9
            if np.max(np.abs(pred - base_last)) <= 6.0 * span:
                return pred
        except Exception:
            pass
    a1l, a0l = np.polyfit(z, seg, 1)             # straight-line fallback
    return a0l + a1l * zf


def _trend_anchored_extrap(tr: np.ndarray, h: int,
                           period: float | None = None) -> np.ndarray:
    """Curvature-aware trend extrapolation anchored at the raw last value: the
    increment from a recent-window quadratic LSI fit added onto ``tr[-1]``.

    Anchoring this way is right when the series is clean enough that the last
    value is itself a good level estimate, as with CO2. On a noisy series it
    carries that value's residual forward as a constant offset;
    :func:`_trend_fit_extrap` is the unbiased alternative, and both are
    offered so the backtest can pick per series.
    """
    steps = np.arange(1, h + 1, dtype=float)
    n = tr.size
    k = n if period is None else min(n, max(2 * int(period) + 2, n // 3, 60))
    k = int(max(30, min(k, n)))
    seg = tr[-k:]
    if seg.size >= 12:
        z = np.arange(seg.size, dtype=float)
        try:
            r = fit_lsi(z, seg, "a0 + a1*z + a2*z**2", "z",
                        k_star=2, filter_data=True)
            a0, a1, a2 = (float(r.coeffs[0]), float(r.coeffs[1]),
                          float(r.coeffs[2]))
            last = float(seg.size - 1)
            zf = last + steps
            pred = ((a0 + a1 * zf + a2 * zf ** 2)
                    - (a0 + a1 * last + a2 * last ** 2) + float(tr[-1]))
            span = float(seg.max() - seg.min()) + 1e-9
            if np.max(np.abs(pred - float(tr[-1]))) <= 6.0 * span:
                return pred
        except Exception:
            pass
    return float(tr[-1]) + _local_slope(tr, period) * steps


def _fc_trend(tr: np.ndarray, h: int) -> np.ndarray:
    """Linear local-slope trend, a straight line from the last value. The
    conservative choice: it holds up on a spurious level-shift "trend" like
    the Nile, which a curved fit would over-extrapolate."""
    return float(tr[-1]) + _local_slope(tr) * np.arange(1, h + 1, dtype=float)


def _make_seasonal_fc(period: float, max_harmonics: int, with_trend: bool,
                      window_periods: float | None = None):
    """Seasonal forecaster: the fitted trend plus multi-harmonic seasonal model
    extrapolated forward, anchored at the fitted trend rather than the raw
    last value.

    The harmonics are fit over the whole series, shrinking the amplitude
    toward what is reliably predictable. That is RMSE-optimal where the
    cycle's amplitude or phase drift, as in the sunspot record. The series is
    then de-seasonalised and its trend extrapolated by
    :func:`_trend_fit_extrap`, giving ``trend(t_future) + seasonal(t_future)``
    with none of the last-value bias described there. The bias bites hardest
    on a seasonal series, since the last point often falls at an extreme of
    the cycle. ``window_periods`` restores the visual amplitude of the current
    cycles at some cost in RMSE."""
    def fc(tr: np.ndarray, h: int) -> np.ndarray:
        m = tr.size
        ts = np.arange(m, dtype=float)
        # seasonal harmonics on the linearly de-trended full series
        if with_trend:
            a1g, a0g = np.polyfit(ts, tr, 1)
            base = a0g + a1g * ts
        else:
            base = np.full(m, float(tr.mean()))
        win = (m if window_periods is None
               else int(min(m, max(int(window_periods * period),
                                    2 * int(period) + 2))))
        seg_t, seg_d = ts[-win:], (tr - base)[-win:]
        k, coef = _fit_seasonal(seg_t - seg_t[0], seg_d, period, max_harmonics)
        phase0 = seg_t[0]
        seas_in = _seasonal_design(ts - phase0, period, k) @ coef
        tf = (m - 1) + np.arange(1, h + 1, dtype=float)
        seas_f = _seasonal_design(tf - phase0, period, k) @ coef
        # De-seasonalise, then extrapolate the trend as a fitted curve with no
        # raw last-value anchor. A flat-trend regime just holds the recent
        # de-seasonalised level.
        deseas = tr - seas_in
        if with_trend:
            trend_f = _trend_fit_extrap(deseas, h, period)
        else:
            lvl_win = min(m, max(2 * int(period) + 2, 30))
            trend_f = np.full(h, float(np.mean(deseas[-lvl_win:])))
        return trend_f + seas_f
    return fc


def _make_seasonal_fc_anchored(period: float, max_harmonics: int,
                               with_trend: bool):
    """Seasonal forecaster anchored at the raw last value: the multi-harmonic
    seasonal pattern continued from ``tr[-1]`` plus a curvature-aware anchored
    trend.

    Best on a clean trend+seasonal series like CO2, where the last sample
    carries little noise. It is offered alongside :func:`_make_seasonal_fc` so
    the backtest keeps whichever forecasts better; the fitted one wins on a
    noisy series, where anchoring at the last value would leave the forecast
    sitting above or below it."""
    def fc(tr: np.ndarray, h: int) -> np.ndarray:
        m = tr.size
        ts = np.arange(m, dtype=float)
        if with_trend:
            a1g, a0g = np.polyfit(ts, tr, 1)
            detr = tr - (a0g + a1g * ts)
        else:
            detr = tr - tr.mean()
        k, coef = _fit_seasonal(ts, detr, period, max_harmonics)
        last = float(m - 1)
        seas_last = float((_seasonal_design(np.array([last]), period, k) @ coef)[0])
        tf = last + np.arange(1, h + 1, dtype=float)
        seas_f = _seasonal_design(tf, period, k) @ coef
        trend_f = ((_trend_anchored_extrap(tr, h, period) - float(tr[-1]))
                   if with_trend else 0.0)
        return float(tr[-1]) + trend_f + (seas_f - seas_last)
    return fc


def _select_forecaster(
    y: np.ndarray,
    candidates: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]],
    *,
    max_h: int = 30,
    folds: int = 5,
    margin: float = 0.98,
) -> tuple[str, Callable[[np.ndarray, int], np.ndarray]]:
    """Rolling-origin backtest each candidate and choose the best non-RW one if
    its mean RMSE is within ``margin`` of the random walk's, else the random
    walk.

    ``margin < 1`` is the strict, RMSE-optimal setting: a non-RW model must
    beat RW by the margin. Level regimes use it and tilt toward RW for
    parsimony. ``margin > 1`` is lenient, keeping a detected structure such as
    a seasonal cycle or a trend unless it is clearly worse than RW. A per-fold
    backtest under-rates a phase-sensitive cyclical forecast that is in fact
    RMSE-competitive over the full record, and a detected cycle should not be
    flattened to a line on that account.

    A series too short to backtest (``n <= 50``) cannot arbitrate between
    candidates. The first one, the random walk in every built-in set, comes
    back with a :class:`UserWarning` and its name suffixed
    ``" (short-series fallback)"`` so the fallback stays visible on the fitted
    model. A candidate that raises during the backtest is warned about and
    scored ``inf`` RMSE for that fold, losing it the selection."""
    n = y.size
    if len(candidates) == 1:
        return candidates[0]
    if n <= 50:
        name, fn = candidates[0]
        warnings.warn(
            f"series too short to backtest-select a forecaster (n={n} <= 50); "
            f"falling back to {name}", UserWarning, stacklevel=2)
        return f"{name} (short-series fallback)", fn
    hb = int(max(5, min(n // 6, max_h)))
    scores: dict[str, list[float]] = {name: [] for name, _ in candidates}
    for k in range(folds):
        end = n - k * hb
        if end - hb < 40:
            break
        train, test = y[:end - hb], y[end - hb:end]
        for name, fn in candidates:
            try:
                fc = np.asarray(fn(train, hb), dtype=float)
                scores[name].append(float(np.sqrt(np.mean((fc - test) ** 2))))
            except Exception as exc:
                warnings.warn(
                    f"forecaster candidate {name!r} failed during backtest: "
                    f"{exc}", UserWarning, stacklevel=2)
                scores[name].append(float("inf"))
    means = {name: (float(np.mean(s)) if s else float("inf"))
             for name, s in scores.items()}
    rw = means.get("random walk", float("inf"))
    non_rw = {nm: v for nm, v in means.items() if nm != "random walk"}
    best = "random walk"
    if non_rw:
        struct = min(non_rw, key=lambda nm: non_rw[nm])
        if non_rw[struct] <= margin * rw:
            best = struct
    return next(c for c in candidates if c[0] == best)


# Built-in names a caller can force or compose into a candidate set.
FORECASTERS = ("random walk", "drift", "mean-reversion", "trend",
               "seasonal", "trend+seasonal")


def _build_named_forecaster(name: str, per: float, max_harmonics: int):
    table = {"random walk": _fc_rw, "drift": _fc_drift,
             "mean-reversion": _fc_meanrev, "trend": _fc_trend}
    if name in table:
        return name, table[name]
    if name in ("seasonal", "trend+seasonal"):
        if not np.isfinite(per):
            raise ValueError(
                f"forecaster {name!r} needs a seasonal period; pass period=...")
        return name, _make_seasonal_fc(per, max_harmonics, name == "trend+seasonal")
    raise ValueError(f"unknown forecaster {name!r}; choose from {FORECASTERS} "
                     "or pass a callable / list")


def _resolve_forecaster(forecaster, auto_candidates, *, per, max_harmonics, y,
                        margin=0.98):
    """Turn the user's ``forecaster=`` argument into a chosen ``(name, fn)``.

    ``"auto"`` backtest-selects over the auto candidate set at the given
    ``margin``; a name forces that built-in; a callable ``(train, h) -> array``
    is used directly; a list of names, callables or ``(name, fn)`` pairs is a
    custom candidate set to backtest-select among.
    """
    if forecaster is None or forecaster == "auto":
        return _select_forecaster(y, auto_candidates, margin=margin)
    if callable(forecaster):
        return "custom", forecaster
    if isinstance(forecaster, str):
        return _build_named_forecaster(forecaster, per, max_harmonics)
    cands: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]] = []
    for item in forecaster:
        if callable(item):
            cands.append((getattr(item, "__name__", "custom"),
                          cast("Callable[[np.ndarray, int], np.ndarray]", item)))
        elif isinstance(item, tuple) and len(item) == 2:
            cands.append((str(item[0]),
                          cast("Callable[[np.ndarray, int], np.ndarray]", item[1])))
        elif isinstance(item, str):
            cands.append(_build_named_forecaster(item, per, max_harmonics))
        else:
            raise TypeError(f"bad forecaster candidate: {item!r}")
    if not cands:
        raise ValueError("forecaster list is empty")
    return _select_forecaster(y, cands, margin=margin)
