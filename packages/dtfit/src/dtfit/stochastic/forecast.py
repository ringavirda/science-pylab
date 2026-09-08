"""Regime-appropriate forecasters and rolling-origin backtest selection.

Every built-in candidate is a function ``(image, h) -> array``: it takes its
parameters from a :class:`SecondOrderImage` of the training stretch and its
anchor from that image's carry."""

from __future__ import annotations

import warnings
from typing import Any, Callable, cast

import numpy as np

from .image import SecondOrderImage
from .estimators import seasonal_series

__all__ = ["FORECASTERS"]

Forecaster = Callable[[SecondOrderImage, int], np.ndarray]

# Built-in names a caller can force or compose into a candidate set.
FORECASTERS = ("random walk", "drift", "mean-reversion", "trend",
               "seasonal", "trend+seasonal")


def bind_forecaster(
    fn: Forecaster, img: SecondOrderImage
) -> Callable[[int], np.ndarray]:
    """Bind an ``(image, h) -> array`` forecaster to the fitted image."""
    def _f(h: int) -> np.ndarray:
        return np.asarray(fn(img, h), dtype=float)
    return _f


def fc_rw(img: SecondOrderImage, h: int) -> np.ndarray:
    """The last observation held flat."""
    return np.full(h, img.last())


def fc_drift(img: SecondOrderImage, h: int) -> np.ndarray:
    """The last observation continued at the average increment."""
    d = (img.last() - img.first()) / max(img.n - 1, 1)
    return img.last() + d * np.arange(1, h + 1, dtype=float)


def fc_meanrev(img: SecondOrderImage, h: int) -> np.ndarray:
    """AR(1) reversion of the last observation toward the sample mean."""
    mu = img.mean()
    g = img.acov()
    phi = float(np.clip(g[1] / g[0], -0.999, 0.999)) if g[0] > 0 else 0.0
    return mu + phi ** np.arange(1, h + 1, dtype=float) * (img.last() - mu)


def fc_trend(img: SecondOrderImage, h: int) -> np.ndarray:
    """The least-squares slope continued from the last observation."""
    slope, _ = img._trend_index()
    return img.last() + slope * np.arange(1, h + 1, dtype=float)


def make_seasonal_fc(
    period: float, max_harmonics: int, with_trend: bool
) -> Forecaster:
    """Trend plus multi-harmonic seasonal continuation anchored at the fitted
    trend, the unbiased choice on a noisy seasonal series.

    The harmonics are referenced to the global sample index, the one ``t0``
    counts from, so a block image forecasts the phase the whole record
    would."""
    def fc(img: SecondOrderImage, h: int) -> np.ndarray:
        seas = img.seasonal(max_harmonics=max_harmonics, freq=1.0 / period)
        idx = float(img.n - 1) + np.arange(1, h + 1, dtype=float)
        cyc = seasonal_series(img.t0 + idx, seas["freq"], seas["coef"])
        if with_trend:
            slope, icpt = img._trend_index()
            base = icpt + slope * (img.t0 + idx)
        else:
            base = np.full(h, img.mean())
        return base + cyc
    return fc


def make_seasonal_fc_anchored(
    period: float, max_harmonics: int, with_trend: bool
) -> Forecaster:
    """The same continuation anchored at the last observation, the choice on a
    clean series whose last sample is itself a good level estimate.

    The harmonics are referenced to the global sample index, as in
    :func:`make_seasonal_fc`."""
    def fc(img: SecondOrderImage, h: int) -> np.ndarray:
        seas = img.seasonal(max_harmonics=max_harmonics, freq=1.0 / period)
        last = float(img.n - 1)
        idx = last + np.arange(1, h + 1, dtype=float)
        cyc = seasonal_series(img.t0 + idx, seas["freq"], seas["coef"])
        cyc0 = float(seasonal_series(
            np.array([img.t0 + last]), seas["freq"], seas["coef"])[0])
        if with_trend:
            slope, _ = img._trend_index()
            base = slope * np.arange(1, h + 1, dtype=float)
        else:
            base = np.zeros(h)
        return img.last() + base + (cyc - cyc0)
    return fc


def select_forecaster(
    y: np.ndarray | None,
    img: SecondOrderImage,
    candidates: list[tuple[str, Forecaster]],
    *,
    max_h: int = 30,
    folds: int = 5,
    margin: float = 0.98,
) -> tuple[str, Forecaster]:
    """Rolling-origin backtest each candidate and keep the best non-random-walk
    one whose mean RMSE is within ``margin`` of the random walk's.

    ``y`` is the training series; ``None`` (a fit given an image alone) has no
    sub-series to backtest on, so the first non-random-walk candidate is taken
    with a :class:`UserWarning` and its name suffixed ``" (no backtest)"``. A
    series too short to backtest (``n <= 50``) falls back to the first
    candidate with the suffix ``" (short-series fallback)"``. A candidate that
    raises during a fold is warned about and scored infinite for it.
    """
    if len(candidates) == 1:
        return candidates[0]
    if y is None:
        name, fn = candidates[-1] if len(candidates) > 1 else candidates[0]
        warnings.warn(
            "a fit from a SecondOrderImage cannot backtest-select a "
            f"forecaster; taking {name}", UserWarning, stacklevel=2)
        return f"{name} (no backtest)", fn
    n = y.size
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
        t_img = SecondOrderImage.of(
            train, lag=min(img.lag, train.size - 1), nfreq=img.nfreq
        )
        for name, fn in candidates:
            try:
                fc = np.asarray(fn(t_img, hb), dtype=float)
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


def build_named_forecaster(
    name: str, per: float, max_harmonics: int
) -> tuple[str, Forecaster]:
    """One built-in candidate by name.

    Raises:
        ValueError: an unknown name, or a seasonal name without a period.
    """
    table = {"random walk": fc_rw, "drift": fc_drift,
             "mean-reversion": fc_meanrev, "trend": fc_trend}
    if name in table:
        return name, table[name]
    if name in ("seasonal", "trend+seasonal"):
        if not np.isfinite(per):
            raise ValueError(
                f"forecaster {name!r} needs a seasonal period; "
                "pass period=...")
        return name, make_seasonal_fc(
            per, max_harmonics, name == "trend+seasonal")
    raise ValueError(f"unknown forecaster {name!r}; choose from {FORECASTERS} "
                     "or pass a callable / list")


def _wrap_callable(
    fn: Callable[[np.ndarray, int], np.ndarray], y: np.ndarray | None
) -> Forecaster:
    """Adapt a user ``(train, h) -> array`` callable to the ``(image, h)``
    contract by holding the training array it was fitted on.

    Raises:
        TypeError: the fit was given an image alone, so no training array
            exists.
    """
    if y is None:
        raise TypeError(
            "a callable forecaster needs the series; fit from an Original or "
            "an array rather than a SecondOrderImage"
        )

    def _f(img: SecondOrderImage, h: int) -> np.ndarray:
        return np.asarray(fn(y[: img.n], h), dtype=float)
    return _f


def resolve_forecaster(
    forecaster: Any,
    auto_candidates: list[tuple[str, Forecaster]],
    *,
    per: float,
    max_harmonics: int,
    img: SecondOrderImage,
    y: np.ndarray | None,
    margin: float = 0.98,
) -> tuple[str, Forecaster]:
    """Turn the caller's ``forecaster=`` argument into a chosen
    ``(name, fn)``.

    Raises:
        ValueError: an unknown built-in name or an empty candidate list.
        TypeError: a candidate that is neither a name, a callable nor a
            ``(name, callable)`` pair.
    """
    if forecaster is None or forecaster == "auto":
        return select_forecaster(y, img, auto_candidates, margin=margin)
    if callable(forecaster):
        return "custom", _wrap_callable(
            cast("Callable[[np.ndarray, int], np.ndarray]", forecaster), y)
    if isinstance(forecaster, str):
        return build_named_forecaster(forecaster, per, max_harmonics)
    cands: list[tuple[str, Forecaster]] = []
    for item in forecaster:
        if callable(item):
            cands.append((getattr(item, "__name__", "custom"),
                          _wrap_callable(item, y)))
        elif isinstance(item, tuple) and len(item) == 2:
            cands.append((str(item[0]), _wrap_callable(item[1], y)))
        elif isinstance(item, str):
            cands.append(build_named_forecaster(item, per, max_harmonics))
        else:
            raise TypeError(f"bad forecaster candidate: {item!r}")
    if not cands:
        raise ValueError("forecaster list is empty")
    return select_forecaster(y, img, cands, margin=margin)
