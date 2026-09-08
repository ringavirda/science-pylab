"""The structured forecasting router.

:func:`auto_forecast` fits and extrapolates. Saturating growth goes to a
logistic, a detected cycle to a joint linear-plus-seasonal fit, anything else
to a quadratic level, behind two guards: persist when the fit cannot beat a
random walk on a held-out training tail, and drop a runaway quadratic to
linear. Each candidate is a :func:`dtfit.fit_lsi` call. A near-random-walk
series falls back to persistence.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from dtfit.types import FittingResult
from dtfit._signal import dominant_period
from dtfit._pandas import (
    HAS_PANDAS,
    as_series,
    capture_index,
    extend_index,
    to_1d_array,
)
from dtfit.image.fit import fit_lsi, fft_frequency_seed


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _looks_like_growth(y: np.ndarray) -> bool:
    if np.any(y <= 0):
        return False
    d = np.diff(y)
    if d.size == 0:
        return False
    monotone = np.mean(np.sign(d) == np.sign(d[np.argmax(np.abs(d))])) > 0.9
    return bool(monotone and (abs(y[-1] / y[0]) > 3 or abs(y[0] / y[-1]) > 3))


def _auto_model(y: np.ndarray, seasonal: bool, season_strength: float) -> str:
    """Pick a model class, with no per-series tuning.

    Saturating growth goes to logistic, a detected cycle to linear+seasonal,
    anything else to a quadratic level that the divergence guard catches if it
    runs away.
    """
    if _looks_like_growth(y):  # already guarantees strictly positive y
        return "logistic"
    _, strength = dominant_period(y)
    return "linear_seasonal" if (seasonal and strength > season_strength) else "poly"


def _poly_seed(y: np.ndarray, t: np.ndarray, deg: int) -> list[float]:
    pc = np.polyfit(t, y, deg)  # numpy: highest power first
    return [float(pc[deg - i]) for i in range(deg + 1)]


def _fit_model(model: str, t: np.ndarray, y: np.ndarray, t_all: np.ndarray,
               period: float | None) -> tuple[np.ndarray, FittingResult]:
    """Fit one model class and evaluate it over ``t_all`` (train + future x).

    Returns the values over ``t_all`` and the :class:`FittingResult` behind
    them, so :func:`auto_forecast` can attach the fit and a prediction band as
    provenance without re-fitting.
    """
    xspan = float(t[-1] - t[0]) or 1.0
    if model == "logistic":
        ylast = float(y[-1])
        # The growth rate scales with the time span. Bracket the seed instead
        # of using a fixed box. A fixed box excludes gentle slopes and
        # lets the global search latch onto a near-vertical step: a degenerate
        # fit that matches in sample and then extrapolates to overflow.
        k_seed = 6.0 / xspan
        r = fit_lsi(
            t, y, "L/(1 + exp(-k*(x - x0)))", "x",
            p0={"L": ylast * 1.5, "k": k_seed, "x0": float(t[0]) + xspan},
            bounds={"L": (ylast * 0.8, ylast * 12.0),
                    "k": (0.2 * k_seed, 8.0 * k_seed),
                    "x0": (float(t[0]), float(t[0]) + 2.5 * xspan)},
            k_star=6)
        return np.asarray(r.model(t_all), dtype=float), r
    if model == "linear":
        r = fit_lsi(t, y, "a0 + a1*x", "x", p0=_poly_seed(y, t, 1))
        return np.asarray(r.model(t_all), dtype=float), r
    if model == "linear_seasonal":
        s = _poly_seed(y, t, 1)
        dx = float(np.mean(np.diff(t))) or 1.0
        if period is not None and period > 0:
            w0 = 2 * np.pi / (period * dx)
        else:
            w0 = fft_frequency_seed(t, y) or (2 * np.pi / xspan)
        amp = float(np.std(y)) + 1e-3
        expr = "a0 + a1*x + A*sin(w*x + p)"
        r = fit_lsi(
            t, y, expr, "x",
            p0={"a0": s[0], "a1": s[1], "A": amp, "p": 0.0, "w": w0},
            bounds={"a0": (-1e6, 1e6), "a1": (-1e6, 1e6),
                    "A": (1e-3, 5 * amp), "p": (-np.pi, np.pi),
                    "w": (0.7 * w0, 1.3 * w0)},
            freq_param="w")
        return np.asarray(r.model(t_all), dtype=float), r
    # poly (quadratic level)
    r = fit_lsi(t, y, "a0 + a1*x + a2*x**2", "x", p0=_poly_seed(y, t, 2))
    return np.asarray(r.model(t_all), dtype=float), r


def _diverges(pred: np.ndarray, y: np.ndarray, k: float = 5.0) -> bool:
    rng = float(np.ptp(y)) or 1.0
    lo, hi = float(y.min()) - k * rng, float(y.max()) + k * rng
    return not np.all((pred >= lo) & (pred <= hi))


def _no_structure(model: str, t: np.ndarray, y: np.ndarray,
                  period: float | None, factor: float = 8.0) -> bool:
    """True when the model cannot get near naive persistence on a held-out
    tail of the training data, which is the near-random-walk signature."""
    n = y.size
    if n < 24:
        return False
    k = int(n * 0.8)
    try:
        sp_tail = _fit_model(model, t[:k], y[:k], t, period)[0][k:n]
        s_rmse = _rmse(y[k:], sp_tail)
    except Exception:
        return True
    p_rmse = _rmse(y[k:], np.full(n - k, y[k - 1])) + 1e-12
    return bool(np.isfinite(s_rmse) and s_rmse > factor * p_rmse)


class ForecastResult(np.ndarray):
    """The forecast values, plus where they came from.

    It subclasses :class:`numpy.ndarray`, which makes it the length-``horizon``
    forecast itself: indexing, ``.shape``, ``len``, arithmetic and the usual
    NumPy functions all work on it directly. It also carries the fit behind
    the numbers; inspecting that costs no second call.

    Attributes:
        model_name: The model that produced the forecast, carrying any
            fallback provenance with it. ``"logistic"`` for a clean fit,
            ``"linear (poly diverged)"`` when the divergence guard dropped a
            runaway quadratic, ``"linear (logistic failed)"`` when the primary
            fit raised, or ``"random_walk"`` / ``"persistence (...)"`` on the
            persistence paths.
        result: The :class:`FittingResult` behind the forecast; ``None`` on
            the persistence and random-walk paths.
        std_band: A length-``horizon`` 1-sigma prediction band (delta method)
            when the fit exposed a covariance and the propagation succeeded,
            ``None`` otherwise. Named ``std_band`` rather than ``std`` so it
            does not shadow ``numpy.ndarray.std``.
        index: The length-``horizon`` future pandas index continuing the ``x``
            passed to :func:`auto_forecast`; ``None`` when ``x`` was not a
            pandas object with an inferable step, or when pandas is absent.
            The forecast values themselves do not depend on it. See
            :meth:`to_series`.
    """

    model_name: str
    result: FittingResult | None
    std_band: np.ndarray | None
    index: Any

    def __new__(
        cls,
        values: np.ndarray | Sequence[float],
        *,
        model_name: str = "",
        result: FittingResult | None = None,
        std_band: np.ndarray | None = None,
        index: Any = None,
    ) -> ForecastResult:
        obj = np.asarray(values, dtype=float).view(cls)
        obj.model_name = model_name
        obj.result = result
        obj.std_band = None if std_band is None else np.asarray(std_band, dtype=float)
        obj.index = index
        return obj

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        # Runs on every construction path: view, slice, ufunc. The scalar
        # provenance carries forward unconditionally, but std_band and index
        # are length-horizon and only align while the array keeps that length,
        # so a slice or a reduction drops them instead of carrying a
        # misaligned band and a wrong-length index.
        if obj is None:
            return
        self.model_name = getattr(obj, "model_name", "")
        self.result = getattr(obj, "result", None)
        src_std = getattr(obj, "std_band", None)
        src_index = getattr(obj, "index", None)
        n = self.shape[0] if self.ndim == 1 else -1
        self.std_band = (
            src_std if src_std is not None and len(src_std) == n else None
        )
        self.index = (
            src_index if src_index is not None and len(src_index) == n else None
        )

    def to_series(self) -> Any:
        """The forecast values as a pandas ``Series`` on :attr:`index`.

        :func:`auto_forecast` always returns the ndarray subclass; this is the
        opt-in pandas view of the same values.

        Returns:
            A pandas ``Series`` of the forecast values, indexed by
            :attr:`index`.

        Raises:
            ValueError: When :attr:`index` is ``None`` and there is no future
                index to align to. That happens when ``x`` was not a pandas
                object with an extendable index, or pandas is not installed.
        """
        if not HAS_PANDAS:
            raise ValueError(
                "ForecastResult.to_series() requires pandas, which is not installed"
            )
        if self.index is None:
            raise ValueError(
                "ForecastResult.to_series() needs a future index, but .index is "
                "None: auto_forecast was not given a pandas Series/DataFrame whose "
                "index has an inferable frequency/step. Pass x as such an object."
            )
        return as_series(np.asarray(self), self.index)


def _persist(
    y: np.ndarray, horizon: int, model_name: str, index: Any = None
) -> ForecastResult:
    """A flat ``y[-1]`` persistence forecast, tagged with its provenance."""
    return ForecastResult(
        np.full(horizon, float(y[-1])), model_name=model_name, result=None,
        std_band=None, index=index,
    )


def _forecast_std(
    result: FittingResult | None, future: np.ndarray
) -> np.ndarray | None:
    """Best-effort 1-sigma band at the future grid, from the fit covariance.

    Returns ``None`` rather than raising when there is no covariance, when the
    delta-method propagation fails, or when the band comes back non-finite or
    the wrong length.
    """
    if result is None or result.cov is None:
        return None
    try:
        _, std = result.predict(future, return_std=True)
    except Exception:
        # The forecast values are still valid; we just report no uncertainty.
        return None
    std = np.asarray(std, dtype=float)
    if std.shape != future.shape or not np.all(np.isfinite(std)):
        return None
    return std


def auto_forecast(
    x: np.ndarray,
    y: np.ndarray,
    horizon: int,
    *,
    model: str = "auto",
    period: float | None = None,
    seasonal: bool = True,
    season_strength: float = 0.05,
) -> ForecastResult:
    """Structured fit-then-extrapolate forecast.

    Routes the model class, applies the no-structure and divergence guards,
    then extrapolates ``horizon`` steps past ``x`` on its uniform grid.

    Args:
        x, y: The observed series; ``x`` near-uniformly sampled.
        horizon: Number of future steps to forecast.
        model: ``"auto"`` to route by structure, or one of ``"logistic"``,
            ``"linear"``, ``"poly"``, ``"linear_seasonal"``, ``"random_walk"``.
        period: Known seasonal period, in samples, for the seasonal fit.
        seasonal: Whether to consider a seasonal model under ``"auto"``.
        season_strength: Minimum cycle strength to pick a seasonal model.

    Returns:
        A :class:`ForecastResult`: the length-``horizon`` forecast as an
        ndarray, carrying ``.model_name``, ``.result``, ``.std_band`` and
        ``.index``.

    ``x`` and ``y`` accept pandas ``Series`` and single-column ``DataFrame``
    inputs; an ndarray or list input gives a bit-identical forecast. When
    ``x``'s index is extendable, a ``DatetimeIndex`` with an inferable
    frequency or an integer-stepped index, the result carries a future
    ``.index`` and :meth:`ForecastResult.to_series` gives the pandas view.
    """
    allowed = {"auto", "logistic", "linear", "poly", "linear_seasonal", "random_walk"}
    if model not in allowed:
        raise ValueError(
            f"unknown model {model!r}; expected one of {sorted(allowed)}"
        )
    x_index = capture_index(x)
    x = to_1d_array(x, "x")
    y = to_1d_array(y, "y")
    fut_index = extend_index(x_index, horizon)
    if horizon <= 0:
        return ForecastResult(np.empty(0), model_name=model, result=None,
                              std_band=None, index=fut_index)
    dx = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    future = x[-1] + dx * np.arange(1, horizon + 1)
    t_all = np.concatenate([x, future])

    chosen = _auto_model(y, seasonal, season_strength) if model == "auto" else model

    # Persistence paths: an explicit random walk, or a structured model that
    # cannot beat naive persistence on a held-out training tail. An explicit
    # random walk never runs the no-structure probe.
    if chosen == "random_walk":
        return _persist(y, horizon, "random_walk", index=fut_index)
    if _no_structure(chosen, x, y, period):
        return _persist(
            y, horizon, f"persistence ({chosen} no structure)", index=fut_index
        )

    result: FittingResult | None = None
    produced = chosen
    try:
        pred, result = _fit_model(chosen, x, y, t_all, period)
    except Exception as exc:
        warnings.warn(
            f"auto_forecast: {chosen} fit failed ({exc}); falling back to linear",
            UserWarning,
            stacklevel=2,
        )
        pred, result = _fit_model("linear", x, y, t_all, period)
        produced = f"linear ({chosen} failed)"

    if chosen == "poly" and _diverges(pred, y):
        try:
            pred, result = _fit_model("linear", x, y, t_all, period)
            produced = "linear (poly diverged)"
        except Exception as exc:
            warnings.warn(
                f"auto_forecast: linear fit failed ({exc}); "
                "falling back to persistence",
                UserWarning,
                stacklevel=2,
            )
            return _persist(
                y, horizon, "persistence (linear failed)", index=fut_index
            )

    return ForecastResult(
        pred[x.size:],
        model_name=produced,
        result=result,
        std_band=_forecast_std(result, future),
        index=fut_index,
    )
