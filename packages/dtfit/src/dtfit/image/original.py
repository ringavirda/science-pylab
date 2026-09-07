"""The original: a sampled signal on an interval."""

from __future__ import annotations

from typing import Any

import numpy as np

from dtfit._pandas import (
    _MULTIVARIATE_MSG,
    is_dataframe,
    is_series,
    to_1d_array,
)
from .grid import Grid


def _coerce(v: Any, name: str) -> np.ndarray:
    if is_series(v) or is_dataframe(v):
        v = to_1d_array(v, name)
    return np.array(v, dtype=float, copy=True)


class Original:
    """A sampled signal ``(x, y)`` with per-sample weights on a domain.

    Positions are sorted non-decreasing; ties are allowed, but the samples
    must span an interval. ``y`` and the weights follow the same sort. ``w``
    is the inverse-variance weight of each sample, ones unless ``w`` or
    ``sigma`` (a per-sample standard deviation, ``w = 1/sigma**2``) is given.
    ``domain`` defaults to ``(x[0], x[-1])`` and must contain every position.
    ``nan_policy`` is ``"raise"`` or ``"omit"``; omission drops a pair when
    ``x``, ``y`` or its weight is non-finite.

    Attributes:
        x, y, w: float arrays of equal length ``n``.
        domain: ``(x0, x1)``.
        grid: the :class:`Grid` descriptor of ``x``.
        weighted: whether any weight differs from one.
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        w: Any = None,
        *,
        sigma: Any = None,
        domain: tuple[float, float] | None = None,
        nan_policy: str = "raise",
    ) -> None:
        if nan_policy not in ("raise", "omit"):
            raise ValueError(
                f"nan_policy must be 'raise' or 'omit', got {nan_policy!r}"
            )
        x = _coerce(x, "data_x")
        y = _coerce(y, "data_y")
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(
                _MULTIVARIATE_MSG.format(
                    name="data_x and data_y",
                    what=f"shapes {x.shape} and {y.shape}",
                )
            )
        if x.size != y.size:
            raise ValueError(
                "data_x and data_y must have the same length; got "
                f"{x.size} and {y.size}."
            )
        if w is not None and sigma is not None:
            raise ValueError("give either w or sigma, not both")
        if sigma is not None:
            s = np.asarray(sigma, dtype=float)
            if s.ndim == 0:
                s = np.broadcast_to(s, x.shape)
            s = s.reshape(-1)
            if s.size != x.size:
                raise ValueError(
                    "sigma must have the same length as data_y; got "
                    f"{s.size} and {x.size}."
                )
            if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
                raise ValueError(
                    "sigma must be finite and strictly positive."
                )
            w_arr = 1.0 / (s * s)
        elif w is not None:
            w_arr = np.array(w, dtype=float, copy=True).reshape(-1)
            if w_arr.size != x.size:
                raise ValueError(
                    "w must have the same length as data_y; got "
                    f"{w_arr.size} and {x.size}."
                )
        else:
            w_arr = np.ones(x.size)
        good = np.isfinite(x) & np.isfinite(y) & np.isfinite(w_arr)
        if not good.all():
            if nan_policy == "raise":
                raise ValueError(
                    "data contains non-finite values (NaN/inf); pass "
                    "nan_policy='omit' to drop them."
                )
            x, y, w_arr = x[good], y[good], w_arr[good]
        if np.any(w_arr <= 0.0):
            raise ValueError("weights must be strictly positive.")
        if x.size < 2:
            raise ValueError(f"need at least 2 samples; got {x.size}.")
        if np.any(np.diff(x) < 0):
            order = np.argsort(x, kind="stable")
            x, y, w_arr = x[order], y[order], w_arr[order]
        if domain is None:
            if x[-1] <= x[0]:
                raise ValueError(
                    "the samples span no interval (all positions equal)."
                )
            dom = (float(x[0]), float(x[-1]))
        else:
            dom = (float(domain[0]), float(domain[1]))
            if dom[0] > x[0] or dom[1] < x[-1] or dom[0] >= dom[1]:
                raise ValueError(
                    f"domain {dom} must contain the samples "
                    f"[{x[0]}, {x[-1]}]."
                )
        self.x = x
        self.y = y
        self.w = w_arr
        self.domain = dom
        self.grid = Grid.of(x)
        self.weighted = bool(np.any(w_arr != 1.0))

    @property
    def n(self) -> int:
        return int(self.x.size)

    def window(self, i0: int, i1: int) -> "Original":
        """The samples ``i0:i1`` as an Original on their own span.

        Re-validates through the constructor, so ``i1 - i0`` must leave at
        least two samples.
        """
        return Original(
            self.x[i0:i1].copy(),
            self.y[i0:i1].copy(),
            self.w[i0:i1].copy(),
        )

    def image(
        self,
        basis: Any = "legendre",
        order: int | None = None,
        *,
        robust: bool = False,
        huber_c: float = 1.345,
    ):
        """The projection of this Original onto a basis; see
        :meth:`Image.of`."""
        from .image import Image

        return Image.of(self, basis, order, robust=robust, huber_c=huber_c)

    def fit(self, model: Any, var: str | None = None, **kwargs: Any):
        """Fit a model to this Original; see :func:`dtfit.image.fit`."""
        from .fit import fit

        return fit(model, self, var, **kwargs)

    def residuals(
        self, model: Any, params: Any, var: str | None = None
    ) -> np.ndarray:
        """``y - f(x; params)`` for a model as in :func:`dtfit.image.fit`."""
        from dtfit.methods._modelinput import resolve_model

        spec = resolve_model(model, var)
        return self.y - spec.eval(self.x, np.asarray(params, dtype=float))
