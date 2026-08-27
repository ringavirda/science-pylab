"""Guarded pandas interop.

pandas is optional. This module owns the one guarded import, and everything
else reaches pandas through these helpers rather than importing it directly.
That is what lets the fitters and :class:`dtfit.FittingResult` take ``Series``
and single-column ``DataFrame`` inputs and hand back index-aligned ``Series``:
pandas in -> pandas out.

With pandas missing, :data:`HAS_PANDAS` is ``False``, the ``is_*`` predicates
return ``False``, :func:`to_1d_array` still coerces plain array-likes, and the
index helpers return ``None`` or the plain ndarray.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pandas is an optional dependency (not in the core install)
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover - exercised only in a pandas-free env
    pd = None  # type: ignore[assignment]
    HAS_PANDAS = False


# One message for the 1-D boundary, shared by every entry point.
_MULTIVARIATE_MSG = (
    "{name} must be 1-D; got {what}. dtfit's integral criteria (LSI/EAC) are "
    "one-dimensional, so multivariate X (several predictors) is not supported. "
    "If instead you have a 1-D signal that is a sum of components along one axis "
    "(e.g. trend + cycle), compose 1-D models with `+` "
    "(models.linear() + models.sine()); see the 'Multivariate data' note in the "
    "docs."
)


def is_series(obj: Any) -> bool:
    """True if ``obj`` is a pandas ``Series``; ``False`` without pandas."""
    return HAS_PANDAS and isinstance(obj, pd.Series)


def is_dataframe(obj: Any) -> bool:
    """True if ``obj`` is a pandas ``DataFrame``; ``False`` without pandas."""
    return HAS_PANDAS and isinstance(obj, pd.DataFrame)


def to_1d_array(obj: Any, name: str = "x") -> np.ndarray:
    """Coerce ``obj`` to a 1-D float ``ndarray``.

    A ``Series`` becomes its float values, and so does a single-column
    ``DataFrame``. A genuinely multivariate input, meaning a multi-column
    ``DataFrame`` or a 2-D array with more than one column, raises
    :class:`ValueError`, since dtfit is one-dimensional. A 1-D ndarray or list
    comes back bit-identical; an ``(n, 1)`` column vector is squeezed.
    """
    if is_series(obj):
        return np.asarray(obj.to_numpy(dtype=float)).reshape(-1)
    if is_dataframe(obj):
        ncol = obj.shape[1]
        if ncol != 1:
            raise ValueError(_MULTIVARIATE_MSG.format(name=name, what=f"a DataFrame with {ncol} columns"))
        return np.asarray(obj.iloc[:, 0].to_numpy(dtype=float)).reshape(-1)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim >= 2 and int(np.prod(arr.shape[1:])) != 1:
        # Do not flatten: that turns an nD mistake into a wrong 1-D fit. A
        # single trailing column is a column vector, squeezed below.
        raise ValueError(_MULTIVARIATE_MSG.format(name=name, what=f"an array of shape {arr.shape}"))
    return arr.reshape(-1)


def capture_index(obj: Any) -> Any:
    """The pandas index of a ``Series`` / ``DataFrame``, else ``None``."""
    if is_series(obj) or is_dataframe(obj):
        return obj.index
    return None


def extend_index(index: Any, horizon: int) -> Any:
    """The length-``horizon`` future index continuing ``index``.

    * ``DatetimeIndex``: continued at its frequency, ``index.freq`` when set
      and :func:`pandas.infer_freq` otherwise; ``None`` if neither gives one.
    * ``RangeIndex`` or any integer-typed index: continued by its constant
      step, either the range step or the gap between the last two labels.
    * anything else: ``None``.

    Returns ``None`` when pandas is absent, ``index`` is ``None`` or empty, or
    ``horizon <= 0``.
    """
    if not HAS_PANDAS or index is None or horizon <= 0 or len(index) == 0:
        return None
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or pd.infer_freq(index)
        if freq is None:
            return None
        return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
    if isinstance(index, pd.RangeIndex):
        step = index.step
        start = int(index[-1]) + step
        return pd.RangeIndex(start, start + step * horizon, step)
    if pd.api.types.is_integer_dtype(getattr(index, "dtype", None)):
        if len(index) < 2:
            return None
        step = int(index[-1]) - int(index[-2])
        last = int(index[-1])
        return pd.Index([last + step * (i + 1) for i in range(horizon)])
    return None


def as_series(values: np.ndarray, index: Any) -> Any:
    """A pandas ``Series`` of ``values`` aligned to ``index``, when possible.

    With pandas absent or ``index`` ``None`` the plain ``ndarray`` comes back
    unchanged; a non-pandas caller is never handed a ``Series``.
    """
    arr = np.asarray(values)
    if HAS_PANDAS and index is not None:
        return pd.Series(arr, index=index)
    return arr
