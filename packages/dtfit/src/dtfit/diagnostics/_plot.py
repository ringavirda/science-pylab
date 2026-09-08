"""Visualization helpers following scikit-learn's ``Display`` convention.

Each class exposes ``from_estimator`` and ``from_predictions`` constructors
plus a ``plot`` method. Plotting draws onto a supplied or freshly created
Matplotlib ``Axes``, stores the artists as ``ax_``, ``figure_`` and the rest,
and returns the display instance. It never calls ``plt.show()``: rendering is
the caller's to control.

Matplotlib is an optional dependency, the ``viz`` extra. Importing this
module does not require it; constructing a plot does.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._report import _basic_stats

try:  # matplotlib is the optional 'viz' extra
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only without the extra
    plt = None  # type: ignore[assignment]

__all__ = ["FitDisplay", "ResidualsDisplay"]


def _require_plt():
    if plt is None:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: "
            "pip install 'dtfit[viz]'"
        )
    return plt


def _as_1d_x(X: np.ndarray) -> np.ndarray:
    """Extract the single input feature as a 1-D array."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        if X.shape[1] != 1:
            raise ValueError(
                "Display helpers support a single input feature; got "
                f"{X.shape[1]} columns."
            )
        X = X[:, 0]
    return X.ravel()


class FitDisplay:
    """Observed data and the fitted curve over a single input feature.

    Attributes set after :meth:`plot`:
        ax_: The Matplotlib axes with the plot.
        figure_: The figure containing ``ax_``.
        line_: The fitted-curve ``Line2D``.
        scatter_: The data scatter ``PathCollection``.
    """

    def __init__(
        self,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        estimator_name: Optional[str] = None,
    ) -> None:
        self.x = np.asarray(x, dtype=float).ravel()
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()
        self.estimator_name = estimator_name

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ax=None,
        **kwargs,
    ) -> "FitDisplay":
        """Build and plot a display from a fitted estimator and data."""
        x = _as_1d_x(X)
        y_pred = np.asarray(estimator.predict(X), dtype=float).ravel()
        disp = cls(
            x, y, y_pred, estimator_name=estimator.__class__.__name__
        )
        return disp.plot(ax=ax, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        x: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        ax=None,
        estimator_name: Optional[str] = None,
        **kwargs,
    ) -> "FitDisplay":
        """Build and plot a display from raw predictions."""
        disp = cls(x, y_true, y_pred, estimator_name=estimator_name)
        return disp.plot(ax=ax, **kwargs)

    def plot(
        self,
        ax=None,
        *,
        data_kwargs: Optional[dict] = None,
        line_kwargs: Optional[dict] = None,
    ) -> "FitDisplay":
        """Render the data scatter and fitted curve onto ``ax``."""
        plt = _require_plt()
        if ax is None:
            _, ax = plt.subplots()

        order = np.argsort(self.x)
        fit_label = (
            f"{self.estimator_name} fit" if self.estimator_name else "fit"
        )
        self.scatter_ = ax.scatter(
            self.x,
            self.y_true,
            **{
                "s": 12, "color": "0.6", "label": "data",
                **(data_kwargs or {}),
            },
        )
        (self.line_,) = ax.plot(
            self.x[order],
            self.y_pred[order],
            **{
                "color": "tab:red",
                "lw": 2,
                "label": fit_label,
                **(line_kwargs or {}),
            },
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class ResidualsDisplay:
    """Residuals (``y_true - y_pred``) plotted against the predicted values.

    The axes title carries R^2 and RMSE.

    Attributes set after :meth:`plot`:
        ax_: The Matplotlib axes with the plot.
        figure_: The figure containing ``ax_``.
        scatter_: The residual scatter ``PathCollection``.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        estimator_name: Optional[str] = None,
    ) -> None:
        self.y_true = np.asarray(y_true, dtype=float).ravel()
        self.y_pred = np.asarray(y_pred, dtype=float).ravel()
        self.residuals = self.y_true - self.y_pred
        self.estimator_name = estimator_name

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        *,
        ax=None,
        **kwargs,
    ) -> "ResidualsDisplay":
        """Build and plot a display from a fitted estimator and data."""
        y_pred = np.asarray(estimator.predict(X), dtype=float).ravel()
        disp = cls(y, y_pred, estimator_name=estimator.__class__.__name__)
        return disp.plot(ax=ax, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        ax=None,
        estimator_name: Optional[str] = None,
        **kwargs,
    ) -> "ResidualsDisplay":
        """Build and plot a display from raw predictions."""
        disp = cls(y_true, y_pred, estimator_name=estimator_name)
        return disp.plot(ax=ax, **kwargs)

    def plot(
        self, ax=None, *, scatter_kwargs: Optional[dict] = None
    ) -> "ResidualsDisplay":
        """Render the residuals-vs-predicted scatter onto ``ax``."""
        plt = _require_plt()
        if ax is None:
            _, ax = plt.subplots()

        self.scatter_ = ax.scatter(
            self.y_pred,
            self.residuals,
            **{"s": 12, "color": "0.5", **(scatter_kwargs or {})},
        )
        ax.axhline(0.0, color="tab:red", lw=1)
        stats = _basic_stats(self.y_true, self.y_pred)
        ax.set_title(
            f"Residuals (R²={stats['r2']:.3f}, "
            f"RMSE={stats['rmse']:.3g})"
        )
        ax.set_xlabel("predicted")
        ax.set_ylabel("residual")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self
