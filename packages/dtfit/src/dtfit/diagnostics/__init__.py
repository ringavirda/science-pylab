"""Fit diagnostics and visualization for dtfit.

These tools evaluate a fitted dtfit model. They take a
:class:`dtfit.FittingResult` and report parameter uncertainty, information
criteria for model comparison, and residual-structure tests. For plain scalar
metrics on ``(y_true, y_pred)`` arrays, use ``sklearn.metrics`` or
``scipy.stats`` directly.

    from dtfit.diagnostics import fit_report, residual_diagnostics, FitDisplay

The ``*Display`` classes need matplotlib (``pip install 'dtfit[viz]'``).
"""

from ._report import fit_report, residual_diagnostics, residual_stats
from ._plot import FitDisplay, ResidualsDisplay

__all__ = [
    "fit_report",
    "residual_diagnostics",
    "residual_stats",
    "FitDisplay",
    "ResidualsDisplay",
]
