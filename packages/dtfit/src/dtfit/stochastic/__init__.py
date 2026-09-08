"""Stochastic-series characterization, forecasting, generation and tracking.

dtfit fits a deterministic ``y = f(t; theta)``, and a genuinely random series
has no such ``f``. What it does have is deterministic functionals: the
autocovariance, the spectrum, the aggregated variance, the trend and cycle.
Their forms are damped exponentials, damped cosines and power laws, exactly
the shapes the batch fitters take. Fit the functional, read the process
parameters out of it.

Layered API:

* estimators: :func:`hurst_aggvar` / :func:`hurst_spectral` (long memory),
  :func:`ar1_reversion` (mean reversion), :func:`garch_persistence`
  (volatility), :func:`cycle_period` (stochastic cycle),
  :func:`decompose_trend_cycle`. Each recovers one stochastic-model parameter
  by feeding a functional to ``fit_lsi`` / ``fit_eac``.
* batch: :func:`fit_stochastic` composes the routes behind significance gates
  into one :class:`StochasticModel` that labels the regime, forecasts by
  backtest model selection and generates fresh realizations
  (:meth:`StochasticModel.simulate`).
* streaming: :class:`StochasticFilter`, the per-input counterpart of the
  second-order stage (EWMA autocovariances read by the EAC equal-areas
  criterion) with a fused change-point detector.

:class:`dtfit.Stochastic` wraps the batch entry point in the ``.fit(x, y)``
convention of :class:`dtfit.Model`; the fitted :class:`StochasticModel` then
offers ``.forecast()``, ``.simulate()`` and ``.summary()``. There is no
``.predict``: a stochastic process is forecast, not point-evaluated.
"""

from .image import SecondOrderImage

from dtfit.stochastic._estimators import (
    sample_acf,
    hurst_aggvar,
    hurst_spectral,
    ar1_reversion,
    ar_order,
    fit_ar,
    fractional_difference,
    garch_persistence,
    cycle_period,
    decompose_trend_cycle,
)
from dtfit.stochastic._model import fit_stochastic, StochasticModel
from dtfit.stochastic._forecast import FORECASTERS
from dtfit.stochastic._filter import StochasticFilter

__all__ = [
    "SecondOrderImage",
    "sample_acf",
    "hurst_aggvar",
    "hurst_spectral",
    "ar1_reversion",
    "ar_order",
    "fit_ar",
    "fractional_difference",
    "garch_persistence",
    "cycle_period",
    "decompose_trend_cycle",
    "fit_stochastic",
    "StochasticModel",
    "FORECASTERS",
    "StochasticFilter",
]
