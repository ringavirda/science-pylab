"""Stochastic-series characterization, forecasting, generation and tracking.

dtfit fits a deterministic ``y = f(t; theta)``, and a genuinely random series
has no such ``f``. What it has is second-order structure: an autocovariance, a
spectrum, the variance of block means across scales, a deterministic mean of
trend plus seasonal cycle, and the volatility of its increments. The tier's
image, :class:`SecondOrderImage`, is the additive sufficient statistic of
exactly that structure, and every estimator, gate, forecaster and generator
here reads from it. The functionals' forms are damped exponentials, damped
cosines and power laws, the shapes the batch fitters take: fit the functional,
read the process parameter out of it.

Layered API:

* the image: :class:`SecondOrderImage` (construction from an
  :class:`~dtfit.image.Original` or a chunked stream, exact ``merge``,
  ``state`` and ``restore``, the read-outs) and :class:`SecondOrderStream`
  (a running image, or a stream of block images).
* estimators, each on a series, an Original or an image:
  :func:`hurst_aggvar` / :func:`hurst_spectral` (long memory),
  :func:`ar1_reversion` (mean reversion), :func:`garch_persistence`
  (volatility), :func:`cycle_period` (stochastic cycle),
  :func:`decompose_trend_cycle`, :func:`dickey_fuller` (unit root).
* batch: :func:`fit_stochastic` composes the routes behind significance gates
  into one :class:`StochasticModel` that labels the regime, forecasts by
  backtest model selection and generates fresh realizations
  (:meth:`StochasticModel.simulate`).
* streaming: :class:`StochasticFilter`, the per-input counterpart tracking
  exponentially weighted second-order statistics, with a fused change-point
  detector.

:class:`dtfit.Stochastic` wraps the batch entry point in the ``.fit(x, y)``
convention of :class:`dtfit.Model`. There is no ``.predict``: a stochastic
process is forecast, not point-evaluated.
"""

from .image import SecondOrderImage
from .stream import SecondOrderStream
from .estimators import (
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
    dickey_fuller,
    adf_pvalue,
    as_image,
)
from .gates import fit_stochastic, StochasticModel, is_nonstationary
from .forecast import FORECASTERS
from .filter import StochasticFilter

__all__ = [
    "SecondOrderImage",
    "SecondOrderStream",
    "as_image",
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
    "dickey_fuller",
    "adf_pvalue",
    "is_nonstationary",
    "fit_stochastic",
    "StochasticModel",
    "FORECASTERS",
    "StochasticFilter",
]
