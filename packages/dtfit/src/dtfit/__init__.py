"""dtfit: differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Written as
part of the author's PhD dissertation.

The public interface is layered, from choosing the engine yourself to letting
it choose.

Batch fitters:
    fit on an Original or an Image; fit_lsi and fit_eac are its Legendre and
    block presets.

High level:
    auto_estimate and auto_forecast route by signal shape. models / Model /
    suggest_models are a catalog of self-seeding model families and an AIC
    recommender for picking structure rather than sympy strings.

Streaming:
    ImageFilter tracks parameters online on the window image; LSIFilter and
    EACFilter fix its basis; start from ``.tracking()`` / ``.robust()``;
    ``result()`` returns a FittingResult with the calibrated covariance.

Scale:
    ImageStream accumulates a signal in fixed memory as it arrives. Its
    blocks are retained and assemble merges them into one Image; a channel
    axis batches many signals through the same accumulator. fit_many fans
    independent fits across processes or threads.

Stochastic series:
    fit_stochastic, StochasticModel and Stochastic characterize a random
    series from its second-order image: the autocovariance, the spectrum,
    the aggregated variance and the trend plus seasonal cycle, each read off
    one additive statistic. The fitted model forecasts, bands and generates.
    StochasticFilter tracks the same structure per input.

Every fit returns a FittingResult: named parameters, uncertainty, an optimizer
``converged`` flag, and extrapolation-aware ``predict``. enable_logging and
logger are the opt-in library logging. dtfit.diagnostics (fit_report, residual
tests, the ``*Display`` helpers) is imported explicitly, after the
scikit-learn convention.

Core dependencies: numpy, scipy, sympy, scikit-learn.
Optional extras: matplotlib, via ``pip install 'dtfit[viz]'``.
"""

from dtfit.__about__ import __version__
from dtfit import diagnostics
from dtfit.types import FittingResult
from dtfit.log import enable_logging, logger
from dtfit.image import fit_lsi, fit_eac, fft_frequency_seed
from dtfit.image import Original, Image, ImageStream, fit, order_for
from dtfit.streaming import (
    ImageFilter,
    LSIFilter,
    EACFilter,
)
from dtfit.image.parallel import fit_many, FittingProblem
from dtfit.auto import auto_estimate, auto_forecast, ForecastResult
from dtfit import models
from dtfit.models import Model, Stochastic, suggest_models, register, unregister
from dtfit import stochastic
from dtfit.stochastic import fit_stochastic, StochasticModel, StochasticFilter

__all__ = [
    "Original",
    "Image",
    "ImageStream",
    "fit",
    "order_for",
    "auto_estimate",
    "auto_forecast",
    "ForecastResult",
    "models",
    "Model",
    "suggest_models",
    "register",
    "unregister",
    "stochastic",
    "Stochastic",
    "fit_stochastic",
    "StochasticModel",
    "StochasticFilter",
    "ImageFilter",
    "EACFilter",
    "LSIFilter",
    "fit_lsi",
    "fft_frequency_seed",
    "fit_eac",
    "fit_many",
    "FittingProblem",
    "FittingResult",
    "enable_logging",
    "logger",
    "diagnostics",
    "__version__",
]
