"""dtfit: differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Written as
part of the author's PhD dissertation.

The public interface is layered, from choosing the engine yourself to letting
it choose.

Batch fitters:
    fit on an Original or an Image; fit_lsi and fit_eac are its Legendre and
    block presets; fit_dsb is the symbolic reference. ensemble_fit is the
    overlapping-window ensemble for densely contaminated data, and
    find_degree selects the polynomial degree for DSB.

Estimator:
    NonlineRegressor, an sklearn-compatible fit/predict/score over LSI, EAC
    and DSB. It composes with Pipeline and GridSearchCV.

High level:
    auto_estimate and auto_forecast route by signal shape. models / Model /
    suggest_models are a catalog of self-seeding model families and an AIC
    recommender for picking structure rather than sympy strings.

Streaming:
    EACFilter and LSIFilter track parameters online through ``partial_fit``;
    start from their ``.tracking()`` / ``.robust()`` presets. FilterBank and
    FusedChiSquareDetector drive many streams at once.

Scale:
    PartitionedLSI, PartitionedEAC and PartitionedBatchLSI are the one-pass
    and distributed map-reduce estimators. fit_lsi_batched is the GEMM-batched
    multi-channel path, fit_many the process/thread fan-out. The
    project_spectra primitive lives in ``dtfit.scale``.

Stochastic series:
    fit_stochastic, StochasticModel and Stochastic fit the deterministic
    functionals of a random process (autocovariance, spectrum, trend/cycle) to
    characterize, forecast and generate it. StochasticFilter tracks that
    structure online. See ``dtfit.stochastic``.

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
from dtfit.methods import (
    fit_lsi,
    fft_frequency_seed,
    fit_eac,
    fit_dsb,
    ensemble_fit,
    EnsembleResult,
    find_degree,
)
from dtfit.image import Original, Image, fit, order_for
from dtfit.estimators import NonlineRegressor
from dtfit.streaming import (
    EACFilter,
    LSIFilter,
    FilterBank,
    FusedChiSquareDetector,
)
from dtfit.scale._parallel import fit_many, FittingProblem
from dtfit.scale._partitioned import PartitionedLSI, PartitionedEAC, PartitionedBatchLSI
from dtfit.scale._batched import fit_lsi_batched
from dtfit.auto import auto_estimate, auto_forecast, ForecastResult
from dtfit import models
from dtfit.models import Model, Stochastic, suggest_models, register, unregister
from dtfit import stochastic
from dtfit.stochastic import fit_stochastic, StochasticModel, StochasticFilter

__all__ = [
    "Original",
    "Image",
    "fit",
    "order_for",
    "NonlineRegressor",
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
    "EACFilter",
    "LSIFilter",
    "FilterBank",
    "FusedChiSquareDetector",
    "PartitionedLSI",
    "PartitionedEAC",
    "PartitionedBatchLSI",
    "fit_lsi_batched",
    "fit_lsi",
    "fft_frequency_seed",
    "fit_eac",
    "fit_dsb",
    "ensemble_fit",
    "EnsembleResult",
    "find_degree",
    "fit_many",
    "FittingProblem",
    "FittingResult",
    "enable_logging",
    "logger",
    "diagnostics",
    "__version__",
]
