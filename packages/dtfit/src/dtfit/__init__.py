"""dtfit: differential-transformation fitting.

Methods for fitting models that are nonlinear in their parameters
(exponential, transcendental, mixed) for nonlinear smoothing and forecasting,
built in the scheme of differential / non-Taylor transformations. Written as
part of the author's PhD dissertation.

Everything is built on one data structure. An Original is a sampled signal;
its Image in a basis at an order is the pair of sums ``S = Phi^T (w y)`` and
``G = Phi^T diag(w) Phi``, a sufficient statistic for estimation in the span
of that basis, additive over sample sets and nested in the order.

Fitting:
    fit(model, data) on either type, with basis="auto" routing by outcome;
    fit_lsi and fit_eac are its Legendre and block presets; order_for gives
    the order a model's sensitivities need. fit_many fans independent fits
    across processes or threads.

Streaming and scale:
    ImageFilter tracks parameters online on the window image; LSIFilter and
    EACFilter fix its basis. ImageStream accumulates a signal in fixed
    memory as it arrives, emits block images and assembles them back.

Models:
    dtfit.models is a catalog of self-seeding families (Model), composable
    with "+", plus register / unregister and suggest_models, which ranks
    families by AIC.

Stochastic series:
    dtfit.stochastic characterizes a random series from its second-order
    image: the autocovariance, the spectrum, the aggregated variance and
    the trend plus seasonal cycle, each read off one additive statistic.
    The fitted model forecasts, bands and generates, and StochasticFilter
    tracks the same structure per input. auto_forecast is the structured
    fit-then-extrapolate router.

Every fit returns a FittingResult: named parameters, uncertainty, an
optimizer ``converged`` flag, and extrapolation-aware ``predict``.
dtfit.diagnostics (fit_report, residual tests, the ``*Display`` helpers) is
imported explicitly, after the scikit-learn convention, as are
dtfit.sklearn (the NonlineRegressor estimator), dtfit.reference (DSB, the
exact-balance ancestor) and dtfit.log (opt-in library logging).

Core dependencies: numpy, scipy, sympy. Optional extras: scikit-learn for
dtfit.sklearn, matplotlib for the plots, via ``pip install 'dtfit[viz]'``.
"""

from dtfit.__about__ import __version__
from dtfit import diagnostics
from dtfit.types import FittingResult
from dtfit import models
from dtfit.image import (
    Original,
    Image,
    ImageStream,
    fit,
    fit_lsi,
    fit_eac,
    order_for,
)
from dtfit.image.parallel import fit_many
from dtfit.streaming import ImageFilter, LSIFilter, EACFilter
from dtfit.models import suggest_models
from dtfit.forecast import auto_forecast, ForecastResult
from dtfit import stochastic

__all__ = [
    "Original",
    "Image",
    "ImageStream",
    "ImageFilter",
    "fit",
    "fit_lsi",
    "fit_eac",
    "LSIFilter",
    "EACFilter",
    "order_for",
    "fit_many",
    "models",
    "suggest_models",
    "auto_forecast",
    "FittingResult",
    "ForecastResult",
    "stochastic",
    "diagnostics",
    "__version__",
]
