"""dtfit.models: picking and constructing the right model.

Which structure you fit matters more than which solver runs. This package
makes that choice ergonomic:

- a catalog of named families (:func:`logistic`, :func:`gaussian`,
  :func:`damped_oscillation`, ...), each a :class:`Model` that reads its own
  ``p0``/``bounds`` off the data. You pick structure, not strings::

      from dtfit import models
      fit = models.logistic().fit(x, y)          # self-seeded

- composition with ``+``, say a trend plus a cycle. The second component is
  seeded on the detrended residual of the first::

      fit = (models.linear() + models.sine()).fit(x, y)

- inference: :func:`suggest_models` shortlists candidates by data shape, fits
  them, and ranks the families by AIC::

      for s in suggest_models(x, y)[:3]:
          print(s.name, s.r2, s.aic)

- extensibility: :func:`register` adds your own family to the catalog, where
  :func:`all_models` returns it and :func:`suggest_models` considers it.
  :func:`unregister` removes it again::

      models.register("myline", lambda: Model("a0 + a1*x", name="myline"))

Families are grouped by ``category`` (trend / growth / decay / sigmoid /
saturating / peak / oscillatory); see :data:`CATALOG`.

The package also carries the input seam every fitter shares:
:func:`resolve_model` turns a string, a ``sympy.Expr`` or a callable into one
:class:`ModelSpec`, and :func:`normalize_p0` / :func:`normalize_bounds` put a
guess and a box into canonical parameter order.
"""

from dtfit._input import (
    ModelSpec,
    normalize_bounds,
    normalize_p0,
    resolve_model,
    result_kwargs,
)
from ._model import Model, as_fit_data
from ._stochastic import Stochastic
from ._suggest import suggest_models, Suggestion
from ._catalog import (
    CATALOG,
    all_models,
    register,
    unregister,
    # trend
    linear,
    quadratic,
    cubic,
    power_law,
    logarithmic,
    sqrt_law,
    # growth
    exponential,
    exp_growth_offset,
    # decay / relaxation
    exp_decay,
    exp_decay_offset,
    first_order,
    biexponential,
    stretched_exponential,
    # sigmoid
    logistic,
    gompertz,
    weibull_cdf,
    tanh_step,
    # saturating / rational
    michaelis_menten,
    hill,
    # peak
    gaussian,
    lorentzian,
    double_gaussian,
    # oscillatory
    sine,
    damped_oscillation,
    fourier_series,
)

__all__ = [
    "Model",
    "Stochastic",
    "ModelSpec",
    "resolve_model",
    "result_kwargs",
    "normalize_p0",
    "normalize_bounds",
    "suggest_models",
    "Suggestion",
    "CATALOG",
    "all_models",
    "register",
    "unregister",
    "linear",
    "quadratic",
    "cubic",
    "power_law",
    "logarithmic",
    "sqrt_law",
    "exponential",
    "exp_growth_offset",
    "exp_decay",
    "exp_decay_offset",
    "first_order",
    "biexponential",
    "stretched_exponential",
    "logistic",
    "gompertz",
    "weibull_cdf",
    "tanh_step",
    "michaelis_menten",
    "hill",
    "gaussian",
    "lorentzian",
    "double_gaussian",
    "sine",
    "damped_oscillation",
    "fourier_series",
]
