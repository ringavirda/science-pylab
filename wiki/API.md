# dtfit API reference

Complete reference for the **public** `dtfit` API -- every name exported from the
top-level package and the `dtfit.diagnostics` submodule. Internals (anything
under a `_`-prefixed module) are not part of the public contract.

For the *ideas* behind these functions read [../guides/](Guides); for the
*math* read [../methods/](Methods). This reference is for looking up exact
signatures, arguments, return types, and behavior.

> **Looking for the experimental adaptations** (`fit_lsi_basis`, `fit_joint`,
> `boosted_fit`)? Those live in the separate `dtfit-experimental` package -- see
> [../experimental/adaptations-api.md](Experimental-Adaptations-API). The
> `InformationFilter` fusion primitive lives there too (`from
> dtfit_experimental import InformationFilter`), not in the stable streaming
> surface. This page covers the **stable** `dtfit` API.

## Conventions

- Fitters take an `Original` or an `Image` directly, or plain 1-D NumPy arrays
  `x`, `y` (from which the batch fitters build an `Original`), and a model as
  a **sympy-style expression string**, a `sympy.Expr`, or a callable `f(x,
  *params)`, plus the name of its main variable, e.g. `fit_lsi(x, y,
  "a*exp(b*x)", "x")`.
- **Parameters are the free symbols** of the expression (everything except the
  variable), and results are ordered by **sorted parameter name** -- a stable
  layout used everywhere. So `"a*exp(b*x)"` has parameters `[a, b]` in that order.
- Batch fitters return a [`FittingResult`](API-Types); streaming filters expose
  `partial_fit` / `predict` / `params_`.
- Keyword-only arguments appear after `*` in the signatures below.

## The public surface, by area

| Area | Names | Page |
|---|---|---|
| **Batch fitting** | `fit`, `Original`, `Image`, `order_for`, `fit_lsi`, `fit_eac` (`coverage`, `fft_frequency_seed` and the `Image` analytics `noise_sigma` / `effective_order` / `decay` / `test_equal` / `test_structure` in `dtfit.image`) | [fitting.md](API-Fitting) |
| **Result type** | `FittingResult` | [types.md](API-Types) |
| **sklearn estimator** | `NonlineRegressor` (in `dtfit.sklearn`) | [estimator.md](API-Estimator) |
| **Forecasting** | `auto_forecast`, `ForecastResult` | [auto.md](API-Auto) |
| **Model framework** | `models`, `suggest_models` (`Model`, `register`, `unregister` and the catalog families in `dtfit.models`) | [models.md](API-Models) |
| **Reference method** | `fit_dsb`, `find_degree` (in `dtfit.reference`) | [dsb.md](Methods-DSB) |
| **Stochastic series** | `stochastic` (`fit_stochastic`, `StochasticModel`, `StochasticFilter`, `SecondOrderImage`, `SecondOrderStream` and the estimators in `dtfit.stochastic`; `Stochastic` in `dtfit.models`) | [stochastic.md](API-Stochastic) |
| **Streaming / online** | `ImageFilter`, `LSIFilter`, `EACFilter` (`DriftDetector` in `dtfit.streaming`) | [streaming.md](API-Streaming) |
| **Streams and scale** | `ImageStream`, `fit_many` (`FittingProblem`, `assemble`, `legendre_transfer`, `block_transfer` in `dtfit.image`) | [scaling.md](API-Scaling) |
| **Diagnostics** | `diagnostics` (`fit_report`, `residual_diagnostics`, `residual_stats`, `FitDisplay`, `ResidualsDisplay`) | [diagnostics.md](API-Diagnostics) |
| **Logging** | `enable_logging`, `logger` (in `dtfit.log`) | [below](#logging) |

The top level holds fifteen names and three subpackages: `Original`, `Image`,
`ImageStream`, `ImageFilter`, `fit`, `fit_lsi`, `fit_eac`, `LSIFilter`,
`EACFilter`, `order_for`, `fit_many`, `suggest_models`, `auto_forecast`,
`FittingResult`, `ForecastResult`, plus `models`, `stochastic` and
`diagnostics`. Everything else is reached through its own module, as the
import map below shows.

## Import map

```python
# batch fitting
from dtfit import (fit, Original, Image, order_for, fit_lsi, fit_eac,
                   FittingResult)
from dtfit.image import coverage, fft_frequency_seed

# forecasting
from dtfit import auto_forecast, ForecastResult

# model framework
from dtfit import models, suggest_models
from dtfit.models import Model, register, unregister, resolve_model

# the reference method (the exact-balance ancestor, not part of fit)
from dtfit.reference import fit_dsb, find_degree

# stochastic series (characterize / forecast / generate / track random data)
from dtfit import stochastic
from dtfit.models import Stochastic
from dtfit.stochastic import (
    fit_stochastic, StochasticModel, StochasticFilter, SecondOrderImage,
    SecondOrderStream, hurst_spectral, ar1_reversion, garch_persistence,
    cycle_period, decompose_trend_cycle, dickey_fuller, ar_order, fit_ar,
    fractional_difference, FORECASTERS)

# sklearn estimator (the only part of dtfit that imports scikit-learn)
from dtfit.sklearn import NonlineRegressor

# streaming
from dtfit import ImageFilter, LSIFilter, EACFilter
from dtfit.streaming import DriftDetector

# streams and scale
from dtfit import ImageStream, fit_many
from dtfit.image import (FittingProblem, assemble, legendre_transfer,
                         block_transfer)

# Image analytics (also reachable as Image methods)
from dtfit.image import (noise_sigma, effective_order, decay,
                         test_equal, test_structure)

# diagnostics (submodule, not top-level - sklearn convention)
from dtfit.diagnostics import (fit_report, residual_diagnostics,
                               residual_stats, FitDisplay, ResidualsDisplay)

# logging
from dtfit.log import enable_logging, logger
```

<a name="logging"></a>
## Logging

`dtfit` is silent by default. Opt in to see what the solvers are doing:

```python
from dtfit.log import enable_logging
enable_logging()              # INFO-level chatter from the methods
enable_logging(level="DEBUG") # more detail
```

- **`dtfit.log.enable_logging(level="INFO")`** -- attach a handler to the
  library logger and
  set its level. Call once at startup.
- **`logger`** -- the underlying `logging.Logger` (`"dtfit"`), if you want to wire
  it into your own logging configuration instead.

Internally the methods emit progress through a small `echo` helper (window counts,
selected polynomial degree, fitted coefficients); none of it prints unless you
enable logging.

## A 30-second example

```python
import numpy as np
from dtfit import fit_lsi
from dtfit.diagnostics import fit_report

x = np.linspace(0, 4, 200)
y = 0.5 + 2.0 * np.exp(0.5 * x) + np.random.default_rng(0).normal(0, 0.2, x.size)

res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x", k_star=6)
print(res.summary())                 # parameters +/- standard errors
print(fit_report(res, x, y)["r2"])   # goodness of fit
y_hat = res.predict(x)               # evaluate the fitted model
```
