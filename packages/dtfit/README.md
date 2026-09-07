# dtfit

[![CI](https://github.com/ringavirda/science-nonline/actions/workflows/ci.yml/badge.svg)](https://github.com/ringavirda/science-nonline/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/ringavirda/science-nonline)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-yes-brightgreen)](https://peps.python.org/pep-0561/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**D**ifferential-**t**ransformation **fit**ting: nonlinear smoothing and
forecasting on time-series / Big Data, using methods built in the scheme of
differential (non-Taylor) transformations. Developed as part of a PhD
dissertation on mathematical models of nonlinear smoothing and prediction.

Models can be a SymPy expression string **or** a plain Python callable
`f(x, *params)`; batch fits accept per-point measurement `sigma` (or sklearn
`sample_weight`); and every fit carries its own quality diagnostics (`R^2`,
`AIC`/`BIC`). All are opt-in — left at their defaults, fits are numerically
identical to before.

## Installation

```bash
python -m venv .venv        # create an isolated environment
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate

# from the repo root (this package lives at packages/dtfit):
pip install -e packages/dtfit            # core
pip install -e 'packages/dtfit[viz]'     # + matplotlib plotting helpers
pip install -e 'packages/dtfit[dev]'     # + test/lint tooling

# Optional: the experimental adaptations + experiment suite (a separate
# distribution that depends on dtfit; never shipped in the dtfit wheel):
pip install -e packages/dtfit-experimental             # dtfit_experimental
pip install -e 'packages/dtfit-experimental[bench]'    # + the suite's baselines/plotting
```

A plain `venv` with PyPI wheels is the reference environment and works
identically on Windows and Linux. (Conda is not recommended on Windows: its
MKL-linked numpy only loads its LAPACK DLLs when the env is *activated*, so
tools that call the interpreter directly — VS Code, `pytest` — crash with a
delay-load error.)

### Optional compiled kernels (faster fitting)

The integral-based methods have an optional C backend (`dtfit._core._native`) for
their hot numeric loops — composite-Simpson window integrals and Gauss-Legendre
projections. Build it with clang (needs LLVM and, on Windows, the Visual Studio
Build Tools C++ workload):

```bash
python build_native.py            # compile into src/dtfit/_core/
python build_native.py --clean    # remove the build artifacts
```

It is entirely optional: without it the package falls back to NumPy/SciPy with
identical results (`dtfit._core._kernels.HAVE_NATIVE` reports the active backend).
Building it speeds up the area-based methods substantially — the streaming
`EACFilter` by roughly 6–13× and batch `EAC` by ~3× — while the
already-vectorized Legendre/LSI paths are largely unchanged.

## Quick start

```python
import numpy as np
import dtfit as dt

x = np.linspace(0, 10, 400)
y = ...  # your observations

# Batch fit, numeric (no polynomial pre-fit needed):
result = dt.fit_eac(x, y, "a*atan(w*x)", "x")
print(result.params)

# ...or through the scikit-learn compatible estimator:
reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x", method="lsi")
reg.fit(x, y)
y_hat = reg.predict(x)
```

### Real-time / streaming

```python
flt = dt.EACFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50)
for t, y in stream:           # bounded-cost per-sample update
    flt.partial_fit(t, y)
print(flt.params_)            # tracks time-varying parameters
```

### Picking a model (the model framework)

Most of fitting is *choosing the right structure*. `dtfit.models` is a catalog of
named families that **seed their own `p0`/`bounds` from the data**, compose with
`+`, and can be ranked for you:

```python
from dtfit import models, suggest_models

fit = models.logistic().fit(x, y)            # self-seeded; no p0/bounds to guess
fit = (models.linear() + models.sine()).fit(x, y)   # trend + cycle

for s in suggest_models(x, y)[:3]:           # infer the model from a scored shortlist
    print(s.name, s.r2, s.aic)
```

### Uncertainty & serialization

A `FittingResult` is self-describing — named parameters, uncertainty, and a
JSON-friendly round-trip:

```python
r = dt.fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1, 1])
r.params                       # {'a': ..., 'b': ...}
r.stderr(); r.confidence_intervals(0.95)
r.rsquared, r.aic, r.bic       # fit-quality diagnostics
y_hat, y_std = r.predict(x, return_std=True)   # prediction band
dt.FittingResult.from_dict(r.to_dict())        # save / ship a fitted model

# a Python callable model + per-point sigma both work (numerically opt-in):
r2 = dt.fit_eac(x, y, lambda x, a, b: a * np.exp(b * x),
                sigma=noise_std)               # signature-order params, weighted fit
```

### Diagnostics & visualization

`dtfit.diagnostics` is **fit-aware** (it takes a `FittingResult`, not bare
arrays) and does *not* reimplement `sklearn.metrics` — use those / `scipy.stats`
for plain scalar metrics. It adds what's specific to evaluating a DT fit:
information criteria for model comparison and residual-structure tests, plus the
`*Display` plot helpers (which never call `plt.show()`).

```python
from dtfit.diagnostics import fit_report, residual_diagnostics, FitDisplay

print(fit_report(r, x, y))            # n, rmse, r2, aic, bic, durbin_watson, params±se
print(residual_diagnostics(r, x, y)) # autocorrelation / normality of residuals

FitDisplay.from_estimator(reg, x, y)  # data + fitted curve (needs the viz extra)
```

## Methods

- **LSI** (`method="lsi"`) — least-squares integral; numeric integral-OLS in the
  differential-transformation scheme (successor to DSBI).
- **EAC** (`method="eac"`) — equal-areas criterion; numeric,
  integration-based and noise-robust (successor to DSBE).
- **EACFilter** — recursive/online EAC with NIS drift detection for
  real-time tracking.
- **DSB** (`method="dsb"`) — symbolic differential spectra balance; kept as the
  analytical reference (requires a polynomial fit first in the pipeline).

### Scaling out

- **`ImageStream`** accumulates a signal in fixed `O(order)` memory as it
  arrives; its blocks checkpoint and `assemble` merges them into one
  `Image`, and a channel axis batches many signals through the same
  accumulator:

  ```python
  s = dtfit.ImageStream("legendre", 6, domain=(0, 10), block=200)
  for x, y in chunks_of_the_stream():
      s.update(x, y)
  img = s.assemble(0, 10)
  ```
- `dtfit.fit_many(problems, n_jobs=-1)` fans many independent fits across cores
  (process or threading backend); the threading backend shares memory and
  avoids pickling, nothing more.
- `dtfit.streaming.FilterBank` runs a bank of independent streaming filters
  (one per channel / satellite / axis) for multi-stream real-time tracking.

Further experimental adaptations (pluggable orthogonal bases, robust
overlapping-window ensembles, joint multi-channel fits, stage-wise boosting,
adaptive windows) live in the separate **`dtfit-experimental`** package
(`dtfit_experimental`); their cross-application evaluation is in its experiment
suite ([../dtfit-experimental/src/dtfit_experimental/experiments/cases/REPORTS.md](../dtfit-experimental/src/dtfit_experimental/experiments/cases/REPORTS.md)).

Each method's mathematical grounding (in differential / non-Taylor
transformations), full algorithm, optimizations, guards, applicability, usage
figures and comparison tables are documented in the
[project wiki](https://github.com/ringavirda/science-nonline/wiki/Methods).

Core dependencies: numpy, scipy, sympy, scikit-learn. Plotting helpers require
the optional `viz` extra (matplotlib).
