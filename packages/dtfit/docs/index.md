# dtfit

**Differential-transformation fitting** -- methods for fitting models that are
*nonlinear in their parameters* (exponential, transcendental, mixed) for
nonlinear smoothing, parameter estimation and forecasting, built in the scheme of
differential / non-Taylor transformations.

dtfit's edge is not one-shot batch curve fitting -- for that, use
`scipy.optimize.curve_fit` with a good initial guess (see
[dtfit vs SciPy](comparison.md)). dtfit is for the cases where SciPy is awkward or
does not apply:

- **Streaming / recursive** parameter tracking (`partial_fit`, one sample at a
  time; `ImageStream` accumulates the fitting statistic in fixed memory).
- **Out-of-memory** and **many-channel** fitting (block images assembled onto a
  coarse domain, channels batched through one shared GEMM).
- **Embedded / real-time** estimation with predictable per-step cost.
- **Robustness without tuning** -- self-seeding models, integral (area) criteria
  that denoise by construction, and the robust image for outlier-contaminated
  samples.

It is complementary to SciPy / lmfit, not a replacement.

---

## Install

```bash
pip install dtfit
```

Optional extras:

```bash
pip install "dtfit[viz]"    # matplotlib for the diagnostics Displays
pip install "dtfit[docs]"   # build this documentation site
```

Core dependencies are NumPy, SciPy, SymPy and scikit-learn. pandas is an
*optional* interop dependency -- every ndarray path works without it, and pandas
`Series` inputs are accepted (and returned) when it is installed.

---

## Quickstart

### A batch fit with `fit_lsi`

Fit an exponential `y = a * exp(b * x)` to noisy data and read the recovered
parameters, their standard errors and the fit quality straight off the result:

```python
import numpy as np
from dtfit import fit_lsi

rng = np.random.default_rng(0)
x = np.linspace(0.0, 2.0, 80)
y_true = 1.5 * np.exp(0.8 * x)
y = y_true + rng.normal(0.0, 0.05 * y_true.std(), x.size)

fit = fit_lsi(x, y, "a*exp(b*x)", "x")

print(fit.params)      # {'a': 1.4991, 'b': 0.8027}
print(fit.stderr())    # {'a': 0.0112, 'b': 0.0048}
print(fit.rsquared)    # 0.997763
```

The returned [`FittingResult`](api/batch-fitting.md#dtfit.FittingResult) is
self-describing -- named parameters, a covariance-derived uncertainty, and fit
diagnostics:

```python
print(fit.aic, fit.bic)          # -397.83  -393.06
print(fit.confidence_intervals())
print(fit.summary())
# FittingResult: a*exp(b*x)
#   a = 1.49913 +/- 0.0112
#   b = 0.802796 +/- 0.00482
#   R^2 = 0.997763
```

### A callable model

The model can be a SymPy string, a `sympy.Expr`, or a plain Python callable
`f(x, *params)`. For a callable the parameters follow the signature order:

```python
import numpy as np
from dtfit import fit_lsi

def model(x, a, b):
    return a * np.exp(b * x)

fit = fit_lsi(x, y, model, "x")
print(fit.params)      # {'a': 1.4991, 'b': 0.8027}
```

A callable carries no expression, so its result cannot be `to_dict`-serialized,
but `predict` (including error bands) still works via the numeric evaluator.

### One image, several models

`fit_lsi` and `fit_eac` build the image for you; imaging the data once with
`Original.image` and calling `.fit()` on it directly reuses that same
projection for as many candidate models as you like:

```python
from dtfit import Original

img = Original(x, y).image("legendre", order=8)
lin = img.fit("a0 + a1*x", "x")
exp = img.fit("a0 + a1*exp(a2*x)", "x")
print(lin.aic, exp.aic)   # compare candidates on the same image
```

### Per-point uncertainty (`sigma`)

Weight the fit by per-point noise -- pass a `sigma` array (larger = less trusted),
exactly like `scipy.optimize.curve_fit`. Here we down-weight a noisier tail:

```python
sigma = np.where(x > 1.0, 1.0, 0.1)   # trust the first half 10x more
fit = fit_lsi(x, y, "a*exp(b*x)", "x", sigma=sigma)
print(fit.params)      # {'a': 1.4855, 'b': 0.8131}
```

Pass `absolute_sigma=True` when `sigma` is an absolute standard deviation (so the
covariance is not rescaled by the reduced chi-square).

### A structured forecast

`auto_forecast` routes the model class by structure (saturating growth ->
logistic; a detected cycle -> linear+seasonal; otherwise a polynomial level),
guards against runaway extrapolation, and reports what it did:

```python
from dtfit import auto_forecast

fc = auto_forecast(x, y, horizon=10)

print(fc.model_name)   # 'linear_seasonal'  (the model it chose)
print(fc[:3])          # [7.4240036  7.53862295 7.65241176]  -- fc *is* an ndarray
print(fc.std_band[:3]) # [0.0608103  0.069421   0.07887614] -- 1-sigma band when available
```

`fc` is an `ndarray` subclass, so it indexes and broadcasts like any array while
also carrying `.model_name`, `.result` (the underlying `FittingResult`),
`.std_band` and `.index`.

### pandas in -> pandas out

When pandas is installed, `Series` / single-column `DataFrame` inputs are accepted
and predictions come back aligned to the input index:

```python
import pandas as pd
from dtfit import fit_lsi

idx = pd.date_range("2021-01-01", periods=x.size, freq="D")
xs, ys = pd.Series(x, index=idx), pd.Series(y, index=idx)

fit = fit_lsi(xs, ys, "a*exp(b*x)", "x")
pred = fit.predict(xs)          # -> pandas Series, indexed by idx
print(type(pred).__name__)      # 'Series'
print(pred.head(3))
# 2021-01-01    1.499127
# 2021-01-02    1.529907
# 2021-01-03    1.561319
# Freq: D, dtype: float64
```

`auto_forecast` goes one step further: give it a pandas `x` with an extendable
index and `ForecastResult.to_series()` returns the forecast on the *future*
index:

```python
fc = auto_forecast(xs, ys, horizon=14)
print(fc.to_series().head(3))   # future dates -> forecast values
```

---

## Where to go next

- **[Choosing a method](guide/choosing-a-method.md)** -- a decision guide for
  method, model and knobs.
- **[LSI](guide/lsi.md)** and **[EAC](guide/eac.md)** -- the two batch fitters,
  their math, and when to use each.
- **[dtfit vs SciPy](comparison.md)** -- an honest, measured head-to-head.
- **[API reference](api/batch-fitting.md)** -- the full public surface.
- **[Versioning & deprecation](versioning.md)** -- stability guarantees.

## The API in one glance

| Tier | You do this | Entry points |
|---|---|---|
| **Methods** | choose the engine | `fit`, `fit_lsi`, `fit_eac` |
| **Estimator** | plug into scikit-learn | `NonlineRegressor` |
| **Models** | pick a shape, not a formula | `models`, `Model`, `suggest_models`, `register` |
| **High-level** | let dtfit choose | `fit(..., basis="auto")`, `auto_forecast` |
| **Streaming** | track live parameters | `ImageFilter`, `LSIFilter`, `EACFilter` |
| **Scale** | run big / many | `ImageStream`, `fit_many` |
| **Stochastic** | genuinely random series | `fit_stochastic`, `StochasticModel` |
