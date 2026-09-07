# API: one-call entry points

Two high-level "just fit it" functions distilled from the domain studies, whose
main finding was that **picking the structurally-correct model/estimator variant
is the biggest lever** -- not the solver. These compose only validated, stable
pieces behind a single call.

- [`auto_estimate`](#auto_estimate) -- recover physical parameters, routing by signal shape
- [`auto_forecast`](#auto_forecast) -- structured fit-then-extrapolate forecast, with safety guards

---

<a name="auto_estimate"></a>
## `auto_estimate`

```python
auto_estimate(x, y, expr, var, *,
              shape="auto", freq_param=None,
              p0=None, bounds=None, param_names=None) -> FittingResult
```

Recover the parameters of `expr` by routing to the estimator that fits the
signal's *shape* (the parameter-estimation study's merged selector).

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `x`, `y` | array | -- | observed samples |
| `expr` | str \| callable | -- | model expression: a SymPy-expression string **or** a plain Python callable `f(x, *params)` (resolved via [`resolve_model`](API-Fitting#also-exported-from-dtfitmethods) and forwarded to whichever base fitter the shape routes to) |
| `var` | str | -- | main variable (a label only for a callable) |
| `shape` | str | `"auto"` | which variant to use (see table below) |
| `freq_param` | str \| None | `None` | angular-frequency parameter name; forwarded to the LSI oscillatory recipe and **implies an oscillatory shape** |
| `p0` | array \| dict \| None | `None` | initial guess -- positional or `{name: value}` dict, forwarded verbatim to the fitters |
| `bounds` | list[(lo, hi)] \| dict \| (lo, hi) \| None | `None` | per-parameter bounds -- any form the fitters accept (pair list, partial `{name: (lo, hi)}` dict, scipy tuple) |
| `param_names` | tuple[str] \| None | `None` | parameter names for a **callable** `expr` whose signature cannot be introspected (an `f(x, *params)` model or a builtin); forwarded to the base fitters. Optional (but validated) for a symbolic `expr` |

**`shape` routing**

| `shape` | routes to |
|---|---|
| `"auto"` | detect a cycle (FFT power share > 0.10) -> oscillatory; else `"bulk"`. `freq_param` given => oscillatory |
| `"oscillatory"` | [`fit_lsi`](API-Fitting#fit_lsi) with the oscillatory recipe |
| `"transient"` / `"peak"` | [`fit_eac`](API-Fitting#fit_eac) (the block image) |
| `"robust"` | [`fit_eac`](API-Fitting#fit_eac) with `robust=True` |
| `"bulk"` | fit both LSI and EAC, keep the lower in-sample RMSE |

A bulk candidate that fails emits a `UserWarning` naming the fitter and the
error; if **both** base fits fail, the raised error carries both underlying
messages.

**Returns** the `FittingResult` from the selected estimator.

```python
from dtfit import auto_estimate
res = auto_estimate(x, y, "a*atan(w*x)", "x", shape="transient")
res = auto_estimate(x, y, "A*sin(w*x + p)", "x", freq_param="w")   # oscillatory
res = auto_estimate(x, y, "a*exp(b*x)", "x")                       # auto > bulk

# a plain Python callable model works too; param_names only needed when
# the signature can't be introspected (an f(x, *params) model or a builtin):
def model(x, a, b):
    return a * np.exp(b * x)
res = auto_estimate(x, y, model, "x")                              # signature-order params
```

Honest ceiling: on clean bulk shapes `auto_estimate` *matches* but does not beat a
well-initialized NLLS.

---

<a name="auto_forecast"></a>
## `auto_forecast`

```python
auto_forecast(x, y, horizon, *,
              model="auto", period=None,
              seasonal=True, season_strength=0.05) -> ForecastResult
```

A structured fit-then-extrapolate forecaster: route the model class, fit it, and
extrapolate `horizon` steps past `x` on its uniform grid. Two safety guards keep
it honest. The return value is a [`ForecastResult`](#forecastresult) -- an
`np.ndarray` subclass, so **every existing caller keeps working unchanged** (it
*is* the length-`horizon` forecast) while also carrying the fit provenance and an
optional uncertainty band.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `x`, `y` | array | -- | observed series (`x` (near-)uniformly sampled) |
| `horizon` | int | -- | number of future steps to forecast |
| `model` | str | `"auto"` | `"auto"` (route by structure) or one of `"logistic"`, `"linear"`, `"poly"`, `"linear_seasonal"`, `"random_walk"`. Any other string **raises `ValueError`** (the closed set is validated up front -- an unrecognised name is no longer silently treated as a quadratic) |
| `period` | float \| None | `None` | known seasonal period (in samples) for the seasonal fit |
| `seasonal` | bool | `True` | consider a seasonal model under `"auto"` |
| `season_strength` | float | `0.05` | minimum detected cycle strength to pick a seasonal model |

**Routing (`model="auto"`)**: saturating positive growth -> `logistic`; a detected
cycle -> `linear_seasonal` (linear + sine); otherwise a quadratic level (`poly`).

**The guards**

- **No-structure guard** -- if the structured model can't get near naive
  persistence on a held-out tail of the *training* data (the near-random-walk
  signature), it falls back to persisting the last value.
- **Divergence guard** -- a runaway quadratic (extrapolating far outside the data
  range) is dropped to a linear fit.

**Returns** a [`ForecastResult`](#forecastresult) -- an `np.ndarray` of the
length-`horizon` forecast (values at the extrapolated x grid) that also carries its
provenance.

```python
from dtfit import auto_forecast
future = auto_forecast(x, y, horizon=30)               # auto-routed + guarded
future = auto_forecast(x, y, horizon=30, model="logistic")

future + 0.0                # still a plain array: indexing, arithmetic, np.* all work
future.model_name           # e.g. "logistic" or "linear (poly diverged)"
if future.std_band is not None:
    band = (future - future.std_band, future + future.std_band)   # 1-sigma band
```

Honest ceiling: near-random-walk series fall back to persistence -- by design, not
defect (see [../guides/README.md](Guides) Sec.6).

<a name="forecastresult"></a>
### `ForecastResult`

The value returned by `auto_forecast` (exported top-level as `dtfit.ForecastResult`).
It **subclasses `np.ndarray`**, so it *is* the length-`horizon` forecast -- every
existing use (indexing, `.shape`, `len`, arithmetic, `np.allclose`, `np.isfinite`,
`np.std`) keeps working. It additionally carries where the numbers came from:

| attribute | type | meaning |
|---|---|---|
| `model_name` | str | the model that actually produced the forecast, **with fallback provenance** -- `"logistic"` for a clean fit, `"linear (poly diverged)"` when the divergence guard dropped a runaway quadratic, `"linear (logistic failed)"` when the primary fit raised, `"random_walk"` / `"persistence (...)"` on the persistence paths |
| `result` | [`FittingResult`](API-Types) \| None | the underlying fit when the forecast came from a real fit; `None` on the persistence / random-walk paths |
| `std_band` | ndarray \| None | a length-`horizon` 1-sigma prediction band (delta method) when the fit exposed a covariance and the propagation succeeded; `None` otherwise. Named `std_band` (**not** `std`) so it does not shadow `numpy.ndarray.std` -- `fc.std()` and `np.std(fc)` keep working |
| `index` | pandas Index \| None | the length-`horizon` **future** index continuing the `x` passed to `auto_forecast` (a `DatetimeIndex` extended by its inferred frequency, an integer/`RangeIndex` by its step); `None` when `x` was not pandas / not extendable |
| `.to_series()` | method | the pandas "in -> out" view: a `Series` of the forecast values indexed by `.index` (raises if `.index` is `None`) |

> The per-step `index` and `std_band` are dropped from a **slice or reduction**
> (`fc[:3]`, `fc.sum()`) since they no longer align -- `fc[:3].std_band` is `None`,
> not a misaligned length-`horizon` band. `model_name` / `result` still carry
> forward.
