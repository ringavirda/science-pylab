# API: the forecasting router

`auto_forecast` is the one high-level entry point left: pick a model class from
the series' structure, fit it, extrapolate, and refuse to when the fit cannot
beat a random walk. Recovering *parameters* needs no router -- `fit(model,
data, basis="auto")` fits the candidate bases and keeps whichever leaves the
smallest residual over the samples ([fitting.md](API-Fitting#fit)), and
[`Model.fit`](API-Models) is the self-seeded spelling of the same call.

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
