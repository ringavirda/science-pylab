# API: stochastic-series characterization

A random series has no curve `y = f(t)` to fit; it has **second-order
structure** -- an autocovariance, a spectrum, the variance of block means
across scales, a deterministic mean of trend plus seasonal cycle, the
volatility of its increments. `SecondOrderImage` is the additive sufficient
statistic of exactly that structure, and every estimator, gate, forecaster and
generator on this page reads from it: fit the functional it carries with
dtfit's own machinery, read the process parameter out of the fitted shape.

Concept and validation: [methods/stochastic.md](Methods-Stochastic) and the
[stochastic-series domain report](Domain-Stochastic-Series). This page is the
exact call reference.

- [`SecondOrderImage`](#image) -- the additive second-order statistic
- [read-outs](#readouts) -- the functionals the image carries
- [`SecondOrderStream`](#stream) -- the accumulator and block-stream form
- [estimators](#estimators) -- the individual functional routes
- [`fit_stochastic`](#fit_stochastic) / [`StochasticModel`](#model) -- the batch
  solution (characterize + forecast + generate)
- [`Stochastic`](#stochastic) -- the same in the model `.fit()` convention
- [`StochasticFilter`](#filter) -- the streaming online twin
- [`FORECASTERS`](#forecasters) -- the forecaster names

```python
# the whole surface
from dtfit.stochastic import (fit_stochastic, StochasticModel,
                              StochasticFilter)
from dtfit import stochastic            # submodule namespace
from dtfit.stochastic import (
    SecondOrderImage, SecondOrderStream, sample_acf, hurst_aggvar,
    hurst_spectral, ar1_reversion, ar_order, fit_ar, fractional_difference,
    garch_persistence, cycle_period, decompose_trend_cycle, dickey_fuller,
    FORECASTERS)
from dtfit.models import Stochastic     # also here (catalog convention)
```

---

<a name="image"></a>
## `SecondOrderImage`

```python
SecondOrderImage(lag=256, nfreq=512, scales=10, *,
                 t0=0, x0=0.0, dx=1.0)
```

The additive second-order statistic of a uniformly sampled series: every field
is a sum over samples, so two images of consecutive stretches merge into the
image of their union. Its fields, for a series `y` at global sample indices
`t = t0 .. t0+n-1`:

- `n`, `sum_y`, and the time cross-sums `sum_t`, `sum_ty`, `sum_t2` (the
  trend);
- lagged sums to lag `lag` of the level, the increments `dy_t = y_t -
  y_{t-1}`, the squared increments and the squared level;
- squared block sums at the dyadic scales `2^0 .. 2^scales`, aligned to the
  global index grid, with the partial block at each end;
- the residue accumulator of the DFT on the fixed grid `f_m = m / (2 nfreq)`,
  `m = 0..nfreq-1`;
- the head and tail carries, the first and last `lag + 1` samples.

At `lag=64`, `scales=8`, `nfreq=128`, the state holds 705 accumulator numbers
(4x65 lagged sums + 6x9 block fields + 256 residues + 2x65 carries + 5
moments) plus a version tag, 712 entries in all with the six budget/position
values -- once `n >= lag + 1`; the carries are shorter before that.

| argument | default | meaning |
|---|---|---|
| `lag` | `256` | lag budget in samples, at least 1; the autocovariance, the Blackman-Tukey spectrum and the unit-root statistic read at most this many lags |
| `nfreq` | `512` | frequency grid size, at least 1; the DFT bins are `f_m = m / (2 nfreq)` |
| `scales` | `10` | scale budget, at least 0; block sums are kept at `2^0 .. 2^scales` samples |
| `t0` | `0` | global sample index of the first sample |
| `x0` | `0.0` | position of global index 0 |
| `dx` | `1.0` | sample spacing in position units, strictly positive |

**Construction and updates**

- `SecondOrderImage.of(data, *, lag=256, nfreq=512, scales=None)` -- the image
  of a whole record: an `Original` on a uniform grid, or a 1-D array at unit
  spacing from position 0. `scales=None` takes `floor(log2 n) - 3`, the
  largest scale with at least eight blocks. `of` treats `nfreq` as a floor: it
  raises it to `2^ceil(log2 n) / 2`, capped at 4096 bins, so the seasonal
  read-out resolves the record it is given -- 8192 residues, 64 KB, is the
  ceiling on the image's largest field. Raises `ValueError` on fewer than two
  samples or a non-uniform grid: the sampling must be uniform.
- `update(y)` -- add the next chunk of samples in place and return self. The
  chunk continues the series, its first sample at global index `t0 + n`.
- `merge(other)` -- a new image of both records, exact: the lagged sums gain
  the cross terms between this image's tail carry and the other's head carry.
  The two images merge only when consecutive (`other.t0 == self.t0 +
  self.n`), on the same budgets and sample grid.
- `state()` / `restore(d)` -- the complete state as JSON-serializable data,
  and loading it back.

```python
import numpy as np
from dtfit.stochastic import SecondOrderImage

x = np.random.default_rng(0).standard_normal(4000)
img = SecondOrderImage.of(x)
print(img.n, img.lag, img.nfreq)
```

---

<a name="readouts"></a>
## Read-outs

Every read-out below is exact under merging: an image assembled from blocks
gives the same numbers as one built from the whole record.

| read-out | returns | reads |
|---|---|---|
| `acov()` | `gamma[0..lag]` | centred autocovariance of the level, divided by `n` |
| `acov_increments()` | `gamma_dy[0..lag]` | of the increments, divided by `n - 1` |
| `acov_volatility()` | `gamma_d2[0..lag]` | of the squared increments, the volatility functional of the returns |
| `acov_squares()` | `gamma_y2[0..lag]` | of the squared level, the volatility functional of a stationary series |
| `detrended_acov()` | `gamma_e[0..lag]` | residual about the least-squares line |
| `residual_acov(freq=None, coef=None)` | `gamma_e[0..lag]` | residual about the line plus the seasonal harmonics at `freq` with coefficients `coef` |
| `trend()` | `(slope, intercept)` | least-squares line `y ~ intercept + slope * x`, in position units |
| `detrended_var()` | float | residual variance about the line, `RSS / n` |
| `aggregated_variance()` | `(m, var)` | block-mean variance against block size, over scales with at least eight complete blocks |
| `dft()` | complex array, length `nfreq` | the DFT of the record on `f_m = m / (2 nfreq)` |
| `spectrum(n_freq=None)` | `(f, S)` | Blackman-Tukey spectrum under a Parzen lag window |
| `seasonal(max_harmonics=1, *, freq=None, halfwidth=4)` | dict | `period`, `freq`, `amp`, `phase`, `coef`, `n_harmonics`, `strength` of the fundamental, from the Dirichlet-kernel fit to the fixed-grid DFT |
| `dickey_fuller(lags=None)` | float | the augmented Dickey-Fuller `tau` statistic, from the Toeplitz normal equations of the autocovariances |
| `mean()`, `first()`, `last()`, `carry()`, `domain` | float / array / tuple | the sample mean, the first and last sample, the last `lag + 1` samples, and `(x0, x1)` |

`seasonal`'s phase reference is the global sample index `t`, so `coef` and
`phase` describe the record wherever it sits in a stream. Its frequency grid
resolves `1 / (2 nfreq)`; a record longer than `2 nfreq` samples puts several
of the record's own frequencies inside one grid bin and warns rather than
under-reading silently -- a stream fixes its grid before it knows the record
length. On the period-50 line `0.02 t + 3 sin(2 pi t / 50) + N(0, 1)`, twenty
seeds, `seasonal` reads period 0.065 percent, amplitude 1.1 percent and phase
0.035 rad off at `n = 600`, and 0.010 percent, 0.46 percent and 0.020 rad off
at `n = 4000` on the grid `of` raises to 2048 bins. On the noiseless line
`3 sin(2 pi t / 50) + 0.5`, the fixed 512-bin grid reads period 50.61 and
amplitude 2.23 at `n = 4000`, versus 49.996 and 3.000 on the grid `of`
chooses.

```python
import numpy as np
from dtfit.stochastic import SecondOrderImage

rng = np.random.default_rng(0)
t = np.arange(600.0)
y = 3.0 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 1, 600)
img = SecondOrderImage.of(y)
seas = img.seasonal(max_harmonics=1)
print(round(seas["period"], 1))
```

---

<a name="stream"></a>
## `SecondOrderStream`

```python
SecondOrderStream(block=None, *, lag=256, nfreq=512, scales=10,
                  x0=0.0, dx=1.0, keep_fine=64, fold=16)
```

A running second-order image, or a stream of block images: the batch
counterpart of [`StochasticFilter`](#filter). Blocks are exact and mergeable
and resolve a change at block granularity, where the filter's exponentially
weighted statistics resolve it within about a half-life.

| argument | default | meaning |
|---|---|---|
| `block` | `None` | samples per block, at least 2; `None` accumulates one running image instead |
| `lag`, `nfreq`, `scales` | `256`, `512`, `10` | the block images' budgets, as [`SecondOrderImage`](#image) |
| `x0`, `dx` | `0.0`, `1.0` | position of global sample index 0, and the sample spacing |
| `keep_fine` | `64` | block images kept at block resolution |
| `fold` | `16` | fine blocks merged into one coarse block once there are more than `keep_fine` |

**Accumulator mode** (`block=None`): `update(x)` folds a chunk into the
running image, `image()` reads it.

**Block mode** (`block` a sample count): `update(x)` returns each block image
that just finished; `close()` finishes a partial block that holds at least two
samples; `blocks(t0, t1)` lists the stored block images whose span lies inside
`[t0, t1]`; `assemble(t0, t1)` merges them into one image, exact when the
blocks are consecutive. Retention keeps the last `keep_fine` blocks at block
resolution and folds older ones into coarse blocks of `fold`.

`checkpoint()` / `resume(state)` save and reload the complete stream state,
bit-exact.

```python
import numpy as np
from dtfit.stochastic import SecondOrderImage, SecondOrderStream

rng = np.random.default_rng(0)
x = rng.standard_normal(4000)
stream = SecondOrderStream(500, lag=64, nfreq=64, scales=6)
for start in range(0, 4000, 137):
    stream.update(x[start:start + 137])
stream.close()
whole = SecondOrderImage(64, 64, 6).update(x)
merged = stream.assemble(0.0, 4000.0)
print(bool(np.max(np.abs(whole.acov() - merged.acov())) < 1e-9))
```

---

<a name="estimators"></a>
## Estimators -- the individual functional routes

Each takes a series, an `Original` or a [`SecondOrderImage`](#image) --
`as_image` builds one if it is not one already, none of these estimators
taking lag or frequency budgets of its own. Most recover a stochastic-model
parameter by feeding a functional read off the image to
[`fit_lsi`](API-Fitting#fit_lsi) / [`fit_eac`](API-Fitting#fit_eac).
`method="lsi"` (default) / `"eac"` pick the engine; `"ols"` / `"acf1"` /
`"yw"` are plain baselines. The AR helpers (`ar_order` / `fit_ar`) are direct
Yule-Walker instead, `fractional_difference` is a transform on the raw
series, and `decompose_trend_cycle` needs the axes themselves -- it takes
`(t, y)` or an `Original`, raising `ValueError` on a bare image with `y`
omitted. Together the AR helpers and `fractional_difference` let the router
tell a finite-order AR(p) apart from genuine long memory.

| function | recovers | functional fit |
|---|---|---|
| `hurst_spectral(data, *, n_freq=None, method="lsi")` | Hurst `H`, `d` | LSI slope of the image's low-frequency Blackman-Tukey spectrum (GPH) |
| `hurst_aggvar(data, *, method="lsi")` | Hurst `H`, `d` | power-law fit of the image's aggregated variance |
| `ar1_reversion(data, *, nlags=None, method="yw")` | AR(1) `phi`, `tau`, `halflife` | `"yw"` reads `gamma_1 / gamma_0` directly; `"lsi"` / `"eac"` fit an exponential to the autocorrelation |
| `garch_persistence(data, *, nlags=None, method="lsi")` | persistence `alpha+beta`, `tau` | exponential fit to the autocorrelation of the squared level |
| `cycle_period(data, *, nlags=None)` | cycle `period`, `w`, `damping` | damped-cosine fit to the autocorrelation (oscillatory recipe) |
| `decompose_trend_cycle(t, y=None, *, max_harmonics=4, with_cycle=True)` | `slope`, `period`, `amp`, fitted `trend`/`cycle`/`residual`, a `forecast(h, dt)` closure | the image's `trend()` and `seasonal()`, leaving a stochastic residual |
| `dickey_fuller(data, *, lags=None)` | `tau`, `pvalue` | the image's Dickey-Fuller statistic and its MacKinnon p-value |
| `ar_order(data, *, max_order=8, ic="aic") -> int` | AR order `p` | Yule-Walker AR(k) on the image's autocorrelation for `k=0..max_order`, order chosen by `ic` |
| `fit_ar(data, order=None, *, max_order=8, ic="aic") -> dict` | `order`, `phi`, `sigma` | Yule-Walker AR(p) off the image's autocorrelation (order auto-selected if `None`) |
| `fractional_difference(x, d, *, ntrunc=None) -> ndarray` | the differenced series `(1-B)^d x` | truncated binomial `(1-B)^d` filter on the raw series -- whitens ARFIMA long memory |
| `sample_acf(data, nlags) -> ndarray` | the biased sample autocorrelation `rho[0..nlags]`, read off the image | |

```python
import numpy as np
from dtfit.stochastic import hurst_spectral, ar1_reversion

returns = np.random.default_rng(0).standard_normal(2000)
hurst_spectral(returns)["H"]        # long-memory exponent
ar1_reversion(returns)["phi"]        # mean-reversion speed
```

Measured on the AR(1), GARCH and self-similar model catalog: the
Yule-Walker AR(1) route has median error 0.013 against 0.019 for the
exponential fit to the ACF; GARCH persistence 0.0056 against 0.0057;
aggregated-variance Hurst 0.039 against 0.043; the Blackman-Tukey spectral
Hurst has bias +0.006 and RMSE 0.056 at `lag=256` against +0.012 and 0.063
for the raw periodogram, and bias -0.060 at `lag=64`.

---

<a name="fit_stochastic"></a>
## `fit_stochastic`

```python
fit_stochastic(data, t=None, *, period=None, max_harmonics=4,
               forecaster="auto", trend_t=3.0, cycle_strength=0.08,
               min_cycles=2.5, lm_hurst=0.68, mr_phi=0.15,
               vol_persist=0.60, lag=256, nfreq=512) -> StochasticModel
```

Characterize an arbitrary series across every route at once and return a
single coherent [`StochasticModel`](#model). `data` is a series, an
`Original`, or a [`SecondOrderImage`](#image) directly -- `lag` and `nfreq`
size the image built from a series and are ignored when `data` already is
one. Every gate below reads that image; the routes are composed in the order
the second-order theory dictates, each behind a **significance gate**:

1. **unit-root gate** -- the augmented Dickey-Fuller `tau` statistic, read
   off the image's autocovariances (`SecondOrderImage.dickey_fuller`) with
   the augmentation lag chosen by AIC over
   `0..min(12 (n/100)^0.25, 12, n // 3, lag - 2)`, and its MacKinnon
   p-value. Below `n = 40` the gate reports "not nonstationary"
   unconditionally rather than trust a verdict from that few samples. An
   I(1) level (random walk) is *differenced* and reported as such, not
   given a spurious trend / cycle / long memory.
2. **deterministic mean** -- the image's least-squares trend (kept only if
   its Newey-West `|t|` exceeds `trend_t` *and* it explains real variance)
   and the image's fixed-grid-DFT seasonal read-out, a multi-harmonic Fourier
   fit (kept only on a genuine repeating spectral peak: `> cycle_strength`
   of the power, repeating `>= min_cycles` times; period detected or
   supplied via `period=`).
3. **long memory** -- the residual's Blackman-Tukey spectrum's GPH slope,
   `H > lm_hurst`. If it fires, a veto: whiten with a Yule-Walker AR(p)
   (order up to 3, chosen off the residual autocorrelation by AIC) and
   recheck the GPH slope of the whitened spectrum against a stricter
   threshold `0.5 + 0.5 * (lm_hurst - 0.5)`, above 128 samples only, so a
   near-unit-root AR(1) is not mislabelled long memory.
4. **mean reversion** (`mr_phi < phi < 0.99`, lag-1 autocorrelation
   significant) and **volatility clustering** -- the excess autocorrelation
   `rho_2 - rho^2` of the squared level over the residual's own
   autocorrelation `rho`, tested for persistence (`> vol_persist`).

A series with no gate open is reported as `regime="white noise / random
walk"`.

**Forecasting is RMSE-optimal backtest model selection** -- a
regime-informed candidate set (random walk, drift, mean reversion, the
image's least-squares trend, two multi-harmonic seasonal continuations)
is rolling-origin backtested and the best kept; the choice is in
`model.forecaster_name`. Backtesting refits each candidate's image on
successive training folds, so it needs the raw series: a fit from a
[`SecondOrderImage`](#image) alone cannot backtest, and the chosen name
carries `" (no backtest)"`.

| argument | default | meaning |
|---|---|---|
| `data` | -- | a series, an `Original`, or a `SecondOrderImage` |
| `t` | `None` | time index when `data` is a series; `None` defaults to `0..n-1`. Periods are detected in **sample units** and converted to `t` units via the median spacing |
| `period` | `None` | seasonal period **in samples** (index steps of the record, not `t` units), else detected from the spectrum |
| `max_harmonics` | `4` | cap on the Fourier harmonics (count chosen by BIC) |
| `forecaster` | `"auto"` | `"auto"` backtest-selects; a name from [`FORECASTERS`](#forecasters) forces one; a callable `(train, h) -> array` is used directly; a list of names/callables is a custom candidate set |
| `trend_t`, `cycle_strength`, `min_cycles`, `lm_hurst`, `mr_phi`, `vol_persist` | -- | the detection gates (above). Tuned defaults; override per series |
| `lag`, `nfreq` | `256`, `512` | budgets of the image built from `data`; ignored when `data` is already a `SecondOrderImage` |

```python
import numpy as np
from dtfit.stochastic import (
    SecondOrderImage, SecondOrderStream, fit_stochastic)

rng = np.random.default_rng(0)
x = np.zeros(4000)
for t in range(1, 4000):
    x[t] = 0.8 * x[t - 1] + rng.normal(0, 1.0)

img = SecondOrderImage.of(x)
g = img.acov()
print(round(float(g[1] / g[0]), 3))          # the AR(1) coefficient
print(fit_stochastic(img).regime)
```

Unit-root verdicts agree with the reference statistic in 139 of 140 measured
series; regime identification is 95 percent over the seven process families
at twenty seeds, against 96 percent for the per-sample pipeline it replaces.

---

<a name="model"></a>
## `StochasticModel`

The unified second-order characterization returned by `fit_stochastic`. A
dataclass of detected components + recovered parameters, plus a forecaster
and a generator.

**Fields** (each detection is behind a gate, so white noise yields none):

| field | meaning |
|---|---|
| `regime` | the primary regime label (`"trend+seasonal"`, `"mean-reverting"`, `"long-memory"`, `"random walk + drift"`, `"white noise / random walk"`, ...) |
| `components` | tuple of detected components (`("trend", "seasonal")`, `("none",)`, ...) |
| `forecaster_name` | the backtest-selected forecaster; below 51 training samples the name carries `" (short-series fallback)"`, and a fit from a `SecondOrderImage` alone carries `" (no backtest)"`, so an untested choice is always visible |
| `trend_slope`, `has_trend` | linear trend |
| `cycle_period`, `cycle_amp`, `has_cycle`, `n_harmonics`, `seasonal` | cycle / seasonal |
| `hurst`, `has_long_memory` | long memory (`d = hurst - 0.5`) |
| `ar1_phi`, `has_mean_reversion` | AR(1) mean reversion |
| `vol_persistence`, `has_vol_clustering` | GARCH-type volatility persistence |
| `sigma`, `sigma_walk` | one-step innovation std / random-walk (first-difference) scale |
| `n`, `level` | length / mean |

**Methods**

- `forecast(h, *, return_conf_int=False, alpha=0.05)` -- forecast `h` steps with
  the selected forecaster. With `return_conf_int` returns `(point, lower, upper)`
  whose band growth matches the forecaster (bounded for mean reversion, `~sqrt(h)`
  for a random walk / drift, widened to `~h^(2H)` when the chosen random walk /
  drift forecaster sits on a model that also carries long memory).
- `simulate(n=None, *, seed=None, rng=None, dist="normal", df=7.0) -> ndarray` --
  draw a **fresh realization** from the detected components: the deterministic mean
  plus a residual matched to the regime (AR(1) / ARFIMA long memory / GARCH /
  integrated walk / white noise). Re-fitting a simulated path recovers the same
  regime -- the honest test that the model is a faithful generator. Pass
  `dist="t"` (with `df` degrees of freedom) for **fat-tailed** Student-t
  innovations -- a heavier-tailed generator for financial-style returns, so a
  simulated path or forecast band reflects real tail risk instead of understating
  it with a Gaussian.
- `fingerprint() -> dict` -- the detected structure as a flat `{name: value}` map
  (for tables).
- `summary() -> str` -- a human-readable multi-line summary.

```python
import numpy as np
from dtfit.stochastic import fit_stochastic

t = np.arange(600.0)
rng = np.random.default_rng(0)
y = 0.02 * t + 3 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 1, 600)
m = fit_stochastic(y)
pt, lo, hi = m.forecast(40, return_conf_int=True)   # forecast + 95% band
sim = m.simulate(600, seed=1)                        # a new series, same structure
m.fingerprint()                                      # {'regime': ..., 'trend slope': ..., ...}
```

---

<a name="stochastic"></a>
## `Stochastic`

```python
Stochastic(*, period=None, max_harmonics=4, forecaster="auto",
          lag=256, nfreq=512, **gates)
Stochastic.fit(x, y=None) -> StochasticModel
```

The stochastic-series model in the catalog `.fit()` convention -- the same
ergonomics as the deterministic families (`dtfit.models.logistic().fit(x, y)`),
but it characterizes a random *series* instead of fitting a `y = f(x)` curve. A
thin wrapper over `fit_stochastic`; constructor arguments mirror it. `**gates`
forwards the detection-gate overrides.

`fit` accepts `fit(series)` (uniform unit time) or `fit(t, series)` (explicit time
index). It returns the fitted [`StochasticModel`](#model), also stored on
`.model_`. It is **not** a `Model` subclass (it has no sympy expression) and is
**not** in the AIC `CATALOG` -- it just shares the calling convention. Available as
`dtfit.models.Stochastic`.

```python
import numpy as np
from dtfit.models import Stochastic

t = np.arange(600.0)
data = 0.02 * t + np.random.default_rng(0).normal(0, 1, 600)
m = Stochastic().fit(data)                   # -> a fitted StochasticModel
m = Stochastic(period=12, forecaster="trend+seasonal").fit(t, data)
print(m.regime, m.forecaster_name)
```

---

<a name="filter"></a>
## `StochasticFilter`

```python
StochasticFilter(nlags=24, halflife=150.0, warmup=80, settle=500, z_thresh=4.0)
```

The **per-input streaming twin** of `fit_stochastic`'s second-order stage -- the
stochastic counterpart of [`EACFilter` / `LSIFilter`](API-Streaming). It maintains
EWMA autocovariances of the level and of `|level - mean|` in `O(K)` per sample (the
running ACF), and reads the parameters in **closed form** using dtfit's principles
in streaming form: the **EAC equal-areas criterion** for the AR(1) persistence and
the volatility persistence, and the **AR(2) characteristic roots** for the cycle --
no per-sample optimization, no batch fit. A two-timescale fused statistic flags
structural breaks (a persistence jump, a volatility switch), once per change, at a
low false-alarm rate. Flat memory, bounded per-sample cost.

> **Scope.** This is the online twin of the *second-order* stage only. Long memory
> (the spectral Hurst) and the unit-root gate need a full spectral sweep or a
> Toeplitz solve over the maintained lag window each time they are read, not the
> filter's closed-form per-sample update, so they are not tracked here.

| argument | default | meaning |
|---|---|---|
| `nlags` | `24` | autocovariance lags maintained (the memory footprint) |
| `halflife` | `150.0` | EWMA half-life (samples) -- how fast the characterization adapts |
| `warmup` | `80` | samples before the detector is active |
| `settle` | `500` | samples after `warmup` to calibrate the detector's in-control baseline before it may flag |
| `z_thresh` | `4.0` | fused-statistic threshold (sigmas) for a flag |

**Methods & attributes**

- `update(x) -> self` -- ingest one sample. `partial_fit(xs)` ingests a **batch**
  (an array of samples, looped through `update`) -- note this differs from
  [`EACFilter` / `LSIFilter`](API-Streaming), whose `partial_fit` takes a **single**
  sample.
- `params_ -> dict` -- current `{level, sigma, ar1_phi, cycle_period, vol_persistence, n}`.
- `snapshot() -> dict` -- `params_` plus a coarse online `regime` label.
- `predict(h) -> ndarray` -- forecast `h` steps by AR(1) mean reversion at the snapshot.
- `n_flags_` -- structural breaks detected so far; `flag_times_` -- a bounded ring
  of recent break sample indices; `last_flag_` -- the most recent break index.

```python
import numpy as np
from dtfit.stochastic import StochasticFilter

f = StochasticFilter(halflife=200)
for x in np.random.default_rng(0).standard_normal(50):
    f.update(x)
    if f.last_flag_ == f.params_["n"]:
        print("regime change at sample", f.last_flag_, "->", f.snapshot()["regime"])
```

Measured on a tracked AR(1) coefficient: the filter reaches RMSE 0.032 and
block images 0.028.

---

<a name="forecasters"></a>
## `FORECASTERS`

The built-in forecaster names accepted by `fit_stochastic(..., forecaster=...)`:

```python
FORECASTERS == ("random walk", "drift", "mean-reversion", "trend",
                "seasonal", "trend+seasonal")
```

Force one with a string, plug your own with a callable `(train, h) -> array`, or
pass a list of candidates to backtest-select among. `"seasonal"` /
`"trend+seasonal"` require a known `period` (detected or via `period=`).
