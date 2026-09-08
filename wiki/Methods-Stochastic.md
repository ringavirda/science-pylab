# Stochastic-series characterization -- mathematical reference

dtfit fits a deterministic `y = f(t; theta)`. A genuinely random series (economic
/ financial data, near a martingale) has no such `f` -- fitting a curve to the path
is meaningless. What it has is **second-order structure**: an autocovariance, a
spectrum, the variance of block means across scales, a deterministic mean of trend
plus seasonal cycle, and the volatility of its increments. `SecondOrderImage` is
the additive sufficient statistic of exactly that structure -- one accumulator that
holds every field a gate or estimator reads, and adds over sample sets the way an
image in the batch fitting core does.

API: [api/stochastic.md](API-Stochastic). Validation against ground truth and the
classical estimators: [the stochastic-series domain report](Domain-Stochastic-Series).

---

## The image

`SecondOrderImage` accumulates, per chunk of samples: the first moments and time
cross-sums (the trend); lagged sums of the level, the increments, the squared
increments and the squared level, to a lag budget; squared block sums at the
dyadic scales; the residues of a fixed-grid DFT; and the first and last `lag + 1`
samples as head and tail carries. Every field is additive over sample sets, so two
images of consecutive stretches merge into the image of their union exactly.

The only cost of merging is at the join: the lagged sums of the merged image gain
the cross terms between the first image's tail carry and the second's head
carry -- the pairs of samples that straddle the boundary and would otherwise be
missed. Carrying `lag + 1` samples on each side is exactly enough to compute them,
so merging two images costs one FFT cross-correlation per lagged-sum field and
nothing that grows with either image's length.

---

## The functionals and their fits

For a second-order stationary process, the deterministic objects below have known
closed forms; the named dtfit fitter recovers the parameter from the functional the
image reads out.

| process feature | functional | its form | dtfit fit |
|---|---|---|---|
| **mean reversion** (OU / AR(1)) | autocovariance `rho(k)` (`image.acov()`) | `phi^k = exp(-k/tau)` | Yule-Walker `gamma_1/gamma_0`, or an exponential fit to the ACF ([`ar1_reversion`](API-Stochastic#estimators)) |
| **stochastic cycle** (AR(2), complex roots) | autocovariance `rho(k)` | `r^k cos(w k + p)` | damped cosine to the ACF, oscillatory recipe ([`cycle_period`](API-Stochastic#estimators)) |
| **volatility clustering** (GARCH(1,1)) | autocovariance of the squares (`image.acov_squares()` / `acov_volatility()`) | `~ (alpha+beta)^k` | exponential to that ACF ([`garch_persistence`](API-Stochastic#estimators)) |
| **long memory** (ARFIMA, `d = H - 1/2`) | Blackman-Tukey spectrum near 0 (`image.spectrum()`) | power law `S(f) ~ c f^{-2d}` | LSI slope of the log-spectrum, GPH ([`hurst_spectral`](API-Stochastic#estimators)) |
| **self-similarity** | aggregated-variance curve (`image.aggregated_variance()`) | power law `Var(block mean at m) ~ c m^{2H-2}` | power-law fit ([`hurst_aggvar`](API-Stochastic#estimators)) |
| **conditional mean** | trend + cycle (`image.trend()` / `image.seasonal()`) | structural curve | the image's own trend plus a Dirichlet-kernel seasonal read-out, leaving a stochastic residual ([`decompose_trend_cycle`](API-Stochastic#estimators)) |

The autocovariance, the block sums and the DFT residues are themselves additive
functionals, so the image accumulates them in `O(n log n)` by the Wiener-Khinchin
FFT rather than storing the record. So the work the fitters do here is the same
weighted-spectral ([LSI](Methods-LSI)) and area ([EAC](Methods-EAC)) matching used
everywhere else -- applied to the functional the image carries rather than to the
path.

---

## The merged solution: gated composition

[`fit_stochastic`](API-Stochastic#fit_stochastic) composes the routes in the order
the second-order theory dictates, each behind a significance gate, so the model
claims only the structure that is really there:

```
   y
   |
 (0) unit-root gate (ADF, ct, AIC lags, from the autocovariances)  --I(1)-->
   |                                          difference; report random walk [+ drift]
   | stationary / trend-stationary
 (1) deterministic mean: the image's LSI trend (|t|>trend_t, R^2 gate)
                         + multi-harmonic Fourier seasonal/cycle off the
                           fixed-grid DFT (spectral-peak gate)
   | residual
 (2) whiten with an AR(1) off the residual autocovariance
   |
 (3) long memory on the INNOVATIONS (spectral Hurst > lm_hurst)
 (4) mean reversion (AR(1) phi)   (5) volatility clustering (excess squared ACF)
```

Two disambiguations matter. Long memory is tested on the **whitened innovations**,
so a near-unit-root AR(1) (whose innovations are white) is not mislabelled as long
memory. Volatility clustering is tested on the **excess autocorrelation**
`rho_2 - rho^2` of the squared level (`image.acov_squares()`) over the residual's
own autocorrelation `rho` -- a Gaussian linear process with autocorrelation `rho`
has squared-series autocorrelation `rho^2` on its own, so subtracting it isolates
genuine ARCH-type structure. `acov_squares` reads the raw level, never detrended
or deseasonalized, so a persistent trend or seasonal component can itself read as
spurious volatility clustering rather than being screened out. The **unit-root
gate** is the load-bearing guard -- without it a random walk's wandering level
draws a spurious trend / cycle / long memory (the classic spurious regression). It
is the augmented Dickey-Fuller statistic of the constant+trend regression,
computed from the image's autocovariances in Toeplitz form (AIC lag selection),
agreeing with `statsmodels.adfuller`'s verdict in 139 of 140 measured series, with
a strict cyclical exemption so a genuine interior spectral peak (a real cycle,
near the unit circle but at `f > 0`) is kept for the stationary branch instead of
being differenced.

---

## Forecasting: backtest model selection

Rather than trusting one structural forecast, `fit_stochastic` rolling-origin
backtests a **regime-informed candidate set** -- random walk, drift, mean reversion,
the image's least-squares trend, and two multi-harmonic seasonal continuations (an
unbiased fitted extrapolation and an anchored one) -- and keeps the RMSE-optimal
one, defaulting to the random walk when nothing beats it. So it beats persistence
wherever some model genuinely can (a drift for GDP, mean reversion for a rate, a
seasonal cycle for CO2) and ties it on a near-martingale, never losing badly. The
seasonal forecast extrapolates the *fitted* trend + seasonal (not the noisy last
value, which would carry that residual forward as a bias) for a noisy series, while
a clean strong-trend series keeps the anchored variant -- the backtest decides which.
Each fold refits its own image of the training stretch, so a fit from a
`SecondOrderImage` with no underlying series cannot be backtest-selected: the chosen
name carries `" (no backtest)"`.

The confidence band is keyed off the **selected** forecaster rather than the
detected flags, so its growth matches the point forecast: **bounded** for mean
reversion, `~sqrt(h)` for a random walk, and the long-memory **`h^(2H)`** band
(`2H > 1`) when the long-memory forecaster is chosen -- a wider envelope than the
plain `sigma^2 * h` random-walk band, which would under-cover a long-memory path.

---

## The generative model

[`StochasticModel.simulate`](API-Stochastic#model) makes the characterization a
*generator*: it composes the detected deterministic mean (trend + multi-harmonic
seasonal) with a stochastic residual drawn to match the detected regime -- a
stationary AR(1) for mean reversion, **ARFIMA(0, d, 0)** with `d = H - 1/2` for long
memory, a GARCH(1,1) path for volatility clustering, an integrated walk (with drift)
for a unit root, white noise otherwise. The honest test is the round-trip: fit a
series, simulate from the fitted model, re-fit the simulation -- a faithful generator
recovers its own regime.

---

## The streaming twin

[`StochasticFilter`](API-Stochastic#filter) is the per-input online version of the
second-order stage, built the way dtfit's other streaming filters are -- incremental,
no batch re-fit. It maintains **EWMA autocovariances** of the level and of
`|level - mean|` in `O(K)` per sample (the running ACF), then reads the parameters
in closed form using dtfit's own principles in streaming form:

- **persistence** (AR(1) `phi`) and **volatility persistence** by the **EAC
  equal-areas criterion** -- for an exp-decaying ACF the ratio of two consecutive
  equal-width area windows is `exp(-g h)`, which pins the decay rate `g` (hence the
  persistence `exp(-g)`) amplitude-free; the streaming form of
  `fit_eac("exp(-g*k)")`. Only lags above the white-noise band `~2/sqrt(n_eff)` count
  as signal, so a fast-decay ACF's noisy tail does not trigger the integration;
- the **cycle** from the AR(2) characteristic roots of the running autocovariances.

Its batch counterpart is `SecondOrderStream`'s block form: block images resolve a
change at block granularity and merge exactly, where the filter's exponentially
weighted statistics resolve it within about a half-life instead. Measured on a
tracked AR(1) coefficient, block images and the filter both reach RMSE 0.033.

A two-timescale **fused statistic** (a fast/slow EWMA of the persistence and log
volatility, normalized by a frozen in-control gap variance) flags a structural break
once per change, at a low false-alarm rate -- the streaming counterpart of the
[`FusedChiSquareDetector`](API-Streaming#fused). Memory and per-sample cost are flat
(independent of the stream length), matching the characteristics of
[`EACFilter` / `LSIFilter`](Methods-Equal-Areas-Filter).

One API note: `StochasticFilter.partial_fit(xs)` ingests a **batch** of samples (a
house-style alias for a loop over `update`), unlike the single-sample
`partial_fit(t, y)` of [`EACFilter` / `LSIFilter`](Methods-Equal-Areas-Filter). Use
`update(x)` for the true one-at-a-time path.
