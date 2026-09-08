# Domain -- Stochastic series (dtfit on random data)

The validation behind the promoted [stochastic-series API](API-Stochastic) and
[methods](Methods-Stochastic). Full runnable report: the
[`stochastic_series` notebook](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/domains/stochastic_series/stochastic_series.ipynb)
in `dtfit-experimental`.

## Intent

Can dtfit -- a *deterministic* curve fitter -- be put to work on genuinely random
series (economic / financial data)? A martingale path has no `y = f(t)` to fit. But
a stochastic process has **deterministic functionals** -- its autocovariance,
spectrum, aggregated variance and trend/cycle -- whose forms (damped
exponentials/cosines, power laws) are exactly the shapes dtfit excels at. So the
domain fits the **functional**, not the **path**, on processes with *known*
parameters, merges the routes into one solution, and tests it on real economic data
against the established toolkit -- reported honestly.

## Methods under test (dtfit)

- **estimators** -- `hurst_spectral` / `hurst_aggvar` (long memory), `ar1_reversion`
  (mean reversion), `garch_persistence` (volatility), `cycle_period` (stochastic
  cycle), `decompose_trend_cycle`; each feeds a functional to `fit_lsi` / `fit_eac`.
- **`fit_stochastic`** -- the merged solution: a gated, ordered pipeline (an ADF
  unit-root gate off the image's autocovariances -> deterministic mean -> whiten ->
  long memory on the innovations -> mean reversion -> volatility) returning a
  `StochasticModel` with the detected regime, a backtest-selected forecast, and a
  generator.
- **`StochasticModel.simulate`** -- draws fresh realizations from the detected
  components (the model is a tunable generator, not just a summary).
- **`StochasticFilter`** -- the per-input streaming twin (EWMA autocovariances read
  by the EAC equal-areas criterion + a fused change-point detector).

## Baseline methods (established)

- **Hurst:** OLS log-log slope, rescaled-range (R/S) analysis, detrended fluctuation
  analysis (DFA).
- **Mean reversion:** the lag-1 autocorrelation; **cycle:** the FFT periodogram peak.
- **Forecasting:** random walk, drift, AR(1), ARIMA(2,1,2), Holt-Winters ETS, Theta,
  seasonal-naive.
- **Streaming reference:** dtfit's own `LSIFilter` (per-sample cost), the bar for the
  filter's flat-memory / bounded-speed characteristics.

## What it shows

| claim | result |
|---|---|
| **parameter recovery works** | E1-E6 all VIABLE at full sample size; the ACF-fit routes (mean reversion, cycle) beat their trivial baselines; the spectral Hurst is competitive with R/S and DFA |
| **regime identification works** | the router lands on the correct regime ~95% of the time; reports "no structure" on white noise (no hallucinated components) |
| **forecasting is honest** | beats the random walk where structure extrapolates -- **CO2 (trend+seasonal) ~0.15x, GDP (drift) ~0.41x, sunspots (cyclical) ~0.72x** -- and ties it on a near-martingale (FX level, T-bill rate); never loses badly (a rolling-origin holdout guard falls back to persistence) |
| **reproduces the literature** | Nelson-Plosser's random-walk-with-drift US GDP, the ~11-year sunspot cycle, Mauna Loa CO2 as trend+season, Hurst's Nile at `H ~ 0.9` (agreeing with R/S and DFA), near-unit-root interest rates, FX as a random walk with volatility clustering (long memory in `\|returns\|`) |
| **it generates, not just summarizes** | the fit -> simulate -> refit round-trip recovers the regime **100%** across every process type |
| **streaming works** | online phi tracking MAE ~0.03, structural-break detection 100% at ~78-step latency, ~0.7 false alarms / 3000 samples; flat memory and ~11 us/sample (faster than `LSIFilter`'s ~36 us) |

## The honest ceiling

None of this predicts the *innovation* -- the martingale component is unfittable by
any deterministic curve. What the solution delivers is a coherent characterization
of a series' *structured* part (memory, reversion, persistence, cycle, trend,
volatility) plus a forecast and a generator that are appropriately humble when there
is no structure to exploit. That, validated across six process families, a 7-series
real gallery, the generative round-trip and the streaming filter, is what cleared
the bar to [promote the solution into stable `dtfit`](API-Stochastic).
