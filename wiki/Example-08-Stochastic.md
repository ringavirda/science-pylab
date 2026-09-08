# Example 08 Stochastic

Stochastic-series characterization, forecasting and generation.

For genuinely *random* data (economic / financial series) dtfit does not fit the
path directly -- it fits the deterministic **functionals** of the process (its
autocovariance, spectrum, trend / cycle) with the same integral fitters, recovers
the second-order regime, and then forecasts, bands and even *generates* fresh
paths of it. fit_stochastic is the one-call entry; StochasticModel is the fitted
result; StochasticFilter tracks the structure online; Stochastic is the catalog
model-wrapper.

Run headless:   python examples/08_stochastic.py

Source: [`packages/dtfit/examples/08_stochastic.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/08_stochastic.py)

```python
import numpy as np

from dtfit import fit_stochastic, StochasticFilter, Stochastic


def characterize_and_forecast(rng) -> None:
    # A mean-reverting AR(1): x_t = phi x_{t-1} + eps. fit_stochastic detects the
    # regime, forecasts with a regime-appropriate band, and can regenerate it.
    n, phi = 1200, 0.6
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1.0)
    model = fit_stochastic(x)
    print("== fit_stochastic: an AR(1) mean-reverting series ==")
    print("detected regime :", model.regime)
    point, lo, hi = model.forecast(10, return_conf_int=True)
    print("10-step forecast:", np.round(point[:3], 3), "...  (band widens with horizon)")
    print("band at h=10    : [{:.2f}, {:.2f}]".format(lo[-1], hi[-1]))
    sim = model.simulate(n, seed=0)
    print("simulate() round-trips:", fit_stochastic(sim).regime)


def detect_trend_and_cycle(rng) -> None:
    # A deterministic trend + cycle buried in noise: the fitter reports the
    # structural components it recovered (fingerprint) alongside the regime.
    t = np.arange(600)
    y = 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 1.0, t.size)
    model = fit_stochastic(y)
    print("\n== fit_stochastic: trend + cycle ==")
    print("regime     :", model.regime)
    print(model.summary())


def online_tracking(rng) -> None:
    # StochasticFilter tracks the second-order structure per sample (O(nlags)),
    # flagging a change-point when the process regime shifts mid-stream.
    a = np.zeros(600)
    for t in range(1, 600):
        phi = 0.4 if t < 300 else 0.95            # persistence jumps at t=300
        a[t] = phi * a[t - 1] + rng.normal(0, 1.0)
    flt = StochasticFilter(warmup=80).partial_fit(a)
    snap = flt.snapshot()
    print("\n== StochasticFilter: online second-order tracking ==")
    print("final ar1_phi  :", round(snap["ar1_phi"], 3), " (tracked up from 0.4 as persistence rose)")
    print("regime label   :", snap["regime"])


def model_wrapper(rng) -> None:
    # The catalog-style wrapper: Stochastic().fit(series) in the same .fit()
    # convention as the deterministic Model families -- it returns a StochasticModel.
    n = 1200
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1.0)
    model = Stochastic().fit(x)
    print("\n== dtfit.Stochastic model wrapper ==")
    print("regime:", model.regime, " | forecaster:", model.forecaster_name)


def main() -> None:
    rng = np.random.default_rng(0)
    characterize_and_forecast(rng)
    detect_trend_and_cycle(rng)
    online_tracking(rng)
    model_wrapper(rng)


if __name__ == "__main__":
    main()
```

## Output (`python examples/08_stochastic.py`)

```text
== fit_stochastic: an AR(1) mean-reverting series ==
detected regime : mean-reverting
10-step forecast: [0.326 0.171 0.075] ...  (band widens with horizon)
band at h=10    : [-2.50, 2.36]
simulate() round-trips: mean-reverting

== fit_stochastic: trend + cycle ==
regime     : trend+seasonal
StochasticModel  regime='trend+seasonal'  n=600
  components: trend, seasonal, long-memory, vol-clustering
  trend       slope = 0.01887
  seasonal    period = 49.99 (1 harmonic(s))
  long memory H = 0.754 (d = 0.254)
  volatility  persistence = 0.986
  innovation sigma = 1.035
  forecaster: trend+seasonal

== StochasticFilter: online second-order tracking ==
final ar1_phi  : 0.863  (tracked up from 0.4 as persistence rose)
regime label   : mean-reverting

== dtfit.Stochastic model wrapper ==
regime: mean-reverting  | forecaster: mean-reversion
```
