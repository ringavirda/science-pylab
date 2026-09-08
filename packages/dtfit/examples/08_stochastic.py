"""Stochastic-series characterization, forecasting and generation.

For genuinely *random* data (economic / financial series) dtfit does not fit
the path directly -- it fits the deterministic **functionals** of the
process (its autocovariance, spectrum, trend / cycle) with the same
integral fitters, recovers the second-order regime, and then forecasts,
bands and even *generates* fresh paths of it. fit_stochastic is the
one-call entry; StochasticModel is the fitted result; StochasticFilter
tracks the structure online; Stochastic is the catalog model-wrapper.

Run headless:   python examples/08_stochastic.py
"""

import numpy as np

from dtfit.models import Stochastic
from dtfit.stochastic import StochasticFilter, fit_stochastic
from dtfit.stochastic import SecondOrderImage, SecondOrderStream


def characterize_and_forecast(rng) -> None:
    # A mean-reverting AR(1): x_t = phi x_{t-1} + eps. fit_stochastic detects
    # the regime, forecasts with a regime-appropriate band, and can
    # regenerate it.
    n, phi = 1200, 0.6
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, 1.0)
    model = fit_stochastic(x)
    print("== fit_stochastic: an AR(1) mean-reverting series ==")
    print("detected regime :", model.regime)
    point, lo, hi = model.forecast(10, return_conf_int=True)
    print("10-step forecast:", np.round(point[:3], 3),
          "...  (band widens with horizon)")
    print("band at h=10    : [{:.2f}, {:.2f}]".format(lo[-1], hi[-1]))
    sim = model.simulate(n, seed=0)
    print("simulate() round-trips:", fit_stochastic(sim).regime)


def detect_trend_and_cycle(rng) -> None:
    # A deterministic trend + cycle buried in noise: the fitter reports the
    # structural components it recovered (fingerprint) alongside the regime.
    t = np.arange(600)
    noise = rng.normal(0, 1.0, t.size)
    y = 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 50) + noise
    model = fit_stochastic(y)
    print("\n== fit_stochastic: trend + cycle ==")
    print("regime     :", model.regime)
    print(model.summary())


def image_and_block_stream(rng) -> None:
    # The tier's own image is the additive statistic every gate reads: build
    # it once over a whole record, or block by block off a stream and merge.
    # Both give the same numbers, so a record too large to hold can still be
    # characterized.
    n = 4000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal(0, 1.0)
    whole = SecondOrderImage.of(x)
    stream = SecondOrderStream(500, lag=whole.lag, nfreq=whole.nfreq,
                               scales=whole.scales)
    for start in range(0, n, 137):
        stream.update(x[start:start + 137])
    stream.close()
    merged = stream.assemble(0.0, float(n))
    print("\n== SecondOrderImage: one additive statistic ==")
    print("image fields    :", whole.acov().size, "lags,",
          whole.dft().size, "frequency bins,",
          whole.aggregated_variance()[0].size, "scales")
    print("blocked == whole:",
          bool(np.max(np.abs(whole.acov() - merged.acov())) < 1e-9))
    g = merged.acov()
    print("AR(1) phi from the image:", round(float(g[1] / g[0]), 3))
    # a bare image has no series to backtest a forecaster on, so name one
    print("regime from the image   :",
          fit_stochastic(merged, forecaster="mean-reversion").regime)


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
    print("final ar1_phi  :", round(snap["ar1_phi"], 3),
          " (tracked up from 0.4 as persistence rose)")
    print("regime label   :", snap["regime"])


def model_wrapper(rng) -> None:
    # The catalog-style wrapper: Stochastic().fit(series) in the same .fit()
    # convention as the deterministic Model families -- it returns a
    # StochasticModel.
    n = 1200
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1.0)
    model = Stochastic().fit(x)
    print("\n== dtfit.models.Stochastic wrapper ==")
    print("regime:", model.regime, " | forecaster:", model.forecaster_name)


def main() -> None:
    rng = np.random.default_rng(0)
    characterize_and_forecast(rng)
    detect_trend_and_cycle(rng)
    image_and_block_stream(rng)
    online_tracking(rng)
    model_wrapper(rng)


if __name__ == "__main__":
    main()
