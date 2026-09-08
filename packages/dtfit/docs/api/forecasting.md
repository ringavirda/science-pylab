# Forecasting & high-level estimation

The "just fit it" entry points. `auto_estimate` recovers physical parameters by
routing on signal *shape*; `auto_forecast` is a structured fit-then-extrapolate
forecaster with no-structure and divergence guards. For genuinely *random* series
(asset returns, rates, river levels) fit the deterministic functionals of the
process with `fit_stochastic` / `StochasticModel` instead of a deterministic
curve -- every gate and estimator reads them off one additive second-order
image of the record, `SecondOrderImage`.

::: dtfit.auto_estimate

::: dtfit.auto_forecast

::: dtfit.ForecastResult

::: dtfit.fit_stochastic

::: dtfit.StochasticModel

::: dtfit.stochastic.SecondOrderImage

::: dtfit.stochastic.SecondOrderStream
