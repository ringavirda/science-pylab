# Forecasting & high-level estimation

`auto_forecast` is a structured fit-then-extrapolate forecaster with
no-structure and divergence guards. Recovering parameters needs no router:
`fit(model, data, basis="auto")` picks the basis by outcome. For genuinely
*random* series (asset returns, rates, river levels) fit the deterministic
functionals of the process with `fit_stochastic` / `StochasticModel` instead
of a deterministic curve -- every gate and estimator reads them off one
additive second-order image of the record, `SecondOrderImage`.

::: dtfit.auto_forecast

::: dtfit.ForecastResult

::: dtfit.stochastic.fit_stochastic

::: dtfit.stochastic.StochasticModel

::: dtfit.stochastic.SecondOrderImage

::: dtfit.stochastic.SecondOrderStream
