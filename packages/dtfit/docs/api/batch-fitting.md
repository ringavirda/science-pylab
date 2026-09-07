# Batch fitting

The core batch fitters run on the **image** of the data -- a fixed-size basis
projection, additive over sample sets. `fit` runs on an `Original` or an
`Image` directly; `fit_lsi` and `fit_eac` are `fit`'s presets in the Legendre
and block bases.

See [the image](../guide/image.md) for the statistic every fitter runs on
and the [guide](../guide/choosing-a-method.md) for the decision tree.

::: dtfit.fit

::: dtfit.Original

::: dtfit.Image

::: dtfit.order_for

::: dtfit.image.coverage

::: dtfit.fit_lsi

::: dtfit.fit_eac

::: dtfit.ensemble_fit

::: dtfit.EnsembleResult

::: dtfit.FittingResult

::: dtfit.methods.resolve_model
