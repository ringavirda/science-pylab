"""The stochastic-series model in the dtfit ``Model`` convention.

The deterministic catalog families fit a ``y = f(x)`` curve.
:class:`Stochastic` instead characterizes a random series across the
stochastic routes (trend, cycle, long memory, mean reversion, volatility)
behind significance gates, and is driven the same way, through ``.fit()``::

    from dtfit.models import Stochastic
    m = Stochastic().fit(series)          # -> a fitted StochasticModel
    print(m.regime, m.summary())
    m.forecast(12)                        # backtest-selected forecast
    m.simulate(200)                       # a fresh realization

A thin parameterised wrapper over :func:`dtfit.stochastic.fit_stochastic`. Its
companion is the streaming :class:`dtfit.stochastic.StochasticFilter`.
"""

from __future__ import annotations

import numpy as np

from dtfit.stochastic import fit_stochastic, StochasticModel

__all__ = ["Stochastic"]


class Stochastic:
    """Stochastic-series model with the catalog ``.fit()`` ergonomics.

    Args:
        period: A seasonal period to use; detected from the spectrum if None.
        max_harmonics: Cap on the Fourier harmonics of the seasonal component.
        forecaster: Forecast control. ``"auto"`` backtest-selects; a name from
            :data:`dtfit.stochastic.FORECASTERS`, a callable
            ``(train, h) -> arr``, or a list of candidates also works.
        lag: Lag budget of the second-order image every gate reads, in
            samples.
        nfreq: Smallest frequency grid of that image; it is raised to
            resolve the record.
        **gates: Detection-gate overrides forwarded to
            :func:`~dtfit.stochastic.fit_stochastic` (``trend_t``,
            ``cycle_strength``, ``min_cycles``, ``lm_hurst``, ``mr_phi``,
            ``vol_persist``).
    """

    name = "stochastic"
    category = "stochastic"

    def __init__(self, *, period: float | None = None, max_harmonics: int = 4,
                 forecaster: object = "auto", lag: int = 256,
                 nfreq: int = 512, **gates: float) -> None:
        self.period = period
        self.max_harmonics = int(max_harmonics)
        self.forecaster = forecaster
        self.lag = int(lag)
        self.nfreq = int(nfreq)
        self.gates = gates
        self.model_: StochasticModel | None = None

    def __repr__(self) -> str:
        return (f"Stochastic(period={self.period!r}, "
                f"forecaster={self.forecaster!r})")

    def fit(
        self, x: np.ndarray, y: np.ndarray | None = None
    ) -> StochasticModel:
        """Fit to a series and return the fitted :class:`StochasticModel`.

        Call as ``fit(series)`` for uniform unit time, or ``fit(t, series)``
        with an explicit time index, mirroring the catalog's ``fit(x, y)``.
        The result is also stored on :attr:`model_`.
        """
        # Pass the inputs through unchanged: ``fit_stochastic`` needs to see a
        # pandas index to give back an indexed forecast, and coerces to float
        # arrays itself.
        if y is None:
            series, t = x, None
        else:
            series, t = y, x
        self.model_ = fit_stochastic(
            series, t, period=self.period, max_harmonics=self.max_harmonics,
            forecaster=self.forecaster, lag=self.lag, nfreq=self.nfreq,
            **self.gates)
        return self.model_
