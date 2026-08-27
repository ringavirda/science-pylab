"""Stochastic-series domain validation.

Asks whether dtfit's deterministic fitters can be put to work on a random
series (economic or financial data), by fitting a stochastic process's
deterministic functionals (its autocovariance, spectrum, aggregated variance
and trend/cycle) and recovering the process's parameters from them.

``backend.py`` holds the ground-truth data generators (ARFIMA, AR(1)/OU,
GARCH(1,1), AR(2) pseudo-cycle, trend+cycle) and the evaluation harness
``run()`` / ``summary()``, which scores each possibility VIABLE, MARGINAL or
NOT VIABLE by its parameter-recovery error against the known truth. The
estimators it exercises live in :mod:`dtfit.stochastic` (``fit_stochastic``,
``StochasticModel``, ``StochasticFilter``, ``dtfit.Stochastic``).
"""
