"""The classical twin of dtfit.stochastic, and the head-to-head against it.

What these pin is the comparison harness, not dtfit. The classical-estimator
model has to recover the same parameters and route the same regimes; only then
is a dtfit advantage the domain experiment reports measured against a fair
textbook baseline.
"""

import numpy as np
import pytest

from dtfit_experimental.experiments.common.classical_stochastic import (
    ols_ar1,
    garch_mle_persistence,
    classical_decompose,
    fit_classical_stochastic,
)
from dtfit_experimental.experiments.domains.stochastic_series.backend import (
    gen_ar1, gen_arfima, gen_garch, gen_ar2_cycle, gen_trend_cycle,
    exp_model_comparison, exp_garch, exp_decompose, exp_merged_router,
)
from dtfit_experimental.experiments.common.metrics import metrics
from dtfit_experimental.experiments.common.baselines import random_walk_forecast


# the individual classical estimators
@pytest.mark.parametrize("phi", [0.5, 0.8, 0.95])
def test_ols_ar1_recovers_phi(phi):
    x = gen_ar1(2000, phi, np.random.default_rng(0))
    assert ols_ar1(x) == pytest.approx(phi, abs=0.05)


def test_garch_mle_recovers_persistence():
    r = gen_garch(4000, 0.05, 0.08, 0.90, np.random.default_rng(0))
    # the generator's alpha + beta is the true persistence, 0.08 + 0.90
    assert garch_mle_persistence(r) == pytest.approx(0.98, abs=0.05)


def test_classical_decompose_recovers_trend_and_period():
    t, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(0))
    dec = classical_decompose(t, y)
    assert dec["slope"] == pytest.approx(0.02, rel=0.15)
    assert dec["period"] == pytest.approx(50.0, rel=0.1)


# the full model routes the same regimes
@pytest.mark.parametrize("gen,expect", [
    (lambda: gen_ar1(1500, 0.7, np.random.default_rng(0)), "mean-rever"),
    (lambda: gen_arfima(4096, 0.35, np.random.default_rng(0)), "long-memory"),
    (lambda: gen_ar2_cycle(1200, 16.0, 0.97, np.random.default_rng(0)), "cyclical"),
    (lambda: np.cumsum(np.random.default_rng(0).standard_normal(1500)), "random walk"),
    (lambda: np.random.default_rng(0).standard_normal(1500), "white noise"),
])
def test_classical_model_routes_regime(gen, expect):
    m = fit_classical_stochastic(gen())
    assert expect in m.regime


def test_classical_forecast_beats_random_walk_on_structure():
    t, y = gen_trend_cycle(440, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(1))
    h = 40
    train, test = y[:-h], y[-h:]
    m = fit_classical_stochastic(train)
    pred = m.forecast(h)
    assert pred.shape == (h,)
    cl = metrics(test, pred)["RMSE"]
    rw = metrics(test, random_walk_forecast(train, h))["RMSE"]
    assert cl < rw  # on a trend-plus-cycle series, structure beats persistence


# the head-to-head wiring
def test_model_comparison_runs_and_is_finite():
    cmp = exp_model_comparison(seeds=3)
    assert cmp["rows"] and cmp["n_cases"] == len(cmp["rows"])
    for row in cmp["rows"]:
        assert np.isfinite(row["dtfit/RW"]) and np.isfinite(row["classical/RW"])
        assert row["winner"] in ("dtfit", "classical")


def test_filled_baselines_are_present():
    # every head-to-head reports a classical foil: a baseline error from E4
    # (GARCH) and E6 (decompose), a router accuracy from E7.
    e4 = exp_garch(seeds=2, n=2500)
    e6 = exp_decompose(seeds=2)
    e7 = exp_merged_router(seeds=2)
    assert np.isfinite(e4["base_err"])
    assert np.isfinite(e6["base_err"])
    assert np.isfinite(e7["classical_accuracy"])
