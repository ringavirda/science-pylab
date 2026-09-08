"""Stochastic-series adaptations: parameter recovery on random processes.

Each test simulates a process whose parameter is known and asserts that the
dtfit-based estimator, which fits a deterministic functional of the process,
gets the truth back within a tolerance. This is the CI-checkable core of the
``stochastic_series`` domain experiment; the full VIABLE / MARGINAL / NOT
VIABLE verdict table lives there.
"""

import numpy as np
import pytest

from dtfit.stochastic import (
    sample_acf,
    hurst_aggvar,
    hurst_spectral,
    ar1_reversion,
    garch_persistence,
    cycle_period,
    decompose_trend_cycle,
    fit_stochastic,
    StochasticModel,
)
from dtfit_experimental.experiments.domains.stochastic_series import backend as B


def _mean(fn, seeds):
    return float(np.mean([fn(s) for s in seeds]))


# the shared functional
def test_sample_acf_white_noise_is_a_spike():
    x = np.random.default_rng(0).standard_normal(4000)
    acf = sample_acf(x, 12)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.12)  # within the white-noise band


# E1/E2 long memory (Hurst)
def test_hurst_spectral_recovers_long_memory():
    H = 0.8  # = d + 1/2 for the ARFIMA d = 0.3 generated below
    err = _mean(lambda s: abs(
        hurst_spectral(B.gen_arfima(4096, 0.3, np.random.default_rng(10 + s)))["H"] - H
    ), range(3))
    assert err < 0.15


def test_hurst_aggvar_detects_long_memory_direction():
    # Aggregated variance is only MARGINAL on the exact value. The bar here is
    # direction: well above the H = 0.5 of white noise, nothing tighter.
    Hs = [hurst_aggvar(B.gen_arfima(4096, 0.4, np.random.default_rng(20 + s)))["H"]
          for s in range(3)]
    assert np.mean(Hs) > 0.6


# E3 mean reversion (AR(1)/OU)
@pytest.mark.parametrize("phi", [0.6, 0.9])
def test_ar1_reversion_recovers_phi(phi):
    est = _mean(lambda s: ar1_reversion(
        B.gen_ar1(1500, phi, np.random.default_rng(30 + s)))["phi"], range(3))
    assert abs(est - phi) / phi < 0.12


# E4 volatility persistence (GARCH(1,1))
def test_garch_persistence_recovers_alpha_plus_beta():
    truth = 0.08 + 0.90
    est = _mean(lambda s: garch_persistence(
        B.gen_garch(4000, 0.05, 0.08, 0.90,
                    np.random.default_rng(40 + s)))["persistence"], range(3))
    assert abs(est - truth) / truth < 0.20


# E5 stochastic cycle (AR(2), complex roots)
def test_cycle_period_recovers_period():
    P = 16.0
    est = _mean(lambda s: cycle_period(
        B.gen_ar2_cycle(1500, P, 0.97, np.random.default_rng(50 + s)))["period"],
        range(3))
    assert abs(est - P) / P < 0.15


# E6 trend + cycle decomposition
def test_decompose_recovers_trend_and_cycle():
    t, y = B.gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(0))
    dec = decompose_trend_cycle(t, y)
    assert abs(dec["period"] - 50.0) / 50.0 < 0.10
    assert abs(dec["slope"] - 0.02) / 0.02 < 0.30
    fc = dec["forecast"](20)
    assert fc.shape == (20,) and np.all(np.isfinite(fc))


# the domain harness
def test_domain_experiments_run_and_are_viable():
    rows = [
        B.exp_ar1(2, n=1000),
        B.exp_cycle(2, n=1000),
        B.exp_garch(2, n=2500),
    ]
    for r in rows:
        assert r["verdict"] in {"VIABLE", "MARGINAL", "NOT VIABLE"}
        assert r["verdict"] != "NOT VIABLE"
    assert "verdict" in B.summary(rows)


# the merged solution: fit_stochastic
def test_merged_white_noise_reports_no_structure():
    m = fit_stochastic(np.random.default_rng(0).standard_normal(1500))
    assert isinstance(m, StochasticModel)
    assert m.regime.startswith("white noise")
    assert m.components == ("none",)
    assert "regime" in m.fingerprint() and "sigma" in m.summary()


def test_merged_random_walk_detected_as_unit_root():
    y = np.cumsum(np.random.default_rng(1).standard_normal(1500))
    m = fit_stochastic(y)
    assert m.regime.startswith("random walk")
    assert not m.has_trend and not m.has_long_memory


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_merged_ar1_is_mean_reverting(phi):
    m = fit_stochastic(B.gen_ar1(1500, phi, np.random.default_rng(2)))
    assert m.has_mean_reversion and not m.has_long_memory
    assert abs(m.ar1_phi - phi) / phi < 0.2


def test_merged_arfima_is_long_memory():
    m = fit_stochastic(B.gen_arfima(4096, 0.3, np.random.default_rng(3)))
    assert m.has_long_memory and m.hurst > 0.65


def test_merged_garch_flags_volatility_clustering():
    m = fit_stochastic(B.gen_garch(4000, 0.05, 0.08, 0.90, np.random.default_rng(4)))
    assert m.has_vol_clustering and not m.has_mean_reversion


def test_merged_trend_cycle_detected_and_forecasts():
    t, y = B.gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    m = fit_stochastic(y)
    assert m.has_trend and m.has_cycle
    pt, lo, hi = m.forecast(30, return_conf_int=True)
    assert pt.shape == (30,) and np.all(hi >= lo) and np.all(np.isfinite(pt))


def test_merged_router_accuracy_is_high():
    r = B.exp_merged_router(seeds=3)
    assert r["accuracy"] >= 80.0
    assert r["verdict"] in {"VIABLE", "MARGINAL"}


def test_seasonal_multiharmonic_detected_and_beats_rw():
    # a sawtooth (non-sinusoidal) season of period 24, on a trend, plus noise
    t = np.arange(600.0)
    y = (0.01 * t + 2.0 * ((t % 24) / 24.0 - 0.5)
         + np.random.default_rng(3).normal(0, 0.15, 600))
    m = fit_stochastic(y)
    assert m.seasonal and m.has_cycle
    assert abs(m.cycle_period - 24.0) < 3.0
    assert m.n_harmonics >= 2          # one harmonic cannot shape a sawtooth
    h = 48
    tr, te = y[:-h], y[-h:]
    fc = fit_stochastic(tr).forecast(h)
    rw = B.bl.random_walk_forecast(tr, h)
    assert np.sqrt(np.mean((fc - te) ** 2)) < np.sqrt(np.mean((rw - te) ** 2))


def test_user_specified_seasonal_period():
    t = np.arange(400.0)
    y = np.sin(2 * np.pi * t / 12.0) + np.random.default_rng(0).normal(0, 0.1, 400)
    m = fit_stochastic(y, period=12.0)
    assert m.has_cycle and m.seasonal
    assert abs(m.cycle_period - 12.0) < 1e-6


def test_forecaster_selection_can_be_controlled():
    from dtfit.stochastic import FORECASTERS
    y = B.gen_ar1(800, 0.6, np.random.default_rng(1))
    # "auto" lands on a known forecaster
    assert fit_stochastic(y).forecaster_name in set(FORECASTERS) | {"custom"}
    # a built-in name is forced through
    assert fit_stochastic(y, forecaster="drift").forecaster_name == "drift"
    # a candidate list is narrowed by backtest
    assert fit_stochastic(y, forecaster=["random walk", "mean-reversion"]) \
        .forecaster_name in {"random walk", "mean-reversion"}
    # a callable is taken as given
    mc = fit_stochastic(y, forecaster=lambda tr, h: np.full(h, tr.mean()))
    assert mc.forecaster_name == "custom" and mc.forecast(5).shape == (5,)


def test_forecaster_unknown_name_raises():
    y = B.gen_ar1(400, 0.5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        fit_stochastic(y, forecaster="nonsense")


def test_merged_forecast_beats_random_walk_on_structured_data():
    y = B.gen_trend_cycle(400, 0.03, 40.0, 3.0, 1.0, np.random.default_rng(6))[1]
    skill, _ = B.forecast_skill(y, 40)
    assert skill["dtfit merged"] < skill["random walk"]


# the unit-root gate: reads the image, no statsmodels in the import path
def test_unit_root_gate_verdicts_without_statsmodels():
    """The unit-root gate reads the image's autocovariances and classifies
    the four canonical regimes with no statsmodels anywhere in the import
    path."""
    from dtfit.stochastic import is_nonstationary
    rng = np.random.default_rng(0)
    assert is_nonstationary(np.cumsum(rng.standard_normal(400)))          # I(1)
    assert not is_nonstationary(0.05 * np.arange(400)
                                + rng.standard_normal(400) * 3.0)         # trend
    assert not is_nonstationary(B.gen_ar1(800, 0.6, rng))                 # AR(1)
    assert not is_nonstationary(rng.standard_normal(400))                 # white


def test_image_unit_root_verdicts_agree_with_statsmodels():
    """The routing statistic comes from the image's autocovariances rather
    than from a per-sample regression; what has to hold is the verdict, and
    it holds on eleven of the twelve records below."""
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    from dtfit.stochastic import SecondOrderImage, dickey_fuller

    def gen(kind, s):
        r = np.random.default_rng(100 + s)
        if kind == "rw":
            return np.cumsum(r.standard_normal(400))
        if kind == "trend":
            return 0.05 * np.arange(400) + r.standard_normal(400) * 3.0
        if kind == "ar1":
            return B.gen_ar1(400, 0.9, r)
        return 3.0 * np.sin(2 * np.pi * np.arange(400) / 20.0) \
            + 0.5 * r.standard_normal(400)

    agree = total = 0
    for kind in ("rw", "trend", "ar1", "cycle"):
        for s in range(3):
            x = gen(kind, s)
            n = x.size
            maxlag = int(min(12 * (n / 100.0) ** 0.25, 12, n // 3))
            ref = sm.adfuller(x, regression="ct", maxlag=maxlag, autolag="AIC")
            img = SecondOrderImage.of(x, lag=48, nfreq=64)
            agree += (ref[1] < 0.05) == (dickey_fuller(img)["pvalue"] < 0.05)
            total += 1
    assert agree >= total - 1


# the generative half: StochasticModel.simulate()
def test_simulate_shape_finite_and_reproducible():
    y = B.gen_ar1(800, 0.6, np.random.default_rng(0))
    m = fit_stochastic(y)
    a = m.simulate(seed=7)
    b = m.simulate(seed=7)
    c = m.simulate(n=200, seed=7)
    assert a.shape == (m.n,) and np.all(np.isfinite(a))
    assert np.allclose(a, b)                 # same seed -> same path
    assert c.shape == (200,)


@pytest.mark.parametrize("name,gen,attr", [
    ("trend+cycle",
     lambda r: B.gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, r)[1], "has_cycle"),
    ("ar1", lambda r: B.gen_ar1(1500, 0.7, r), "has_mean_reversion"),
    ("garch",
     lambda r: B.gen_garch(4000, 0.05, 0.08, 0.90, r), "has_vol_clustering"),
])
def test_simulate_round_trip_recovers_regime(name, gen, attr):
    """fit -> simulate -> refit recovers the same structural flag: the model
    generates the process it characterizes, it does not merely summarize it."""
    hits = sum(
        bool(getattr(fit_stochastic(
            fit_stochastic(gen(np.random.default_rng(10 + s))).simulate(seed=s)),
            attr))
        for s in range(5)
    )
    assert hits >= 4   # one of the five seeds may sit on a boundary


def test_simulate_round_trip_preserves_no_structure_regimes():
    rng = np.random.default_rng(1)
    for gen, head in [
        (lambda: np.cumsum(rng.standard_normal(1500)), "random walk"),
        (lambda: rng.standard_normal(1500), "white noise"),
    ]:
        m = fit_stochastic(gen())
        m2 = fit_stochastic(m.simulate(seed=0))
        assert m2.regime.split()[0] == m.regime.split()[0]


def test_simulate_long_memory_path_has_elevated_hurst():
    """A simulated realization carries real long memory: Hurst above 0.62 on
    average, clear of the 0.5 of white noise, out of ARFIMA and not AR(1)."""
    from dtfit.stochastic import hurst_spectral
    # the same series test_merged_arfima_is_long_memory uses
    m = fit_stochastic(B.gen_arfima(4096, 0.3, np.random.default_rng(3)))
    assert m.has_long_memory
    hs = [hurst_spectral(m.simulate(seed=s))["H"] for s in range(4)]
    assert np.mean(hs) > 0.62


def test_simulate_trend_path_actually_trends():
    t, y = B.gen_trend_cycle(600, 0.05, 40.0, 2.0, 1.0, np.random.default_rng(2))
    m = fit_stochastic(y)
    sim = m.simulate(seed=0)
    # the fitted upward trend has to survive into the simulated path
    assert np.corrcoef(np.arange(sim.size), sim)[0, 1] > 0.5


# the streaming filter: online characterization and change detection
def test_filter_tracks_ar1_phi_online():
    """The filter's online AR(1) phi, the dtfit EAC equal-areas criterion read
    per input off a running ACF, converges on the truth. The 0.08 budget is set
    by the low-phi case: a phi = 0.3 ACF decays within a couple of lags and
    offers the least to integrate, while at phi = 0.85 the error runs several
    times smaller."""
    from dtfit import StochasticFilter
    for phi in (0.3, 0.6, 0.85):
        errs = []
        for s in range(3):
            f = StochasticFilter(halflife=300, warmup=100)
            f.partial_fit(B.gen_ar1(3000, phi, np.random.default_rng(s)))
            errs.append(abs(f.params_["ar1_phi"] - phi))
        assert np.mean(errs) < 0.08


def test_filter_detects_persistence_break():
    """A persistence jump, phi 0.2 -> 0.9 at the midpoint, is flagged inside
    the 400 samples following the break on at least four of the five seeds."""
    from dtfit import StochasticFilter
    hits = 0
    for s in range(5):
        r = np.random.default_rng(10 + s)
        a = B.gen_ar1(1500, 0.2, r)
        b = B.gen_ar1(1500, 0.9, r)
        b = b - b.mean() + a[-1]
        f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
        f.partial_fit(np.concatenate([a, b]))
        if any(1500 <= t <= 1900 for t in f.flag_times_):
            hits += 1
    assert hits >= 4


def test_filter_low_false_alarm_on_stationary_stream():
    """The false-alarm half of the break test runs at the same settings."""
    from dtfit import StochasticFilter
    counts = []
    for s in range(5):
        f = StochasticFilter(warmup=80, settle=500, z_thresh=4.0)
        f.partial_fit(B.gen_ar1(3000, 0.6, np.random.default_rng(30 + s)))
        counts.append(f.n_flags_)
    assert np.mean(counts) <= 1.5      # a stray flag is fine, a stream is not


def test_filter_experiment_is_viable():
    r = B.exp_online_filter(seeds=3)
    assert r["verdict"] in {"VIABLE", "MARGINAL"}
    assert r["break_hit_rate"] >= 75.0
    tr = B.filter_trace(B.gen_ar1(1000, 0.7, np.random.default_rng(0)))
    assert tr["phi"].shape == (1000,) and np.isfinite(tr["phi"][-1])


# real economic data (USD/UAH, bundled CSV)
# experiments/data is gitignored; the CSV is simply absent in CI.
needs_usd_uah = pytest.mark.skipif(
    not (B.EXPERIMENTS_DIR / "data" / "usd_uah_2014_2015.csv").exists(),
    reason="bundled USD/UAH CSV not available (experiments/data is gitignored)",
)


@needs_usd_uah
def test_real_usd_uah_level_is_random_walk_and_ties_rw():
    rd = B.exp_real_data()
    # the FX level is a near-random walk; the router has to call it one
    assert rd["level_regime"].startswith("random walk")
    # and tie the random-walk benchmark on the holdout, within 2 percent
    rmse = rd["forecast_rmse"]
    assert rmse["dtfit merged"] <= rmse["random walk"] * 1.02
    # the stylized fact: the long memory lives in the volatility, not the
    # level. The squared returns' excess autocorrelation on this record sits
    # just under the white-noise band, so the clustering shows in the Hurst of
    # the absolute returns rather than in the gate.
    assert rd["abs_returns_hurst"]["dtfit spectral"] > 0.55


# real data from statsmodels: reproducing the textbook results
HAS_SM = True
try:
    import statsmodels.api  # noqa: F401
except Exception:
    HAS_SM = False

sm_only = pytest.mark.skipif(not HAS_SM, reason="statsmodels not installed")


@sm_only
def test_real_gdp_is_random_walk_with_drift():
    # Nelson-Plosser (1982): US real GDP is a random walk with drift. The drift
    # is a stochastic trend, not the deterministic one has_trend flags.
    m = fit_stochastic(B._sm_series("gdp"))
    assert "random walk" in m.regime and m.has_trend is False


@sm_only
def test_real_sunspots_is_cyclical_near_11_years():
    # the canonical ~11-year solar cycle
    m = fit_stochastic(B._sm_series("sunspots"))
    assert "cyclical" in m.regime
    assert 8.0 <= m.cycle_period <= 14.0


@sm_only
def test_real_nile_has_long_memory():
    # Hurst's (1951) canonical long-memory series, H ~ 0.9 in the literature
    h = B.hurst_comparison(B._sm_series("nile"))
    assert h["dtfit spectral"] > 0.70
    assert h["R/S"] > 0.65   # the classic estimator has to agree


@sm_only
def test_real_suite_never_forecasts_much_worse_than_random_walk():
    # the merged solution beats the random walk where structure extrapolates,
    # ties it where there is none, and never loses by more than 5 percent
    for r in B.exp_real_suite()["rows"]:
        ratio = r["merged/RW"]
        if r["n"] > 0 and ratio == ratio:  # not NaN
            assert ratio <= 1.05
