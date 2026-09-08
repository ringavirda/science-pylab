"""``fit_stochastic``: the gate sequence on the second-order image, the
forecasters that read it and the generator that round-trips through it."""

import numpy as np
import pytest

from dtfit import fit_stochastic, StochasticModel, Stochastic
from dtfit.stochastic import (
    FORECASTERS, SecondOrderImage, is_nonstationary,
)
from dtfit.stochastic.forecast import (
    make_seasonal_fc, make_seasonal_fc_anchored,
)
from stochastic.processes import (
    gen_ar1, gen_ar2, gen_ar2_cycle, gen_arfima, gen_garch, gen_trend_cycle,
)


def test_white_noise_reports_no_structure():
    m = fit_stochastic(np.random.default_rng(0).standard_normal(1500))
    assert isinstance(m, StochasticModel)
    assert m.regime.startswith("white noise") and m.components == ("none",)


def test_random_walk_is_detected_as_a_unit_root():
    m = fit_stochastic(
        np.cumsum(np.random.default_rng(1).standard_normal(1500)))
    assert m.regime.startswith("random walk") and not m.has_trend
    assert "unit-root" in m.components


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_ar1_is_mean_reverting(phi):
    m = fit_stochastic(gen_ar1(1500, phi, np.random.default_rng(2)))
    assert m.has_mean_reversion and abs(m.ar1_phi - phi) / phi < 0.2


def test_arfima_is_long_memory():
    m = fit_stochastic(gen_arfima(4096, 0.3, np.random.default_rng(3)))
    assert m.has_long_memory and m.hurst > 0.65


def test_strong_long_memory_still_detected():
    got = sum(fit_stochastic(gen_arfima(4096, 0.4, np.random.default_rng(s)))
              .has_long_memory for s in range(5))
    assert got >= 4


def test_ar_p_is_not_mislabeled_long_memory():
    for series in (gen_ar2(3000, 0.5, 0.3, 0),
                   gen_ar1(3000, 0.95, np.random.default_rng(0))):
        m = fit_stochastic(series)
        assert not m.has_long_memory
        assert m.has_mean_reversion


def test_garch_is_flagged_as_volatility_clustering():
    got = sum(fit_stochastic(gen_garch(
        4000, 0.05, 0.08, 0.90,
        np.random.default_rng(700 + s))).has_vol_clustering
        for s in range(5))
    assert got >= 4


def test_trend_cycle_is_detected_and_forecasts():
    _, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    m = fit_stochastic(y)
    assert m.has_trend and m.has_cycle
    assert m.cycle_period == pytest.approx(50.0, rel=0.02)
    pt, lo, hi = m.forecast(30, return_conf_int=True)
    assert pt.shape == (30,) and np.all(hi >= lo) and np.all(np.isfinite(pt))
    assert "trend" in m.summary() and "regime" in m.fingerprint()


def test_regime_identification_over_the_process_families():
    """The gate: the image pipeline labels the process families spec 13.4
    names, at its 20 seeds. The seven families the per-sample pipeline also
    covers must hit at least 132 of 140; the off-grid cycle, resolved only
    through the leakage-corrected amplitude rather than a raw grid bin, must
    hit all 20."""
    cases = [
        ("white noise",
         lambda s: np.random.default_rng(s).standard_normal(1500),
         lambda m: m.regime.startswith("white noise")),
        ("random walk",
         lambda s: np.cumsum(np.random.default_rng(s).standard_normal(1500)),
         lambda m: m.regime.startswith("random walk")),
        ("AR(1)", lambda s: gen_ar1(1500, 0.7, np.random.default_rng(s)),
         lambda m: "mean-revert" in m.regime),
        ("ARFIMA", lambda s: gen_arfima(4096, 0.3, np.random.default_rng(s)),
         lambda m: "long-memory" in m.regime),
        ("GARCH", lambda s: gen_garch(4000, 0.05, 0.08, 0.90,
                                      np.random.default_rng(s)),
         lambda m: m.has_vol_clustering),
        ("AR(2) cycle", lambda s: gen_ar2_cycle(1500, 16.0, 0.97,
                                                np.random.default_rng(s)),
         lambda m: any(w in m.regime
                       for w in ("cyclical", "cycle", "seasonal"))),
        ("trend + cycle",
         lambda s: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0,
                                   np.random.default_rng(s))[1],
         lambda m: "trend+" in m.regime),
    ]
    hits = total = 0
    per = []
    for name, gen, ok in cases:
        h = sum(bool(ok(fit_stochastic(gen(700 + s)))) for s in range(20))
        per.append(f"{name}: {h}/20")
        hits += h
        total += 20
    assert hits >= total - 8, "; ".join(per)

    # period 50 at n = 4000 falls between the bins of a fixed 512-bin
    # grid, where the leakage-corrected amplitude is what finds it
    off_grid = sum(
        bool("trend+" in fit_stochastic(
            gen_trend_cycle(4000, 0.02, 50.0, 3.0, 1.0,
                            np.random.default_rng(700 + s))[1]).regime)
        for s in range(20)
    )
    assert off_grid == 20, f"off-grid cycle: {off_grid}/20"


def test_a_block_with_a_nonzero_origin_forecasts_the_record_phase():
    """The seasonal coefficients and every forecaster that continues them are
    referenced to the global sample index, so a block starting at index 1000
    continues the cycle in the phase the whole record is in."""
    t = np.arange(2000.0)
    y = 3.0 * np.sin(2 * np.pi * t / 37.0 + 0.4)
    blk = SecondOrderImage(256, 1024, 7, t0=1000).update(y[1000:])
    whole = SecondOrderImage.of(y)
    truth = 3.0 * np.sin(2 * np.pi * np.arange(2000.0, 2005.0) / 37.0 + 0.4)
    for make in (make_seasonal_fc, make_seasonal_fc_anchored):
        fc = make(37.0, 1, False)(blk, 5)
        assert np.max(np.abs(fc - truth)) < 0.05
        assert np.max(np.abs(fc - make(37.0, 1, False)(whole, 5))) < 0.05
    with pytest.warns(UserWarning, match="cannot backtest-select"):
        m = fit_stochastic(blk)
    assert np.max(np.abs(np.asarray(m.forecast(5)) - truth)) < 0.1


def test_time_axis_units_do_not_change_seasonality_or_forecast():
    t, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    m1 = fit_stochastic(y, t)
    for m, dt in ((fit_stochastic(y, 0.5 * t), 0.5),
                  (fit_stochastic(y, 1900.0 + t / 12.0), 1.0 / 12.0)):
        assert m.has_cycle and m.seasonal
        assert m.cycle_period == pytest.approx(m1.cycle_period)
        assert m.n_harmonics == m1.n_harmonics
        assert m.cycle_amp == pytest.approx(m1.cycle_amp, rel=1e-6)
        assert m.trend_slope == pytest.approx(m1.trend_slope / dt, rel=1e-6)
        assert np.allclose(m.forecast(40), m1.forecast(40))


def test_period_kwarg_is_in_samples_on_any_time_axis():
    t = np.arange(400, dtype=float)
    y = (3.0 * np.sin(2.0 * np.pi * t / 40.0)
         + 0.5 * np.random.default_rng(8).standard_normal(400))
    m = fit_stochastic(y, 0.5 * t, period=40)
    assert m.has_cycle and m.seasonal and m.cycle_period == 40.0
    assert abs(m.cycle_amp - 3.0) < 0.5


def test_short_series_fallback_warns_and_is_visible_in_the_name():
    k = np.arange(40, dtype=float)
    y = (3.0 * np.sin(2.0 * np.pi * k / 8.0)
         + 0.1 * np.random.default_rng(3).standard_normal(40))
    with pytest.warns(UserWarning, match=r"too short to backtest-select"):
        m = fit_stochastic(y)
    assert m.forecaster_name == "random walk (short-series fallback)"
    pt, lo, hi = m.forecast(5, return_conf_int=True)
    assert pt.shape == (5,) and np.all(hi >= lo)
    assert np.all(np.diff(hi - lo) > 0)
    # a forced name looks up the forecaster directly, bypassing the
    # unit-root gate and the backtest, so the result does not depend on
    # which regime the gate calls this short white-noise draw
    m2 = fit_stochastic(np.random.default_rng(0).standard_normal(40),
                         forecaster="random walk")
    assert m2.forecaster_name == "random walk"


def test_forecaster_control():
    y = gen_ar1(800, 0.6, np.random.default_rng(1))
    assert fit_stochastic(y).forecaster_name in set(FORECASTERS) | {"custom"}
    assert fit_stochastic(y, forecaster="drift").forecaster_name == "drift"
    assert fit_stochastic(y, forecaster=["random walk", "mean-reversion"]) \
        .forecaster_name in {"random walk", "mean-reversion"}
    mc = fit_stochastic(y, forecaster=lambda tr, h: np.full(h, tr.mean()))
    assert mc.forecaster_name == "custom" and mc.forecast(5).shape == (5,)
    with pytest.raises(ValueError):
        fit_stochastic(y, forecaster="nonsense")


def test_failing_forecast_candidate_warns_and_loses_selection():
    y = gen_ar1(800, 0.6, np.random.default_rng(1))

    def bad(train, h):
        raise RuntimeError("candidate exploded")

    with pytest.warns(UserWarning,
                      match=r"candidate 'bad' failed during backtest"):
        m = fit_stochastic(y, forecaster=[("bad", bad), "random walk"])
    assert m.forecaster_name == "random walk"


def test_fitting_from_an_image_skips_the_backtest():
    _, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    img = SecondOrderImage.of(y)
    with pytest.warns(UserWarning, match="cannot backtest-select"):
        m = fit_stochastic(img)
    assert m.forecaster_name.endswith("(no backtest)")
    assert m.has_trend and m.has_cycle
    assert m.forecast(10).shape == (10,)
    with pytest.raises(TypeError, match="callable forecaster"):
        fit_stochastic(img, forecaster=lambda tr, h: np.zeros(h))


def test_unit_root_gate_verdicts():
    rng = np.random.default_rng(0)
    assert is_nonstationary(np.cumsum(rng.standard_normal(400)))
    assert not is_nonstationary(0.05 * np.arange(400)
                                + rng.standard_normal(400) * 3)
    assert not is_nonstationary(gen_ar1(800, 0.6, rng))
    assert not is_nonstationary(rng.standard_normal(400))
    assert not is_nonstationary(rng.standard_normal(8))    # too short to test


def test_unit_root_gate_measured_rates_at_n_100():
    # the AIC-selected augmentation lag keeps the false-positive rate near
    # alpha and the power against a random walk near the retired procedure's
    trials = 30
    hits = sum(
        is_nonstationary(np.random.default_rng(s).standard_normal(100))
        for s in range(trials))
    assert hits / trials < 0.10
    power = sum(
        is_nonstationary(
            np.cumsum(np.random.default_rng(s).standard_normal(100)))
        for s in range(trials))
    assert power / trials >= 0.80


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def test_hurst_stage_failure_warns_and_the_gate_stays_off(monkeypatch):
    import dtfit.stochastic.gates as g
    monkeypatch.setattr(g, "gph_slope", _boom)
    with pytest.warns(UserWarning, match="stage Hurst/long-memory failed"):
        m = fit_stochastic(gen_ar1(600, 0.6, np.random.default_rng(2)))
    assert not m.has_long_memory


def test_volatility_stage_failure_warns_in_both_branches(monkeypatch):
    import dtfit.stochastic.gates as g
    monkeypatch.setattr(g, "_persistence", _boom)
    gar = gen_garch(2000, 0.05, 0.08, 0.90, np.random.default_rng(3))
    with pytest.warns(UserWarning, match="stage GARCH/vol-clustering failed"):
        m = fit_stochastic(gar)
    assert not m.has_vol_clustering
    with pytest.warns(UserWarning,
                      match="stage unit-root vol-clustering failed"):
        m = fit_stochastic(np.cumsum(gar))
    assert not m.has_vol_clustering and m.regime.startswith("random walk")


@pytest.mark.parametrize("gen,attr", [
    (lambda r: gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, r)[1], "has_cycle"),
    (lambda r: gen_ar1(1500, 0.7, r), "has_mean_reversion"),
    (lambda r: gen_garch(4000, 0.05, 0.08, 0.90, r), "has_vol_clustering"),
])
def test_simulate_round_trip_recovers_regime(gen, attr):
    hits = sum(bool(getattr(fit_stochastic(
        fit_stochastic(gen(np.random.default_rng(10 + s))).simulate(seed=s)),
        attr)) for s in range(5))
    assert hits >= 4


def test_simulate_is_reproducible_and_finite():
    m = fit_stochastic(gen_ar1(800, 0.6, np.random.default_rng(0)))
    a, b = m.simulate(seed=7), m.simulate(seed=7)
    assert a.shape == (m.n,) and np.all(np.isfinite(a)) and np.allclose(a, b)
    assert m.simulate(n=200, seed=7).shape == (200,)


def test_simulate_student_t_has_fat_tails_and_unit_scale():
    from scipy.stats import kurtosis
    m = fit_stochastic(gen_ar1(3000, 0.7, np.random.default_rng(1)))
    sn = m.simulate(5000, seed=0, dist="normal")
    st = m.simulate(5000, seed=0, dist="t", df=4)
    assert float(kurtosis(st)) > float(kurtosis(sn)) + 1.5
    assert abs(np.std(st) - np.std(sn)) < 0.25 * np.std(sn)
    with pytest.raises(ValueError, match="dist must be"):
        m.forecast(5, return_conf_int=True, dist="cauchy")


def test_long_memory_simulate_variance_matches_sigma():
    from dtfit.stochastic.simulate import _sim_long_memory
    for H in (0.7, 0.9):
        stds = [np.std(_sim_long_memory(
            2000, H, 1.0, np.random.default_rng(s))) for s in range(6)]
        assert abs(float(np.mean(stds)) - 1.0) < 0.1


def test_model_wrapper_fits_through_the_catalog_convention():
    y = gen_ar1(1200, 0.7, np.random.default_rng(1))
    s = Stochastic()
    m = s.fit(y)
    assert isinstance(m, StochasticModel) and m is s.model_
    assert m.has_mean_reversion
    t = np.arange(y.size, dtype=float)
    assert Stochastic().fit(t, y).regime == m.regime
    assert Stochastic(forecaster="drift").fit(y).forecaster_name == "drift"


def test_model_wrapper_forwards_lag_and_nfreq(monkeypatch):
    import dtfit.models._stochastic as ms
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return fit_stochastic(*args, **kwargs)

    monkeypatch.setattr(ms, "fit_stochastic", spy)
    y = gen_ar1(200, 0.5, np.random.default_rng(0))
    Stochastic(lag=17, nfreq=33).fit(y)
    assert seen["lag"] == 17 and seen["nfreq"] == 33


def test_series_forecast_is_a_future_indexed_series():
    pd = pytest.importorskip("pandas")
    _, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    idx = pd.date_range("2000-01-01", periods=y.size, freq="D")
    ms = fit_stochastic(pd.Series(y, index=idx))
    ma = fit_stochastic(y)
    assert ms._index is not None and ma._index is None
    fc = ms.forecast(30)
    assert isinstance(fc, pd.Series) and len(fc) == 30
    assert fc.index.equals(
        pd.date_range(idx[-1] + pd.Timedelta(days=1), periods=30, freq="D"))
    assert np.array_equal(fc.to_numpy(), ma.forecast(30))


def test_series_conf_int_is_three_aligned_series():
    pd = pytest.importorskip("pandas")
    _, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(5))
    idx = pd.date_range("2000-01-01", periods=y.size, freq="D")
    pt, lo, hi = fit_stochastic(pd.Series(y, index=idx)).forecast(
        20, return_conf_int=True)
    for obj in (pt, lo, hi):
        assert isinstance(obj, pd.Series) and len(obj) == 20
    assert pt.index.equals(lo.index) and pt.index.equals(hi.index)
    assert np.all(hi.to_numpy() >= lo.to_numpy())
    apt, alo, ahi = fit_stochastic(y).forecast(20, return_conf_int=True)
    assert np.array_equal(pt.to_numpy(), apt)
    assert np.array_equal(lo.to_numpy(), alo)
    assert np.array_equal(hi.to_numpy(), ahi)


def test_ndarray_forecast_stays_ndarray():
    m = fit_stochastic(gen_ar1(1500, 0.7, np.random.default_rng(2)))
    assert m._index is None
    assert isinstance(m.forecast(10), np.ndarray)
    assert all(isinstance(o, np.ndarray)
               for o in m.forecast(10, return_conf_int=True))


def test_integer_index_forecast_continues_the_step():
    pd = pytest.importorskip("pandas")
    y = gen_ar1(1500, 0.7, np.random.default_rng(2))
    idx = pd.RangeIndex(10, 10 + 2 * y.size, 2)
    fc = fit_stochastic(pd.Series(y, index=idx)).forecast(5)
    assert isinstance(fc, pd.Series)
    assert list(fc.index) == [int(idx[-1]) + 2 * (i + 1) for i in range(5)]


def test_unit_root_series_forecast_is_indexed():
    pd = pytest.importorskip("pandas")
    y = np.cumsum(np.random.default_rng(1).standard_normal(1500))
    idx = pd.date_range("2010-01-01", periods=y.size, freq="D")
    ms = fit_stochastic(pd.Series(y, index=idx))
    assert ms.regime.startswith("random walk")
    fc = ms.forecast(12)
    assert isinstance(fc, pd.Series) and len(fc) == 12
    assert np.array_equal(fc.to_numpy(), fit_stochastic(y).forecast(12))
