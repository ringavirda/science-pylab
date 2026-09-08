"""Model-class routing and fallbacks in ``auto_forecast``.

It picks a model class, then applies the no-structure and divergence guards,
falling back down a chain that ends at persistence.
"""

import numpy as np
import pytest

from dtfit import auto_forecast
from sklearn.metrics import r2_score


def test_auto_forecast_logistic_growth():
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))  # saturating growth
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert fc.shape == (30,)
    assert r2_score(y[n_tr:], fc) > 0.9


def test_auto_forecast_seasonal_beats_persistence():
    t = np.linspace(0, 20, 400)
    y = 0.5 * t + 3.0 * np.sin(2 * np.pi * t / 2.0)
    n_tr = 320
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=80, period=40.0)
    persist = np.full(80, y[n_tr - 1])
    rmse_fc = np.sqrt(np.mean((y[n_tr:] - fc) ** 2))
    rmse_p = np.sqrt(np.mean((y[n_tr:] - persist) ** 2))
    assert rmse_fc < rmse_p


def test_auto_forecast_no_structure_guard_fires_on_reverting_ramp():
    # The ramp reverses out of sample. A structured fit of the training tail
    # extrapolates the local slope and overshoots persistence badly.
    t = np.linspace(0, 30, 300)
    y = np.r_[np.linspace(0, 10, 150), np.linspace(10, 0, 150)]  # up then down
    fc = auto_forecast(t[:240], y[:240], horizon=60, model="poly")
    # Either guard is an acceptable outcome: persist, or bound the runaway.
    # A blow-up is not.
    rng = float(np.ptp(y[:240]))
    assert np.all(np.abs(fc - y[239]) <= 5 * rng)


def test_auto_forecast_random_walk_stays_bounded():
    # A random walk has no structure to extrapolate. Losing to persistence is
    # the honest outcome here; losing by more than 3x is not.
    rng = np.random.default_rng(6)
    y = np.cumsum(rng.normal(0, 1.0, 300))
    t = np.arange(y.size, dtype=float)
    n_tr = 240
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=60)
    persist = np.full(60, y[n_tr - 1])
    rmse_fc = np.sqrt(np.mean((y[n_tr:] - fc) ** 2))
    rmse_p = np.sqrt(np.mean((y[n_tr:] - persist) ** 2))
    assert rmse_fc < 3.0 * rmse_p


def test_auto_forecast_explicit_random_walk():
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    fc = auto_forecast(t, y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])


def test_auto_forecast_zero_horizon():
    t = np.linspace(0, 1, 50)
    assert auto_forecast(t, np.exp(t), horizon=0).size == 0


def test_auto_forecast_failed_model_warns_and_falls_back_to_linear():
    # A series ending negative inverts the logistic seed's L bounds and the
    # fit raises. The fallback has to warn, not silently swap models, and it
    # must still return a horizon-length forecast. Twenty samples keep the
    # no-structure guard (n < 24) out of the way.
    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, -1.0, 20)
    with pytest.warns(
        UserWarning, match=r"logistic fit failed .*falling back to linear"
    ):
        fc = auto_forecast(t, y, horizon=5, model="logistic")
    assert fc.shape == (5,)
    assert np.all(np.isfinite(fc))
    # The fallback records which primary model failed.
    assert fc.model_name == "linear (logistic failed)"


def test_auto_forecast_divergent_poly_failed_linear_falls_to_persistence(
    monkeypatch,
):
    # The second fallback: the divergence guard fires and its linear refit
    # raises as well, leaving persistence at y[-1]. Reaching that needs a stub
    # where poly "fits" but runs away and linear raises.
    import dtfit.forecast as fc_mod

    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, 2.0, 20)

    def fake_fit_model(chosen, x, yy, t_all, period):
        # _fit_model returns (values, FittingResult); this stub has no real
        # fit behind it, hence the None.
        if chosen == "poly":
            return np.full(t_all.size, 1e12), None  # wildly divergent prediction
        raise RuntimeError("boom")

    monkeypatch.setattr(fc_mod, "_fit_model", fake_fit_model)
    with pytest.warns(
        UserWarning, match=r"linear fit failed .*falling back to persistence"
    ):
        fc = fc_mod.auto_forecast(t, y, horizon=5, model="poly")
    assert fc.shape == (5,)
    assert np.allclose(fc, y[-1])
    # The persistence fallback records why it persisted, and carries no fit.
    assert fc.model_name == "persistence (linear failed)"
    assert fc.result is None and fc.std_band is None


# ForecastResult: provenance, the std band, and ndarray semantics
def _horizon_std_ok(fc, horizon):
    # .std_band is either absent or a finite band of exactly horizon length.
    return fc.std_band is None or (
        isinstance(fc.std_band, np.ndarray)
        and fc.std_band.shape == (horizon,)
        and np.all(np.isfinite(fc.std_band))
    )


def test_auto_forecast_returns_ndarray_and_forecastresult():
    from dtfit.forecast import ForecastResult

    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert isinstance(fc, np.ndarray)
    assert isinstance(fc, ForecastResult)
    assert fc.shape == (30,)
    assert np.all(np.isfinite(fc))
    assert len(fc) == 30
    assert np.allclose(fc, np.asarray(fc))  # ndarray semantics intact
    assert fc.model_name == "logistic"
    from dtfit import FittingResult

    assert isinstance(fc.result, FittingResult)
    assert _horizon_std_ok(fc, 30)


def test_auto_forecast_std_band_is_delta_method_predict_std():
    # On a covariance-bearing fit the band is populated; it is exactly the
    # predict std evaluated on the extrapolated future grid.
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    fc = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    assert fc.std_band is not None
    assert isinstance(fc.std_band, np.ndarray)
    assert fc.std_band.shape == (30,)
    assert np.all(np.isfinite(fc.std_band))
    x = t[:n_tr]
    dx = float(np.mean(np.diff(x)))
    future = x[-1] + dx * np.arange(1, 31)
    _, std_ref = fc.result.predict(future, return_std=True)
    np.testing.assert_allclose(fc.std_band, std_ref)


def test_forecastresult_does_not_shadow_ndarray_std():
    # The band lives on .std_band, leaving ndarray.std free to reduce.
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t + 0.5 * t**2
    fc = auto_forecast(t, y, horizon=10, model="poly")
    assert np.isfinite(fc.std())
    assert np.isfinite(np.std(fc))


def test_auto_forecast_explicit_model_name_and_result():
    from dtfit import FittingResult

    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t + 0.5 * t**2
    fc = auto_forecast(t, y, horizon=10, model="poly")
    assert fc.model_name == "poly"
    assert isinstance(fc.result, FittingResult)
    assert _horizon_std_ok(fc, 10)


def test_auto_forecast_random_walk_provenance():
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    fc = auto_forecast(t, y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])
    assert fc.model_name == "random_walk"
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_no_structure_provenance(monkeypatch):
    # The guard is forced rather than provoked. Its factor-8 threshold is
    # deliberately hard to trip on real data, and a natural trigger would tie
    # the test to one particular fit realisation.
    import dtfit.forecast as fc_mod

    monkeypatch.setattr(fc_mod, "_no_structure", lambda *a, **k: True)
    t = np.linspace(0, 30, 300)
    y = 1.0 + 2.0 * t
    n_tr = 240
    fc = fc_mod.auto_forecast(t[:n_tr], y[:n_tr], horizon=60, model="poly")
    assert fc.model_name.startswith("persistence (") and "no structure" in fc.model_name
    assert fc.model_name == "persistence (poly no structure)"
    assert np.allclose(fc, y[n_tr - 1])
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_divergence_guard_reports_provenance(monkeypatch):
    # The stub makes poly "fit" and then run away. Here the divergence guard's
    # linear refit succeeds; the forecast is then a real linear one.
    import dtfit.forecast as fc_mod
    from dtfit import FittingResult

    real_fit_model = fc_mod._fit_model
    t = np.linspace(0, 1, 20)
    y = np.linspace(1.0, 2.0, 20)

    def fake_fit_model(chosen, x, yy, t_all, period):
        if chosen == "poly":
            return np.full(t_all.size, 1e12), None  # divergent
        return real_fit_model(chosen, x, yy, t_all, period)

    monkeypatch.setattr(fc_mod, "_fit_model", fake_fit_model)
    fc = fc_mod.auto_forecast(t, y, horizon=5, model="poly")
    assert isinstance(fc, fc_mod.ForecastResult)
    assert fc.model_name == "linear (poly diverged)"
    assert isinstance(fc.result, FittingResult)
    assert _horizon_std_ok(fc, 5)


def test_auto_forecast_zero_horizon_is_forecastresult():
    from dtfit.forecast import ForecastResult

    fc = auto_forecast(np.linspace(0, 1, 50), np.exp(np.linspace(0, 1, 50)), horizon=0)
    assert isinstance(fc, ForecastResult)
    assert fc.size == 0
    assert fc.result is None and fc.std_band is None


def test_auto_forecast_std_fallback_when_no_covariance(monkeypatch):
    # A missing covariance leaves std_band as None instead of crashing; the
    # values and model_name still arrive.
    import dtfit.forecast as fc_mod

    real_fit_model = fc_mod._fit_model
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t

    def strip_cov(chosen, x, yy, t_all, period):
        pred, res = real_fit_model(chosen, x, yy, t_all, period)
        res.cov = None  # simulate a method that produced no covariance
        return pred, res

    monkeypatch.setattr(fc_mod, "_fit_model", strip_cov)
    fc = fc_mod.auto_forecast(t, y, horizon=10, model="linear")
    assert fc.std_band is None
    assert fc.model_name == "linear"
    assert fc.shape == (10,)


# pandas interop
def test_auto_forecast_series_values_match_ndarray_path():
    # A pandas input adds .index and nothing else; the values are identical.
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    idx = pd.date_range("2020-01-01", periods=n_tr, freq="D")
    fc_np = auto_forecast(t[:n_tr], y[:n_tr], horizon=30)
    fc_pd = auto_forecast(pd.Series(t[:n_tr], index=idx), y[:n_tr], horizon=30)
    np.testing.assert_array_equal(np.asarray(fc_np), np.asarray(fc_pd))


def test_auto_forecast_series_datetimeindex_future_index_and_to_series():
    pd = pytest.importorskip("pandas")
    from dtfit.forecast import ForecastResult

    n_tr = 90
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    idx = pd.date_range("2020-01-01", periods=n_tr, freq="D")
    xs = pd.Series(t[:n_tr], index=idx)

    fc = auto_forecast(xs, y[:n_tr], horizon=30)
    # Still the ndarray-subclass forecast; the index is additive.
    assert isinstance(fc, ForecastResult)
    assert fc.shape == (30,)
    # The future index continues the input at its inferred daily step.
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert len(fc.index) == 30
    assert fc.index[0] == idx[-1] + pd.Timedelta(days=1)
    s = fc.to_series()
    assert isinstance(s, pd.Series)
    assert s.index.equals(fc.index)
    np.testing.assert_array_equal(s.to_numpy(), np.asarray(fc))


def test_auto_forecast_integer_index_continues_by_step():
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 3, 200)
    y = 1.0 + 2.0 * t
    idx = pd.RangeIndex(0, 200)
    fc = auto_forecast(pd.Series(t, index=idx), y, horizon=10, model="linear")
    assert fc.index is not None
    assert list(fc.index) == list(range(200, 210))


def test_auto_forecast_ndarray_has_no_index_and_to_series_raises():
    # The error message depends on the environment: "no future index" when
    # pandas is installed, "requires pandas" when it is not. Matching either
    # keeps the test correct both ways.
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    fc = auto_forecast(t[:90], y[:90], horizon=30)
    assert fc.index is None
    with pytest.raises(ValueError, match="index|pandas"):
        fc.to_series()


def test_auto_forecast_series_no_freq_index_is_none():
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 12, 120)
    y = 1000.0 / (1 + np.exp(-0.8 * (t - 6)))
    n_tr = 90
    # widening gaps, so pd.infer_freq gives up and returns None
    idx = pd.DatetimeIndex(
        pd.Timestamp("2020-01-01") + pd.to_timedelta(np.cumsum(np.arange(1, n_tr + 1)), "D")
    )
    fc = auto_forecast(pd.Series(t[:n_tr], index=idx), y[:n_tr], horizon=30)
    assert fc.index is None
    with pytest.raises(ValueError):
        fc.to_series()


def test_auto_forecast_persistence_path_carries_future_index():
    # random_walk stands in for the persistence return paths generally.
    pd = pytest.importorskip("pandas")
    t = np.linspace(0, 5, 100)
    y = np.sin(t)
    idx = pd.date_range("2021-06-01", periods=100, freq="D")
    fc = auto_forecast(pd.Series(t, index=idx), y, horizon=10, model="random_walk")
    assert np.allclose(fc, y[-1])
    assert isinstance(fc.index, pd.DatetimeIndex) and len(fc.index) == 10
    s = fc.to_series()
    assert s.index.equals(fc.index)


def test_forecastresult_slice_drops_length_dependent_metadata():
    # A slice is shorter than the horizon. Carrying the per-step .index and
    # .std_band over would leave them at length 30: fc[:3].std_band would read
    # as a full band, and fc[:3].to_series() would crash.
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2020-01-01", periods=90, freq="D")
    x = pd.Series(np.linspace(0, 12, 90), index=idx)
    y = pd.Series(1000.0 / (1 + np.exp(-0.8 * (np.linspace(0, 12, 90) - 6))),
                  index=idx)
    fc = auto_forecast(x, y, horizon=30)
    assert len(fc.index) == 30 and len(fc.std_band) == 30
    sl = fc[:3]
    assert len(sl) == 3
    assert sl.index is None
    assert sl.std_band is None
    with pytest.raises(ValueError, match="future index"):
        sl.to_series()
    # The full forecast is untouched by the slice.
    assert (fc.to_series().index == fc.index).all()
    assert np.isfinite(fc.std())
