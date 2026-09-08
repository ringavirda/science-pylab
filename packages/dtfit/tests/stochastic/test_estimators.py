"""The tier's estimators, each reading a functional of the second-order
image."""

import numpy as np
import pytest

from dtfit.image import Original
from dtfit.stochastic import (
    SecondOrderImage, adf_pvalue, ar1_reversion, ar_order, cycle_period,
    decompose_trend_cycle, dickey_fuller, fit_ar, fractional_difference,
    garch_persistence, hurst_aggvar, hurst_spectral, sample_acf,
)
from stochastic.processes import (
    gen_ar1, gen_ar2, gen_ar2_cycle, gen_arfima, gen_garch, gen_trend_cycle,
)


def test_estimators_take_a_series_an_original_or_an_image():
    y = gen_ar1(1500, 0.7, np.random.default_rng(0))
    img = SecondOrderImage.of(y)
    orig = Original(np.arange(1500.0), y)
    got = [ar1_reversion(d)["phi"] for d in (y, orig, img)]
    assert got[0] == pytest.approx(got[1]) and got[0] == pytest.approx(got[2])
    assert hurst_aggvar(img)["H"] == pytest.approx(hurst_aggvar(y)["H"])
    assert ar_order(img) == ar_order(y)


def test_sample_acf_white_noise_is_a_spike():
    acf = sample_acf(np.random.default_rng(0).standard_normal(4000), 12)
    assert acf[0] == pytest.approx(1.0)
    assert np.all(np.abs(acf[1:]) < 0.12)


def test_sample_acf_truncates_at_an_image_s_own_lag_budget():
    y = np.random.default_rng(0).standard_normal(4000)
    img = SecondOrderImage.of(y, lag=32)
    acf = sample_acf(img, 128)
    assert acf.shape == (33,)
    assert acf.shape == (min(128, img.lag) + 1,)


@pytest.mark.parametrize("phi", [0.6, 0.9])
def test_ar1_reversion_recovers_phi(phi):
    est = np.mean([ar1_reversion(
        gen_ar1(1500, phi, np.random.default_rng(30 + s)))["phi"]
        for s in range(3)])
    assert abs(est - phi) / phi < 0.12


def test_yule_walker_beats_the_exponential_acf_fit_on_ar1():
    """The default route is the Yule-Walker coefficient off the image's
    autocovariance; the exponential fit stays available and is the weaker of
    the two."""
    truth = 0.7
    yw, lsi = [], []
    for s in range(12):
        y = gen_ar1(1500, truth, np.random.default_rng(s))
        img = SecondOrderImage.of(y)
        yw.append(abs(ar1_reversion(img)["phi"] - truth))
        lsi.append(abs(ar1_reversion(img, method="lsi")["phi"] - truth))
    assert np.median(yw) < np.median(lsi)
    assert np.median(yw) < 0.03


def test_ar1_reversion_reports_tau_and_halflife():
    r = ar1_reversion(gen_ar1(2000, 0.8, np.random.default_rng(1)))
    assert r["tau"] == pytest.approx(-1.0 / np.log(r["phi"]))
    assert r["halflife"] == pytest.approx(r["tau"] * np.log(2.0))


def test_garch_persistence_recovers_alpha_plus_beta():
    truth = 0.08 + 0.90
    est = np.mean([garch_persistence(gen_garch(4000, 0.05, 0.08, 0.90,
                   np.random.default_rng(40 + s)))["persistence"]
                   for s in range(3)])
    assert abs(est - truth) / truth < 0.20


def test_hurst_aggvar_recovers_long_memory():
    ests = [hurst_aggvar(
        gen_arfima(4096, 0.3, np.random.default_rng(60 + s)))["H"]
        for s in range(5)]
    assert abs(float(np.mean(ests)) - 0.8) < 0.2
    assert all(0.0 <= e <= 1.0 for e in ests)


def test_hurst_aggvar_eac_matches_loglog():
    x = gen_arfima(4096, 0.3, np.random.default_rng(77))
    h_eac = hurst_aggvar(x, method="eac")["H"]
    h_lsi = hurst_aggvar(x, method="lsi")["H"]
    assert 0.0 < h_eac < 1.0 and abs(h_eac - h_lsi) < 0.15


def test_hurst_aggvar_needs_three_scales():
    with pytest.raises(RuntimeError, match="too few usable scales"):
        hurst_aggvar(SecondOrderImage.of(np.arange(40.0), scales=1))


def test_hurst_spectral_recovers_long_memory():
    err = np.mean([abs(hurst_spectral(gen_arfima(
        4096, 0.3, np.random.default_rng(10 + s)))["H"] - 0.8)
        for s in range(3)])
    assert err < 0.15


def test_blackman_tukey_hurst_is_unbiased_at_the_default_lag_budget():
    """The windowed spectrum at the default lag budget (256) is unbiased; a
    lag budget below 128 is biased low."""
    long_lag, short_lag = [], []
    for s in range(10):
        x = gen_arfima(4096, 0.3, np.random.default_rng(100 + s))
        long_lag.append(hurst_spectral(SecondOrderImage.of(x, lag=256),
                                       method="ols")["H"])
        short_lag.append(hurst_spectral(SecondOrderImage.of(x, lag=64),
                                        method="ols")["H"])
    assert abs(float(np.mean(long_lag)) - 0.8) < 0.04
    assert float(np.mean(short_lag)) - 0.8 < -0.02


def test_fit_ar_recovers_ar2_order_and_coefficients():
    orders = [ar_order(gen_ar2(2000, 0.5, 0.3, s)) for s in range(7)]
    assert max(set(orders), key=orders.count) == 2
    fit = fit_ar(gen_ar2(4000, 0.5, 0.3, 1))
    assert fit["order"] == 2
    assert np.allclose(np.asarray(fit["phi"], dtype=float), [0.5, 0.3],
                       atol=0.08)
    assert fit["sigma"] == pytest.approx(1.0, abs=0.1)


def test_ar_order_white_noise_is_zero():
    assert ar_order(np.random.default_rng(0).standard_normal(2000)) == 0
    assert fit_ar(np.random.default_rng(0).standard_normal(2000))["order"] == 0


def test_fractional_difference_special_cases():
    x = np.cumsum(np.random.default_rng(0).standard_normal(64))
    assert np.allclose(fractional_difference(x, 0.0), x)
    assert np.allclose(fractional_difference(x, 1.0)[1:], np.diff(x))


def test_fractional_difference_whitens_long_memory():
    y = gen_arfima(4096, 0.3, np.random.default_rng(2))
    d = hurst_spectral(y)["d"]
    w = fractional_difference(y, d)
    assert (abs(hurst_spectral(w)["H"] - 0.5)
            < abs(hurst_spectral(y)["H"] - 0.5))


def test_cycle_period_recovers_the_period():
    est = np.mean([cycle_period(gen_ar2_cycle(
        1500, 16.0, 0.97, np.random.default_rng(50 + s)))["period"]
        for s in range(3)])
    assert abs(est - 16.0) / 16.0 < 0.15


def test_decompose_recovers_trend_and_cycle():
    t, y = gen_trend_cycle(600, 0.02, 50.0, 3.0, 1.0, np.random.default_rng(0))
    dec = decompose_trend_cycle(t, y, max_harmonics=1)
    assert abs(dec["period"] - 50.0) / 50.0 < 0.10
    assert abs(dec["slope"] - 0.02) / 0.02 < 0.30
    assert dec["trend"].shape == (600,) and dec["cycle"].shape == (600,)
    assert np.std(dec["residual"]) < 1.3
    fc = dec["forecast"](10)
    assert fc.shape == (10,) and np.all(np.isfinite(fc))
    # the Original form takes both axes from the object
    same = decompose_trend_cycle(Original(t, y), max_harmonics=1)
    assert same["slope"] == pytest.approx(dec["slope"])
    with pytest.raises(ValueError, match="an Original"):
        decompose_trend_cycle(SecondOrderImage.of(y))


def _adf_design(x, lag):
    """ADF (ct) regression design at ``lag`` difference lags, built directly
    in numpy so this reference stays independent of the shipped estimator.
    The response is the difference ``dy_t``; the columns are
    ``[y_{t-1}, dy_{t-1}, .., dy_{t-lag}, 1, t]``, with the level lag first so
    that its coefficient is the gamma being tested."""
    dx = np.diff(x)
    nobs = dx.size - lag
    cols = [x[lag:lag + nobs]]                          # level lag y_{t-1}
    for j in range(1, lag + 1):
        cols.append(dx[lag - j:lag - j + nobs])         # dy_{t-j}
    cols.append(np.ones(nobs))                          # const
    cols.append(np.arange(1, nobs + 1, dtype=float))    # linear trend
    return dx[lag:], np.column_stack(cols)


def _direct_dickey_fuller(y, lags):
    """The tau statistic of a direct OLS of the constant+trend ADF
    regression at a fixed lag, the reference ``dickey_fuller`` is checked
    against."""
    y1, x1 = _adf_design(np.asarray(y, dtype=float), lags)
    beta, *_ = np.linalg.lstsq(x1, y1, rcond=None)
    resid = y1 - x1 @ beta
    dof = y1.size - x1.shape[1]
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se = np.sqrt(s2 * xtx_inv[0, 0])
    return float(beta[0] / se)


def test_dickey_fuller_agrees_with_a_direct_ols_reference():
    """A numpy-only reference, checked without skipping so the parity claim
    behind dropping the vendored ADF regression stays verified when
    statsmodels is not installed."""
    agree = 0
    cases = []
    for s in range(3):
        r = np.random.default_rng(200 + s)
        cases.append(np.cumsum(r.standard_normal(600)))
        cases.append(0.05 * np.arange(600) + r.standard_normal(600) * 3.0)
        cases.append(gen_ar1(600, 0.9, r))
        cases.append(3.0 * np.sin(2 * np.pi * np.arange(600) / 20.0)
                     + 0.5 * r.standard_normal(600))
    for x in cases:
        n = x.size
        p = max(1, int(min(12.0 * (n / 100.0) ** 0.25, 12, 64 - 2)))
        img = SecondOrderImage.of(x, lag=64, nfreq=64)
        shipped = dickey_fuller(img, lags=p)["tau"]
        ref = _direct_dickey_fuller(x, p)
        agree += (adf_pvalue(shipped) < 0.05) == (adf_pvalue(ref) < 0.05)
    assert agree == len(cases)


def test_dickey_fuller_reads_the_unit_root_off_the_image():
    walk = np.cumsum(np.random.default_rng(0).standard_normal(800))
    assert dickey_fuller(walk)["pvalue"] > 0.05
    ar = gen_ar1(800, 0.5, np.random.default_rng(1))
    assert dickey_fuller(ar)["pvalue"] < 0.05
    assert adf_pvalue(-30.0) == 0.0 and adf_pvalue(5.0) == 1.0


def test_dickey_fuller_agrees_with_statsmodels_verdicts():
    """The routing statistic is computed from the image's autocovariances; its
    unit-root verdict is what has to agree with the reference test."""
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    agree = 0
    cases = []
    for s in range(3):
        r = np.random.default_rng(100 + s)
        cases.append(np.cumsum(r.standard_normal(600)))
        cases.append(0.05 * np.arange(600) + r.standard_normal(600) * 3.0)
        cases.append(gen_ar1(600, 0.9, r))
        cases.append(3.0 * np.sin(2 * np.pi * np.arange(600) / 20.0)
                     + 0.5 * r.standard_normal(600))
    for x in cases:
        n = x.size
        maxlag = int(min(12 * (n / 100.0) ** 0.25, 12, n // 3))
        ref = sm.adfuller(x, regression="ct", maxlag=maxlag, autolag="AIC")
        img = SecondOrderImage.of(x, lag=64, nfreq=64)
        agree += (ref[1] < 0.05) == (dickey_fuller(img)["pvalue"] < 0.05)
    assert agree >= len(cases) - 1
