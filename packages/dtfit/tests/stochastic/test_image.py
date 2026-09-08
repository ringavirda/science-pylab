"""The stochastic tier's second-order image: exactness under chunking and
merging, and the read-outs every gate reads."""

import numpy as np
import pytest

from dtfit.image import Original
from dtfit.stochastic import SecondOrderImage
from stochastic.processes import gen_ar2


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


def ar1(n, phi, seed, sigma=1.0, burn=200):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def direct_acov(y, lag):
    n = y.size
    m = y.mean()
    return np.array([np.sum((y[k:] - m) * (y[:n - k] - m)) / n
                     for k in range(lag + 1)])


def chunked(y, cuts, lag, nfreq, scales):
    """The image of ``y`` built as consecutive images and merged."""
    parts = []
    prev = 0
    for c in list(cuts) + [y.size]:
        if c <= prev:
            continue
        parts.append(SecondOrderImage(lag, nfreq, scales, t0=prev)
                     .update(y[prev:c]))
        prev = c
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p)
    return out


def chunked_right(y, cuts, lag, nfreq, scales):
    """The image of ``y`` built as consecutive images and merged right to
    left, so each merge's receiver is the shorter, earlier chunk rather
    than the long running total."""
    parts = []
    prev = 0
    for c in list(cuts) + [y.size]:
        if c <= prev:
            continue
        parts.append(SecondOrderImage(lag, nfreq, scales, t0=prev)
                     .update(y[prev:c]))
        prev = c
    out = parts[-1]
    for p in reversed(parts[:-1]):
        out = p.merge(out)
    return out


def direct_dickey_fuller(y, lags):
    """The tau statistic of a direct OLS of the constant+trend ADF
    regression at a fixed lag, the reference dickey_fuller is checked
    against."""
    y1, x1 = _adf_design(np.asarray(y, dtype=float), lags)
    beta, *_ = np.linalg.lstsq(x1, y1, rcond=None)
    resid = y1 - x1 @ beta
    dof = y1.size - x1.shape[1]
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se = np.sqrt(s2 * xtx_inv[0, 0])
    return float(beta[0] / se)


def test_autocovariance_is_exact_under_chunking_and_merging():
    y = ar1(5000, 0.7, 31)
    whole = SecondOrderImage(64, 128, 8).update(y)
    stream = SecondOrderImage(64, 128, 8)
    for s in range(0, 5000, 733):
        stream.update(y[s:s + 733])
    merged = chunked(y, range(733, 5000, 733), 64, 128, 8)
    assert np.max(np.abs(whole.acov() - stream.acov())) < 1e-14
    assert np.max(np.abs(whole.acov() - merged.acov())) < 1e-14
    assert np.max(np.abs(merged.acov() - direct_acov(y, 64))) < 1e-14


def test_increment_and_square_autocovariances_are_exact():
    y = ar1(5000, 0.7, 31)
    merged = chunked(y, range(733, 5000, 733), 64, 128, 8)
    dy = np.diff(y)
    assert np.max(np.abs(
        merged.acov_increments() - direct_acov(dy, 64))) < 1e-14
    assert np.max(np.abs(
        merged.acov_volatility() - direct_acov(dy * dy, 64))) < 1e-13
    assert np.max(np.abs(
        merged.acov_squares() - direct_acov(y * y, 64))) < 1e-13


def test_trend_and_detrended_autocovariance_are_exact():
    y = ar1(5000, 0.7, 31)
    t = np.arange(5000.0)
    whole = SecondOrderImage(64, 128, 8).update(y)
    merged = chunked(y, range(733, 5000, 733), 64, 128, 8)
    slope, icpt = np.polyfit(t, y, 1)
    assert whole.trend()[0] == pytest.approx(slope, rel=1e-12)
    assert whole.trend()[1] == pytest.approx(icpt, rel=1e-9)
    assert merged.trend()[0] == pytest.approx(slope, rel=1e-12)
    e = y - (icpt + slope * t)
    assert np.max(np.abs(merged.detrended_acov() - direct_acov(e, 64))) < 1e-14


def test_aggregated_variance_is_exact_and_matches_direct_blocks():
    y = ar1(5000, 0.7, 31)
    whole = SecondOrderImage(64, 128, 8).update(y)
    merged = chunked(y, range(733, 5000, 733), 64, 128, 8)
    mw, vw = whole.aggregated_variance()
    mm, vm = merged.aggregated_variance()
    assert np.array_equal(mw, mm)
    assert np.max(np.abs(vw - vm)) < 1e-14
    # the scales with at least eight complete blocks: 2^0 .. 2^(log2(n) - 3)
    assert mw.tolist() == [float(1 << j) for j in range(9)]
    for m, v in zip(mm.astype(int), vm):
        nb = 5000 // m
        blocks = y[:nb * m].reshape(nb, m).mean(axis=1)
        assert v == pytest.approx(float(blocks.var(ddof=1)), rel=1e-12)


def test_merge_is_exact_under_a_right_fold_and_a_binary_tree():
    y = ar1(4000, 0.7, 5)
    whole = SecondOrderImage(16, 32, 10).update(y)
    scale = float(np.max(np.abs(whole.acov())))
    fields = ("ss", "nb", "lead_sum", "lead_len", "part_sum", "part_len")

    # a right fold makes each merge's receiver the short, earlier chunk,
    # the one that may not itself reach a block boundary at every scale
    right = chunked_right(y, range(500, 4000, 500), 16, 32, 10)
    assert np.max(np.abs(right.acov() - whole.acov())) < 1e-10 * scale
    for name in fields:
        assert np.max(np.abs(
            getattr(right, name) - getattr(whole, name))) < 1e-9
    mw, vw = whole.aggregated_variance()
    mr, vr = right.aggregated_variance()
    assert np.array_equal(mw, mr)
    assert np.max(np.abs(vw - vr)) < 1e-9

    # a binary-tree merge of unaligned leaves
    parts = [SecondOrderImage(16, 32, 10, t0=500 * i)
             .update(y[500 * i:500 * (i + 1)]) for i in range(8)]
    while len(parts) > 1:
        parts = [parts[i].merge(parts[i + 1])
                 for i in range(0, len(parts), 2)]
    tree = parts[0]
    assert np.max(np.abs(tree.acov() - whole.acov())) < 1e-10 * scale
    for name in fields:
        assert np.max(np.abs(
            getattr(tree, name) - getattr(whole, name))) < 1e-9


def test_merge_handles_an_empty_image_on_either_side():
    y = ar1(500, 0.7, 6)
    img = SecondOrderImage(16, 32, 4).update(y)
    left = SecondOrderImage(16, 32, 4).merge(img)
    right = img.merge(SecondOrderImage(16, 32, 4, t0=500))
    assert left.n == img.n == right.n
    assert np.array_equal(left.acov(), img.acov())
    assert np.array_equal(right.acov(), img.acov())


def test_mean_matches_the_sample_mean():
    y = ar1(500, 0.5, 8)
    img = SecondOrderImage(16, 32, 4).update(y)
    assert img.mean() == pytest.approx(float(np.mean(y)), rel=1e-12)


def test_fixed_grid_dft_is_exact_under_merging_and_matches_the_direct_sum():
    y = ar1(5000, 0.7, 31)
    whole = SecondOrderImage(64, 128, 8).update(y)
    merged = chunked(y, range(733, 5000, 733), 64, 128, 8)
    scale = float(np.max(np.abs(whole.dft())))
    assert np.max(np.abs(whole.dft() - merged.dft())) < 1e-12 * scale
    f = np.arange(128) / 256.0
    direct = np.exp(-2j * np.pi * np.outer(f, np.arange(5000.0))) @ y
    assert np.max(np.abs(merged.dft() - direct)) < 1e-9 * scale


def test_exactness_holds_over_random_chunkings_and_budgets():
    # the fixed cut at 1 leaves a single-sample first part, an unaligned
    # global index; the rest are random cuts, down to a single sample
    rng = np.random.default_rng(7)
    for trial in range(12):
        n = int(rng.integers(200, 1500))
        lag = int(rng.integers(4, 64))
        scales = int(rng.integers(1, 6))
        y = (np.cumsum(rng.standard_normal(n)) if trial % 2
             else rng.standard_normal(n))
        whole = SecondOrderImage(lag, 32, scales).update(y)
        cuts = sorted({1, 2, int(rng.integers(3, 7))}
                      | {int(c) for c in rng.integers(1, n, size=4)})
        merged = chunked(y, cuts, lag, 32, scales)
        for name in ("c_y", "c_dy", "c_d2", "c_sq"):
            a, b = getattr(whole, name), getattr(merged, name)
            assert np.max(np.abs(a - b)) < 1e-10 * float(np.max(np.abs(a)))
        scale = float(np.max(np.abs(whole.acov())))
        assert np.max(np.abs(whole.acov() - merged.acov())) < 1e-12 * scale
        assert np.max(np.abs(whole.detrended_acov()
                             - merged.detrended_acov())) < 1e-10 * scale
        dscale = float(np.max(np.abs(whole.dft())))
        assert np.max(np.abs(whole.dft() - merged.dft())) < 1e-10 * dscale
        assert whole.trend()[0] == pytest.approx(merged.trend()[0], rel=1e-10)
        _, vw = whole.aggregated_variance()
        _, vm = merged.aggregated_variance()
        if vw.size:
            assert np.max(np.abs(vw - vm)) < 1e-11 * float(np.max(vw))


def test_global_indices_stay_exact_two_million_samples_into_a_stream():
    """The index moments are closed forms in Python integers, so a block far
    down a stream reads the trend and the residual autocovariance the same
    samples read at index zero."""
    y = ar1(5000, 0.7, 31)
    t0 = 2_000_000
    near = SecondOrderImage(64, 128, 8).update(y)
    far = SecondOrderImage(64, 128, 8, t0=t0).update(y)
    assert far.sum_t == sum(range(t0, t0 + y.size))
    assert far.sum_t2 == sum(t * t for t in range(t0, t0 + y.size))
    slope, icpt = near.trend()
    assert far.trend()[0] == pytest.approx(slope, rel=1e-9)
    assert far.trend()[1] == pytest.approx(icpt - slope * t0, rel=1e-9)
    g_far, g_near = far.detrended_acov(), near.detrended_acov()
    assert np.all(np.isfinite(g_far))
    assert np.max(np.abs(g_far - g_near)) < 1e-8 * abs(g_near[0])


def test_state_round_trips_through_json():
    import json

    y = ar1(600, 0.5, 4)
    img = SecondOrderImage(32, 64, 5).update(y)
    back = SecondOrderImage(32, 64, 5).restore(
        json.loads(json.dumps(img.state())))
    assert back.n == img.n
    assert np.array_equal(back.acov(), img.acov())
    assert np.array_equal(back.dft(), img.dft())
    assert back.trend() == img.trend()


def test_merge_rejects_mismatched_or_non_consecutive_images():
    y = ar1(400, 0.5, 5)
    a = SecondOrderImage(16, 32, 4, t0=0).update(y[:200])
    b = SecondOrderImage(16, 32, 4, t0=200).update(y[200:])
    assert a.merge(b).n == 400
    with pytest.raises(ValueError, match="consecutive"):
        a.merge(SecondOrderImage(16, 32, 4, t0=199).update(y[199:]))
    with pytest.raises(ValueError, match="lag, nfreq and scales"):
        a.merge(SecondOrderImage(8, 32, 4, t0=200).update(y[200:]))
    with pytest.raises(ValueError, match="sample grid"):
        a.merge(SecondOrderImage(16, 32, 4, t0=200, dx=2.0).update(y[200:]))


def test_of_requires_a_uniform_grid_and_two_samples():
    y = ar1(300, 0.5, 6)
    x = np.arange(300.0)
    img = SecondOrderImage.of(Original(x, y))
    assert img.n == 300 and img.lag == 256
    assert img.scales == 5          # floor(log2(300)) - 3
    x[100] = 100.5
    with pytest.raises(ValueError, match="uniformly sampled"):
        SecondOrderImage.of(Original(x, y))
    with pytest.raises(ValueError, match="at least 2 samples"):
        SecondOrderImage.of(np.array([1.0]))


def test_constructor_rejects_impossible_budgets():
    for kwargs in ({"lag": 0}, {"nfreq": 0}, {"scales": -1}):
        with pytest.raises(ValueError):
            SecondOrderImage(**kwargs)
    with pytest.raises(ValueError, match="t0"):
        SecondOrderImage(8, 8, 2, t0=-1)
    with pytest.raises(ValueError, match="dx"):
        SecondOrderImage(8, 8, 2, dx=0.0)
    with pytest.raises(ValueError, match="at least one sample"):
        SecondOrderImage(8, 8, 2).update(np.zeros(0))
    with pytest.raises(ValueError, match="1-D"):
        SecondOrderImage(8, 8, 2).update(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        SecondOrderImage(8, 8, 2).update(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="no samples"):
        SecondOrderImage(8, 8, 2).mean()


def test_position_axis_carries_into_the_trend_and_the_domain():
    y = 0.5 + 0.25 * np.arange(200.0)
    img = SecondOrderImage.of(Original(3.0 + 0.5 * np.arange(200.0), y))
    slope, icpt = img.trend()
    assert slope == pytest.approx(0.5, rel=1e-12)      # 0.25 per sample / dx
    assert icpt == pytest.approx(-1.0, rel=1e-9)       # 0.5 - 0.5 * 3.0
    assert img.domain == pytest.approx((3.0, 102.5))
    assert img.first() == pytest.approx(0.5)
    assert img.last() == pytest.approx(50.25)
    assert img.carry().size == img.lag + 1


def test_spectrum_and_seasonal_read_a_clean_sinusoid():
    t = np.arange(600.0)
    y = 3.0 * np.sin(2 * np.pi * t / 50.0) + 0.5
    img = SecondOrderImage.of(y, lag=64, nfreq=512)
    f, s = img.spectrum(n_freq=200)
    assert f.size == 200 and np.argmax(s) == pytest.approx(600 / 50 - 1, abs=1)
    d = img.seasonal(max_harmonics=1)
    assert d["period"] == pytest.approx(50.0, rel=2e-3)
    assert d["amp"] == pytest.approx(3.0, rel=2e-2)
    assert abs(np.angle(np.exp(1j * d["phase"]))) < 0.05
    # the line at f = 1/50 falls between grid bins; the leakage-corrected
    # amplitude carries the whole variance of a noiseless sinusoid anyway
    assert d["strength"] == pytest.approx(1.0, abs=0.01)
    # a forced frequency skips the search and fits at that period
    forced = img.seasonal(max_harmonics=1, freq=1.0 / 50.0)
    assert forced["period"] == pytest.approx(50.0, rel=1e-12)
    assert forced["amp"] == pytest.approx(3.0, rel=2e-2)


def test_the_grid_resolves_the_record_and_says_so_when_it_cannot():
    t = np.arange(4000.0)
    y = 3.0 * np.sin(2 * np.pi * t / 50.0) + 0.5
    # of raises the grid to 2^ceil(log2 n) / 2, one record frequency per bin
    img = SecondOrderImage.of(y, lag=64, nfreq=512)
    assert img.nfreq == 2048
    d = img.seasonal(max_harmonics=1)
    assert d["period"] == pytest.approx(50.0, rel=1e-3)
    assert d["amp"] == pytest.approx(3.0, rel=1e-3)
    assert d["strength"] == pytest.approx(1.0, abs=0.01)
    # a stream fixes its grid before it knows how long the record will be;
    # under-resolved, the same line reads period 50.6 and amplitude 2.23
    coarse = SecondOrderImage(64, 512, 8).update(y)
    with pytest.warns(UserWarning, match="frequency grid resolves"):
        under = coarse.seasonal(max_harmonics=1)
    assert abs(under["period"] - 50.0) > 20 * abs(d["period"] - 50.0)
    assert abs(under["amp"] - 3.0) > 0.5


def test_seasonal_recovers_two_harmonics_by_bic():
    t = np.arange(600.0)
    y = (2.0 * np.sin(2 * np.pi * t / 50.0)
         + 1.0 * np.sin(2 * np.pi * 2 * t / 50.0 + 0.3) + 0.5)
    img = SecondOrderImage.of(y, lag=64, nfreq=512)
    d = img.seasonal(max_harmonics=2)
    assert d["n_harmonics"] == 2
    a, b = d["coef"][0], d["coef"][1]
    a2, b2 = d["coef"][2], d["coef"][3]
    assert np.hypot(a, b) == pytest.approx(2.0, abs=0.05)
    assert np.hypot(a2, b2) == pytest.approx(1.0, abs=0.05)


def test_seasonal_is_empty_on_a_record_too_short_to_carry_a_cycle():
    d = SecondOrderImage.of(np.arange(4.0), lag=2, nfreq=8).seasonal()
    assert d["n_harmonics"] == 0 and d["amp"] == 0.0
    assert not np.isfinite(d["period"])


def test_dickey_fuller_separates_a_walk_from_a_stationary_series():
    walk = np.cumsum(np.random.default_rng(0).standard_normal(800))
    tau_walk = SecondOrderImage.of(walk, lag=64, nfreq=64).dickey_fuller()
    tau_ar = SecondOrderImage.of(
        ar1(800, 0.5, 1), lag=64, nfreq=64).dickey_fuller()
    # -3.42 is the five percent critical value of the constant-plus-trend
    # regression, the one the p-value surface is built on
    assert tau_walk > -3.42 > tau_ar
    # a degenerate record leaves the normal equations singular
    assert not np.isfinite(SecondOrderImage.of(np.zeros(200)).dickey_fuller())


def test_dickey_fuller_matches_a_direct_ols_regression():
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(2000)
    shipped = SecondOrderImage.of(noise, lag=64, nfreq=64).dickey_fuller(12)
    assert shipped == pytest.approx(direct_dickey_fuller(noise, 12), rel=0.03)

    x = ar1(800, 0.5, 3)
    shipped = SecondOrderImage.of(x, lag=64, nfreq=64).dickey_fuller(12)
    assert shipped == pytest.approx(direct_dickey_fuller(x, 12), rel=0.03)


def test_dickey_fuller_selects_the_lag_by_aic():
    # an exact AR(2) reparametrizes with one differenced lag, so AIC never
    # drops to 0 the way it does on white noise
    ar2_lags = [SecondOrderImage.of(
        gen_ar2(600, 0.6, -0.3, s), lag=64, nfreq=64
    ).dickey_fuller(return_lag=True)[1] for s in range(60)]
    assert min(ar2_lags) >= 1

    wn_lags = [SecondOrderImage.of(
        np.random.default_rng(s).standard_normal(600), lag=64, nfreq=64
    ).dickey_fuller(return_lag=True)[1] for s in range(60)]
    assert sum(p == 0 for p in wn_lags) >= 30


def test_dickey_fuller_does_not_over_reject_a_unit_root_with_ma_noise():
    # the augmentation lags exist for exactly this family: a unit root whose
    # innovation is a moving average, not white noise
    trials = 200
    rejections = 0
    for seed in range(trials):
        e = np.random.default_rng(seed).standard_normal(801)
        y = np.cumsum(e[1:] - 0.8 * e[:-1])
        tau = SecondOrderImage.of(y, lag=64, nfreq=64).dickey_fuller()
        if tau < -3.42:
            rejections += 1
    # 43/200 measured against a nominal five percent test
    assert rejections / trials < 0.30


def test_residual_autocovariance_removes_the_trend_exactly():
    t = np.arange(1200.0)
    noise = ar1(1200, 0.6, 9)
    y = 2.0 + 0.01 * t + noise
    img = SecondOrderImage.of(y, lag=32, nfreq=64)
    g = img.residual_acov()
    slope, icpt = np.polyfit(t, y, 1)
    e = y - (icpt + slope * t)
    assert np.max(np.abs(g - direct_acov(e, 32))) < 1e-12
    # the seasonal part subtracts the harmonics' own autocovariance
    ys = y + 3.0 * np.sin(2 * np.pi * t / 40.0)
    imgs = SecondOrderImage.of(ys, lag=32, nfreq=512)
    seas = imgs.seasonal(max_harmonics=1)
    gr = imgs.residual_acov(seas["freq"], seas["coef"])
    g_e = direct_acov(e, 32)
    # the harmonic's own autocovariance goes; what is left of the amplitude's
    # sampling error moves the variance by a few percent of A^2/2
    assert gr[0] == pytest.approx(g_e[0], rel=0.03)
    assert gr[1] / gr[0] == pytest.approx(g_e[1] / g_e[0], abs=0.012)
    # without the subtraction the harmonic dominates the lag-1 correlation
    gd = imgs.detrended_acov()
    assert gd[1] / gd[0] > 0.85
