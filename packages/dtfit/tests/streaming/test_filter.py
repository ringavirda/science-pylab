"""Streaming filters: online EAC and LSI, both with NIS drift detection."""

import numpy as np
import pytest

from dtfit import ImageFilter, EACFilter, LSIFilter


def test_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = EACFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001],
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_params_and_predict_shapes():
    flt = EACFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_partial_fit_returns_self():
    flt = EACFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


def _feed_step(low: float, high: float, n: int = 120):
    """Run the filter over a clean level step and return
    (filter, direction)."""
    rng = np.random.default_rng(0)
    levels = (np.r_[np.full(n, low), np.full(n, high)]
              + rng.normal(0, 0.02, 2 * n))
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20, q_diag=[1e-3, 1e-3],
        adaptive_window=False,
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    return flt, direction


def test_drift_detected_upward():
    flt, direction = _feed_step(1.0, 3.0)
    assert flt.n_drifts_ >= 1
    assert direction == 1  # +1 means an upward shift


def test_drift_detected_downward():
    flt, direction = _feed_step(3.0, 1.0)
    assert flt.n_drifts_ >= 1
    assert direction == -1


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_coast_matches_predict_in_support(cls):
    """coast() reduces to predict() at and before the anchor (in-window)."""
    rng = np.random.default_rng(0)
    t = np.arange(40) * 0.1
    y = 2.0 + 3.0 * t + 0.5 * t**2 + rng.normal(0, 0.05, t.size)
    flt = cls("c0 + c1*t + c2*t**2", "t", p0=[y[0], 0.0, 0.0], window_size=15)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    xin = t[-5:]  # all at or before the anchor
    assert np.allclose(flt.coast(xin), flt.predict(xin))


def test_coast_stays_bounded_where_cubic_diverges():
    """Past the window a fitted cubic's predict() runs away. The order-1 coast
    holds constant velocity and stays on a straight line."""
    rng = np.random.default_rng(1)
    t = np.arange(40) * 0.1
    y = 2.0 + 3.0 * t - 0.1 * t**3 + rng.normal(0, 0.05, t.size)
    flt = LSIFilter("c0 + c1*t + c2*t**2 + c3*t**3", "t",
                    p0=[y[0], 0.0, 0.0, 0.0], window_size=15, order=5)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    a = t[-1]
    gap = a + np.arange(1, 21) * 0.1  # 2 s past support
    c1 = flt.coast(gap, order=1)
    assert np.all(np.isfinite(c1))
    # an order-1 coast is exactly linear in (x - a), so its second difference
    # vanishes where the raw cubic predict() keeps curving
    assert np.allclose(np.diff(c1, 2), 0.0, atol=1e-9)
    assert not np.allclose(np.diff(flt.predict(gap), 2), 0.0, atol=1e-9)


def test_coast_rejects_regressor_models():
    flt = LSIFilter("c0 + c1*t + S", "t", regressors="S")
    with pytest.raises(NotImplementedError):
        flt.coast(np.array([1.0]))


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_predict_cov_is_nonneg_shaped_and_contracts(cls):
    """predict_cov maps the parameter covariance into output space. The result
    is non-negative, shaped like x, and contracts as the estimate is
    identified."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 800)
    y = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    flt = cls("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=40,
              q_diag=[1e-4, 1e-4])
    flt.partial_fit(t[0], y[0])
    early = float(flt.predict_cov(np.array([10.0]))[0])
    for ti, yi in zip(t[1:], y[1:]):
        flt.partial_fit(ti, yi)
    v = flt.predict_cov(np.array([5.0, 10.0, 15.0]))
    assert v.shape == (3,)
    assert np.all(v >= 0.0)
    assert float(flt.predict_cov(np.array([10.0]))[0]) < early
    band = np.sqrt(flt.predict_cov(np.array([10.0])))
    assert np.all(np.isfinite(band))


def test_no_false_drift_on_stable_signal():
    # A single seed at the detector's own false-alarm rate (ruling 7: one
    # run in twenty-four) is a lottery in both directions, so this runs
    # the whole batch and bounds the total at that measured rate rather
    # than demanding zero.
    n_false = 0
    for seed in range(24):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 40, 2000)
        y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
        flt = EACFilter(
            "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
            q_diag=[0.05, 0.001], adaptive_window=False,
        )
        for ti, yi in zip(t, y):
            flt.partial_fit(ti, yi)
        n_false += flt.n_drifts_ > 0
    assert n_false <= 1


def test_block_order_tracks():
    # order > 1 gives the block basis more than one window.
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
    flt = EACFilter(
        "A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50,
        q_diag=[0.05, 0.001], order=4,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_subareas_detect_amplitude_jump_on_oscillation():
    # On an oscillation an amplitude jump nets to very little signed area;
    # a single scalar area barely registers it. Splitting the window into
    # sub-areas gives the energy NIS a chi^2(order) statistic with that many
    # independent channels, which is what catches the jump here.
    n = 900
    t = np.linspace(0, 40, n)
    half = n // 2
    amp = np.where(np.arange(n) < half, 2.0, 3.5)
    y = amp * np.sin(1.8 * t) + np.random.default_rng(3).normal(0, 0.2, n)

    def detect(order):
        """The block image with five windows."""
        flt = EACFilter(
            "A*sin(w*t)", "t", p0=[2.0, 1.8], window_size=60, order=order,
            q_diag=[3e-3, 1e-4], drift_reset="inflate", adaptive_window=False,
        )
        first, false = None, 0
        for i in range(n):
            flt.partial_fit(t[i], y[i])
            if flt.drift_flag_:
                if i < half:
                    false += 1
                elif first is None:
                    first = i
        return first, false

    first, false = detect(order=5)
    assert first is not None
    assert first - half < 60        # within one window stride of the shift
    assert false == 0               # no false alarm before it


def test_inflate_drift_reset_detects_step_and_keeps_window():
    rng = np.random.default_rng(0)
    levels = (np.r_[np.full(120, 1.0), np.full(120, 3.0)]
              + rng.normal(0, 0.02, 240))
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20,
        q_diag=[1e-3, 1e-3], drift_reset="inflate", adaptive_window=False,
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1
    # "inflate" keeps the sliding window populated where "full" clears it
    assert len(flt._t) > 0


# LSIFilter: streaming LSI, online integral least-squares.
def test_lsi_filter_tracks_stable_sine():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    flt = LSIFilter(
        "A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=50, order=5,
        q_diag=[1e-3, 5e-4],
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)

    p = flt.params_
    assert abs(p["A"] - 3.0) < 1.0
    assert abs(p["w"] - 1.5) < 0.5


def test_lsi_filter_recovers_exponential():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6, 600)
    y = 2.5 * np.exp(-0.6 * t) + rng.normal(0, 0.05, t.size)
    flt = LSIFilter(
        "a*exp(b*t)", "t", p0=[1.0, -0.2], window_size=50, order=5,
        q_diag=[1e-4, 1e-4],
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    p = flt.params_
    assert abs(p["a"] - 2.5) < 0.3
    assert abs(p["b"] + 0.6) < 0.15


def test_lsi_filter_params_and_predict_shapes():
    flt = LSIFilter("A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=10)
    assert set(flt.params_) == {"A", "w"}
    out = flt.predict(np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)


def test_lsi_filter_partial_fit_returns_self():
    flt = LSIFilter("A*sin(w*t)", "t")
    assert flt.partial_fit(0.0, 0.0) is flt


def test_lsi_filter_no_false_drift_on_stable_signal():
    # See test_no_false_drift_on_stable_signal: bounded at the measured
    # rate over the same batch of seeds, not asserted zero on one.
    n_false = 0
    for seed in range(24):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 40, 2000)
        y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)
        flt = LSIFilter(
            "A*sin(w*t)", "t", p0=[2.0, 1.5], window_size=50, order=5,
            q_diag=[1e-3, 5e-4], adaptive_window=False,
        )
        for ti, yi in zip(t, y):
            flt.partial_fit(ti, yi)
        n_false += flt.n_drifts_ > 0
    assert n_false <= 1


def test_lsi_filter_detects_level_step():
    rng = np.random.default_rng(0)
    levels = (np.r_[np.full(150, 1.0), np.full(150, 3.0)]
              + rng.normal(0, 0.02, 300))
    x = np.linspace(0, 1.5, levels.size)
    flt = LSIFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20, order=5,
        q_diag=[1e-3, 1e-3], adaptive_window=False,
    )
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1  # +1 means an upward level shift


def test_filter_survives_exp_overflow_without_nan_poisoning():
    """An unbounded model whose rate wanders must not NaN-poison the filter for
    good: predict() stays finite through a transient overflow.

    The climb model ``z0+c*(1-exp(-k*t))`` is the dangerous shape. Its
    time-constant can be driven toward the singular value; without the guard
    every later prediction comes back NaN.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 400)
    y = 5.0 + 4.0 * (1.0 - np.exp(-t / 3.0)) + rng.normal(0, 0.3, t.size)
    flt = EACFilter(
        "z0 + c*(1-exp(-k*t))", "t", p0=[3.0, 0.3, 4.0], window_size=40,
        q_diag=[1e-3, 1e-3, 1e-3], order=3,
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
        assert np.all(np.isfinite(flt.p)), "filter parameters went non-finite"
    pred = flt.predict(t)
    assert np.all(np.isfinite(pred)), "predict() returned NaN/inf"


def test_lsi_filter_survives_nonfinite_update():
    """The Legendre filter shares the same non-finite-update guard."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 12, 300)
    y = 5.0 + 4.0 * (1.0 - np.exp(-t / 3.0)) + rng.normal(0, 0.3, t.size)
    flt = LSIFilter(
        "z0 + c*(1-exp(-k*t))", "t", p0=[3.0, 0.3, 4.0], window_size=30,
        order=4, q_diag=[1e-3, 1e-3, 1e-3],
    )
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert np.all(np.isfinite(flt.p))
    assert np.all(np.isfinite(flt.predict(t)))


def test_last_residual_is_the_forecast_innovation():
    """``last_residual_`` is NaN before the window fills, then equals the
    one-step forecast error y_new - f(t_new; p) at the newest sample."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6, 200)
    y = 2.0 + 0.5 * t + 0.1 * t**2 + rng.normal(0, 0.2, t.size)
    flt = EACFilter(
        "c0 + c1*t + c2*t**2", "t", p0=[0.0, 0.0, 0.0], window_size=15,
        q_diag=[1e-2, 1e-2, 1e-2], order=3,
    )
    assert np.isnan(flt.last_residual_)  # nothing ingested yet
    seen_finite = False
    for ti, yi in zip(t, y):
        yhat = float(flt.predict(np.array([ti]))[0])  # pre-update prediction
        flt.partial_fit(ti, yi)
        if np.isfinite(flt.last_residual_):
            assert abs(flt.last_residual_ - (yi - yhat)) < 1e-9
            seen_finite = True
    assert seen_finite


def test_lsi_filter_exposes_last_residual():
    flt = LSIFilter(
        "c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
        q_diag=[1e-2, 1e-2],
    )
    assert np.isnan(flt.last_residual_)
    t = np.linspace(0, 5, 120)
    for ti in t:
        flt.partial_fit(ti, 1.0 + 0.5 * ti)
    assert np.isfinite(flt.last_residual_)


def test_accumulative_window_acquires_before_full():
    """The window is accumulative. Both filters give a usable estimate before
    window_size samples have arrived, with no full-window dead time. The
    estimate only improves once the window fills."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 24, 1200)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.05 * 3.0, t.size)
    for cls, kw in [(EACFilter, dict(window_size=60, order=2)),
                    (LSIFilter, dict(window_size=60, order=5))]:
        flt = cls("A*sin(w*t)", "t", p0=[1.0, 1.0], q_diag=[1e-3, 1e-3],
                  adaptive_window=False, **kw)
        assert 0 < flt.min_window < flt.W
        mid = None
        for i, (ti, yi) in enumerate(zip(t, y)):
            flt.partial_fit(ti, yi)
            if i == flt.W - 2:            # one step before the window fills
                mid = abs(flt.params_["A"] - 3.0) / 3.0
        assert mid is not None and mid < 0.5
        assert abs(flt.params_["A"] - 3.0) < 0.3


def test_static_models_grow_to_the_cap():
    """The adaptive window sizes itself from the data: a model that fits
    the window keeps growing to the cap whatever its shape, and the estimate
    then carries the whole record. A polynomial and an oscillation both
    reach it; the polynomial's cap is the record, so its intercept is fitted
    from every sample."""
    def run(expr, p0, true, T, n, seed, cap):
        ts = np.linspace(0, T, n)
        import sympy as sp
        sym = sp.Symbol("t")
        mdl = sp.sympify(expr)
        ps = sorted((s for s in mdl.free_symbols if s != sym), key=str)
        f = sp.lambdify([sym, *ps], mdl, "numpy")
        clean = f(ts, *[true[str(s)] for s in ps])
        y = clean + np.random.default_rng(seed).normal(
            0, 0.05 * (clean.std() + 1e-9), n)
        flt = LSIFilter(expr, "t", p0=p0, window_size=cap, order=5,
                        q_diag=[1e-4] * len(ps))
        for ti, yi in zip(ts, y):
            flt.partial_fit(ti, yi)
        err = np.mean([abs(flt.params_[str(s)] - true[str(s)]) /
                       abs(true[str(s)]) for s in ps]) * 100
        return flt._W_eff, err

    w_osc, e_osc = run("A*sin(w*t)", [1.0, 1.0], {"A": 3.0, "w": 1.5},
                       24, 1200, 0, 300)
    w_poly, e_poly = run("c0+c1*t+c2*t**2", [0.0, 0.0, 0.0],
                         {"c0": 1.0, "c1": 2.0, "c2": 0.5}, 6, 600, 0, 600)
    assert w_osc >= 270 and w_poly >= 540
    assert e_osc < 3.0 and e_poly < 6.0


def test_adaptive_window_collapses_and_regrows_on_drift():
    """On a regime change the adaptive window collapses back to min_window (to
    flush stale old-regime data) and then re-grows as the new regime is
    identified."""
    rng = np.random.default_rng(1)
    n = 900
    t = np.linspace(0, 40, n)
    half = n // 2
    amp = np.where(np.arange(n) < half, 2.0, 3.5)
    y = amp * np.sin(1.8 * t) + rng.normal(0, 0.2, n)
    flt = LSIFilter("A*sin(w*t)", "t", p0=[2.0, 1.8], window_size=120,
                    adaptive_window=True, order=6, q_diag=[3e-3, 1e-4],
                    drift_reset="inflate")
    W = np.empty(n)
    for i in range(n):
        flt.partial_fit(t[i], y[i])
        W[i] = flt._W_eff
    assert flt.n_drifts_ >= 1
    assert W[:half].max() > 3 * flt.min_window  # grew wide before the change
    post_min = W[half:half + 60].min()
    assert post_min <= flt.min_window + 2      # collapsed back to min_window
    assert W[-1] > post_min + 5                # then re-grew past the collapse
    assert abs(flt.params_["A"] - 3.5) < 0.4   # on the new amplitude


def test_adaptive_window_shrinks_on_maneuver():
    """The window shrinks while the model's residual over the window is
    autocorrelated and grows while the model fits. A static fit keeps its
    wide window. The same model tracks a manoeuvring signal on a shorter,
    more responsive window, with nothing tuned by hand."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 1500)
    # static: the line model matches the data, the residuals stay white and
    # the window grows wide
    y_static = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    fs = LSIFilter("c0 + c1*t", "t", p0=[2.0, 0.5], window_size=120,
                   adaptive_window=True, order=3, q_diag=[1e-3, 1e-3])
    for ti, yi in zip(t, y_static):
        fs.partial_fit(ti, yi)
    # manoeuvring: the same line model must chase a curving signal. It lags,
    # the residual autocorrelates into runs of one sign and the window shrinks
    y_man = 3.0 * np.sin(0.4 * t) + rng.normal(0, 0.05, t.size)
    fm = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=120,
                   adaptive_window=True, order=3, q_diag=[1e-3, 1e-3])
    for ti, yi in zip(t, y_man):
        fm.partial_fit(ti, yi)
    assert fm._W_eff < fs._W_eff // 2
    assert fm._resid_corr > fs._resid_corr      # the autocorrelation drives it
    assert fs._W_eff > 3 * fs.min_window        # the static fit stays wide


def test_block_adaptive_window_grows_to_the_cap_and_stays_accurate():
    """The block filter's adaptive window on a static oscillation grows to
    the cap like the Legendre filter's, and the estimate stays accurate
    there: the cap is the memory bound the caller sets."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 700)
    clean = 2.0 * np.sin(2.5 * t)
    y = clean + rng.normal(0, 0.05 * clean.std(), t.size)
    flt = EACFilter("A*sin(w*t)", "t", p0=[1.5, 2.0], window_size=300,
                    order=2, q_diag=[1e-4, 1e-4])
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert flt._W_eff >= 0.9 * flt.W
    assert abs(flt.params_["A"] - 2.0) / 2.0 < 0.1


def test_min_window_is_respected_and_clamped():
    """``min_window`` controls when acquisition starts and is clamped
    sanely."""
    f = LSIFilter("A*sin(w*t)", "t", window_size=40, order=5, min_window=12)
    assert f.min_window == 12
    # below the floor of order + 2 it is clamped up, above window_size down
    assert LSIFilter("A*sin(w*t)", "t", window_size=40, order=5,
                     min_window=1).min_window == 7        # order + 2
    assert LSIFilter("A*sin(w*t)", "t", window_size=40, order=5,
                     min_window=999).min_window == 40     # window_size
    # the block filter needs two samples per window.
    assert EACFilter(
        "A*sin(w*t)", "t", window_size=60, order=2
    ).min_window == 4


def test_inflate_scales_covariance_for_both_filters():
    """``inflate`` multiplies the parameter covariance, both with an explicit
    factor and with the configured default. It is the hook an external detector
    re-arms the filter through."""
    for cls, kw in [
        (EACFilter, dict(window_size=15)),
        (LSIFilter, dict(window_size=20, order=3)),
    ]:
        flt = cls("c0 + c1*t", "t", p0=[0.0, 0.0],
                  q_diag=[1e-2, 1e-2], drift_inflation=50.0, **kw)
        p0 = flt.P.copy()
        flt.inflate(7.0)
        assert np.allclose(flt.P, p0 * 7.0)
        flt.inflate()  # defaults to drift_inflation
        assert np.allclose(flt.P, p0 * 7.0 * 50.0)


# Robust mode: in-window residual winsorization rejects gross outliers.
def _outlier_sine(seed, frac):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 40, 1600)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.15, t.size)
    mask = rng.random(t.size) < frac
    y[mask] += rng.normal(0, 24.0, int(mask.sum()))  # gross spikes (~8x)
    return t, y


def test_robust_mode_resists_outliers_both_filters():
    """With 10% gross outliers the robust filter's fit over its last window
    stays far closer to the clean signal than the plain one's, for the block
    and the Legendre filter alike. The fit is judged by its prediction on
    that window: a 60-sample window of a sine cannot separate an aliased
    amplitude and frequency pair that predicts the same samples, so
    parameter identity is not the test."""
    for cls, kw in [(EACFilter, dict(window_size=60, order=5)),
                    (LSIFilter, dict(window_size=60, order=5))]:
        t, y = _outlier_sine(0, 0.10)
        clean = 3.0 * np.sin(1.5 * t)

        def err(robust):
            flt = cls("A*sin(w*t)", "t", p0=[1.0, 1.0],
                      q_diag=[1e-3, 1e-3], robust=robust, **kw)
            for ti, yi in zip(t, y):
                flt.partial_fit(ti, yi)
            tail = t[-60:]
            return float(np.sqrt(np.mean(
                (flt.predict(tail) - clean[-60:]) ** 2))) / 3.0

        assert err(True) < 0.5 * err(False)
        assert err(True) < 0.20


def test_robust_mode_clean_signal_matches_default():
    """On a clean signal the robust gate never fires; the estimate must not
    degrade against the plain filter."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 40, 1600)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.15, t.size)
    out = {}
    for robust in (False, True):
        flt = LSIFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=60,
                        order=5, q_diag=[1e-3, 1e-3], robust=robust)
        for ti, yi in zip(t, y):
            flt.partial_fit(ti, yi)
        out[robust] = abs(flt.params_["A"] - 3.0)
    assert out[True] < 0.3
    assert out[True] <= out[False] + 0.1


def test_robust_mode_still_detects_drift():
    """Winsorizing the residual around its median preserves a sustained shift,
    which leaves a genuine regime change detectable under robust mode."""
    rng = np.random.default_rng(0)
    levels = (np.r_[np.full(120, 1.0), np.full(120, 3.0)]
              + rng.normal(0, 0.02, 240))
    x = np.linspace(0, 1.0, levels.size)
    flt = EACFilter("a*exp(b*x)", "x", p0=[1.0, 0.0], window_size=20,
                    q_diag=[1e-3, 1e-3], robust=True, adaptive_window=False)
    direction = 0
    for xi, yi in zip(x, levels):
        flt.partial_fit(xi, yi)
        if flt.drift_flag_:
            direction = flt.last_drift_direction_
    assert flt.n_drifts_ >= 1
    assert direction == 1


# External regressors on LSI and EAC: the model may depend on measured
# side-channels as well as t, while the measurement stays integral or
# spectral.
def test_external_regressor_recovers_and_improves_both_filters():
    """A model ``c0 + c1*t + Sx`` carries a measured basis ``Sx`` as an
    external regressor. Both filters recover (c0, c1). The richer model
    fuses through the integral update well enough to beat the raw
    measurement."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 500)
    Sx = 0.5 * t**2 * np.sin(0.3 * t)      # an arbitrary measured channel
    truth = 3.0 - 0.8 * t + Sx
    y = truth + rng.normal(0, 0.3, t.size)
    raw = float(np.sqrt(np.mean((y - truth) ** 2)))
    for cls, kw in [(LSIFilter, dict(order=4)),
                    (EACFilter, dict(order=2))]:
        flt = cls("c0 + c1*t + Sx", "t", regressors="Sx", p0=[0.0, 0.0],
                  window_size=20, q_diag=[1e-3, 1e-3], **kw)
        sm = np.zeros_like(t)
        for i in range(t.size):
            flt.partial_fit(t[i], y[i], regressors={"Sx": Sx[i]})
            sm[i] = float(flt.predict(np.array([t[i]]),
                                       regressors={"Sx": Sx[i]})[0])
        p = flt.params_
        assert abs(p["c0"] - 3.0) < 0.4
        assert abs(p["c1"] + 0.8) < 0.1
        smoothing = float(np.sqrt(np.mean((sm[40:] - truth[40:]) ** 2)))
        assert smoothing < 0.5 * raw


def test_external_regressor_accepts_sequence_and_missing_raises():
    flt = LSIFilter("a*u + b*v", "t", regressors=["u", "v"], p0=[1.0, 1.0],
                    window_size=10, order=3)
    flt.partial_fit(0.0, 1.0, regressors=[2.0, 3.0])  # positional sequence
    with pytest.raises(ValueError):
        flt.partial_fit(0.1, 1.0)  # regressors required but omitted


def test_external_regressor_name_clashing_with_sympy_singleton():
    """A regressor named ``S`` collides with a SymPy singleton and is still
    usable: the parser binds regressor names to plain Symbols."""
    for cls in (LSIFilter, EACFilter):
        flt = cls("c0 + S", "t", regressors="S", p0=[0.0], window_size=10)
        assert "S" not in flt.params_  # S is the regressor, not a parameter
        for ti in np.linspace(0, 1, 25):
            flt.partial_fit(ti, 2.0 + ti, regressors={"S": ti})
        assert abs(flt.params_["c0"] - 2.0) < 0.2


def test_external_regressor_predict_needs_regressors():
    flt = EACFilter("a*u + b", "t", regressors="u", p0=[1.0, 0.0],
                    window_size=8)
    for ti in np.linspace(0, 1, 20):
        flt.partial_fit(ti, 2.0 * ti, regressors={"u": ti})
    out = flt.predict(np.array([0.0, 1.0]),
                      regressors={"u": np.array([0.0, 1.0])})
    assert out.shape == (2,)
    with pytest.raises(ValueError):
        flt.predict(np.array([0.0]))                  # no regressors supplied


def test_filter_presets_configure_and_track():
    # Presets are thin constructors over the full knob set; overrides win.
    t = np.linspace(0, 6, 300)
    y = 2.0 * np.sin(1.5 * t)
    f = LSIFilter.tracking("A*sin(w*x)", "x")
    assert f.adaptive_window is True
    for ti, yi in zip(t, y):
        f.partial_fit(ti, yi)
    assert abs(f.params_["A"] - 2.0) < 0.2

    g = EACFilter.robust("A*sin(w*x)", "x")
    assert g._robust is True and g.drift_reset == "inflate"


@pytest.mark.parametrize("cls,kw", [
    (EACFilter, dict(window_size=20, order=2)),
    (LSIFilter, dict(window_size=20, order=4)),
])
def test_nonfinite_sample_skipped_at_entry(cls, kw):
    """A NaN observation or timestamp mid-stream is skipped at ingestion with a
    ``RuntimeWarning``. The finiteness guard has to run before the sample can
    reach the window: once inside it would poison every innovation until it
    slid out again. State stays identical to the same stream without the bad
    sample; the next good sample updates normally."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 8, 160)
    y = 2.0 + 0.7 * t + rng.normal(0, 0.05, t.size)

    def make():
        return cls("c0 + c1*t", "t", p0=[1.0, 1.0],
                   q_diag=[1e-3, 1e-3], **kw)

    clean, dirty = make(), make()
    mid = 80
    for i, (ti, yi) in enumerate(zip(t, y)):
        clean.partial_fit(ti, yi)
        if i == mid:
            res_before = dirty.last_residual_
            with pytest.warns(RuntimeWarning,
                              match="non-finite sample skipped"):
                assert dirty.partial_fit(ti - 1e-4, float("nan")) is dirty
            with pytest.warns(RuntimeWarning,
                              match="non-finite sample skipped"):
                dirty.partial_fit(float("nan"), yi)     # NaN timestamp
            assert dirty.last_residual_ == res_before  # untouched by skips
        dirty.partial_fit(ti, yi)
        # identical at every step: the skips changed nothing; updating
        # resumes on the very next good sample
        np.testing.assert_array_equal(dirty.p, clean.p)
        np.testing.assert_array_equal(dirty.P, clean.P)


def test_nonfinite_regressor_skipped_at_entry():
    """A NaN in an external-regressor channel is skipped like a NaN in y."""
    for cls, kw in [(EACFilter, dict()), (LSIFilter, dict(order=3))]:
        flt = cls("a*u + b", "t", regressors="u", p0=[1.0, 0.0],
                  window_size=10, **kw)
        for ti in np.linspace(0, 1, 30):
            flt.partial_fit(ti, 2.0 * ti + 1.0, regressors={"u": ti})
        p_before = flt.p.copy()
        n_before = len(flt._t)
        with pytest.warns(RuntimeWarning, match="non-finite sample skipped"):
            flt.partial_fit(1.05, 3.1, regressors={"u": float("nan")})
        np.testing.assert_array_equal(flt.p, p_before)
        assert len(flt._t) == n_before   # nothing entered the window


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_drift_reset_validated_at_construction(cls):
    """``drift_reset`` accepts only 'full' or 'inflate'. Anything else raises
    at construction instead of silently behaving as 'full'."""
    for ok in ("full", "inflate"):
        assert cls("a*t", "t", drift_reset=ok).drift_reset == ok
    with pytest.raises(ValueError, match="drift_reset"):
        cls("a*t", "t", drift_reset="typo")


# Callable models: the filters accept a plain Python f(t, *params) in place of
# a SymPy-expression string and evaluate it numerically. The whole
# partial_fit/predict/params_/predict_cov path works. Only coast() and
# coast_cov() are unavailable, needing symbolic time-derivatives a callable
# cannot give.
def _sine_callable(t, A, w):
    """A plain callable twin of the ``"A*sin(w*t)"`` expression string."""
    return A * np.sin(w * t)


@pytest.mark.parametrize("cls,kw", [
    (EACFilter, dict(window_size=50, q_diag=[0.05, 0.001])),
    (LSIFilter, dict(window_size=50, order=5, q_diag=[1e-3, 5e-4])),
])
def test_callable_model_tracks_like_string(cls, kw):
    """A filter built from a callable ``f(t, A, w)`` recovers the sine about as
    well as the equivalent string-expression filter."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 2000)
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.3, t.size)

    def err(model):
        flt = cls(model, "t", p0=[1.0, 1.0], **kw)
        assert flt.model.symbolic is isinstance(model, str)
        for ti, yi in zip(t, y):
            flt.partial_fit(ti, yi)
        p = flt.params_
        return abs(p["A"] - 3.0) + abs(p["w"] - 1.5)

    e_str = err("A*sin(w*t)")
    e_call = err(_sine_callable)
    assert e_call < 1.2
    assert e_call < 2.0 * e_str + 0.2        # about as well as the string one


@pytest.mark.parametrize("cls,kw", [
    (EACFilter, dict(window_size=40, q_diag=[5e-3, 5e-3], order=2)),
    (LSIFilter, dict(window_size=40, order=4, q_diag=[5e-3, 5e-3])),
])
def test_callable_model_tracks_drifting_parameter(cls, kw):
    """A callable-backed filter tracks a drifting parameter as well as the
    string-expression filter does. The signal here is a fixed-rate sine on a
    linearly ramping amplitude; both follow it with comparable RMS error."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 30, 1500)
    amp = 2.0 + 0.05 * t          # amplitude drifts from 2.0 to 3.5
    y = amp * np.sin(1.2 * t) + rng.normal(0, 0.1, t.size)

    def track_rms(model):
        flt = cls(model, "t", p0=[2.0, 1.2], **kw)
        est = np.full(t.size, np.nan)
        for i, (ti, yi) in enumerate(zip(t, y)):
            flt.partial_fit(ti, yi)
            est[i] = flt.params_["A"]
        good = ~np.isnan(est)
        return float(np.sqrt(np.mean((est[good] - amp[good]) ** 2)))

    rms_str = track_rms("A*sin(w*t)")
    rms_call = track_rms(_sine_callable)
    assert rms_call < 0.5
    assert rms_call < 1.5 * rms_str + 0.1     # about as well as the string


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_callable_model_coast_raises(cls):
    """coast() and coast_cov() need symbolic time-derivatives. On a
    callable-backed filter they raise, with a message pointing at the
    expression-string form."""
    t = np.linspace(0, 6, 300)
    y = 3.0 * np.sin(1.2 * t)
    flt = cls(_sine_callable, "t", p0=[3.0, 1.2], window_size=40)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    for method in (flt.coast, flt.coast_cov):
        with pytest.raises(NotImplementedError, match="symbolic model"):
            method(np.array([100.0]))
    # predict / predict_cov remain available on the callable filter.
    assert flt.predict(np.array([1.0, 2.0])).shape == (2,)
    assert np.all(flt.predict_cov(np.array([1.0, 2.0])) >= 0.0)


def test_callable_model_preserves_signature_order():
    """A callable's parameters follow signature order, not the sorted order a
    string model gets: ``f(t, w, A)`` yields names ``('w', 'A')``, with ``p0``
    lined up the same way."""
    def f(t, w, A):
        return A * np.sin(w * t)

    for cls in (EACFilter, LSIFilter):
        flt = cls(f, "t", p0=[1.3, 2.5])
        assert list(flt.params_) == ["w", "A"]
        assert flt.params_["w"] == 1.3 and flt.params_["A"] == 2.5


def test_callable_model_param_names_kwarg_for_varargs():
    """A callable with an opaque ``f(t, *ps)`` signature is usable once
    ``param_names`` is supplied, which then fixes the parameter order."""
    def g(t, *ps):
        A, w = ps
        return A * np.sin(w * t)

    for cls in (EACFilter, LSIFilter):
        flt = cls(g, "t", param_names=["A", "w"], p0=[3.0, 1.5])
        assert list(flt.params_) == ["A", "w"]
    # without param_names the opaque callable is rejected up front
    with pytest.raises(ValueError):
        EACFilter(g, "t")


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_callable_model_rejects_regressors(cls):
    """External regressors are symbolic-only. Combining them with a callable
    model raises at construction: there is no symbolic form to split."""
    with pytest.raises(ValueError, match="regressors"):
        cls(_sine_callable, "t", regressors="S")


@pytest.mark.parametrize("cls", [EACFilter, LSIFilter])
def test_callable_model_predict_cov(cls):
    """predict_cov works for a callable model too, contracting as data
    arrives."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 20, 800)
    y = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)

    def line(x, c0, c1):
        return c0 + c1 * x

    flt = cls(line, "t", p0=[0.0, 0.0], window_size=40, q_diag=[1e-4, 1e-4])
    flt.partial_fit(t[0], y[0])
    early = float(flt.predict_cov(np.array([10.0]))[0])
    for ti, yi in zip(t[1:], y[1:]):
        flt.partial_fit(ti, yi)
    assert float(flt.predict_cov(np.array([10.0]))[0]) < early
    p = flt.params_
    assert abs(p["c0"] - 2.0) < 0.3 and abs(p["c1"] - 0.5) < 0.1


def test_aliases_fix_the_basis():
    assert LSIFilter("a*t", "t").basis.name == "legendre"
    assert EACFilter("a*t", "t", order=3).basis.name == "block"
    assert EACFilter("a*t", "t", order=3).order == 3
    with pytest.raises(TypeError, match="basis"):
        LSIFilter("a*t", "t", basis="block")
    f = ImageFilter("a*t", "t", basis="block", order=4)
    assert isinstance(f, ImageFilter) and not isinstance(f, EACFilter)


def test_retired_keywords_are_rejected():
    for kw in (dict(r=1.0), dict(adapt_r=True), dict(adapt_noise=True),
               dict(n_sub=2)):
        with pytest.raises(TypeError):
            LSIFilter("a*t", "t", **kw)
    assert not hasattr(LSIFilter("a*t", "t"), "param_cov_")
    assert not hasattr(LSIFilter("a*t", "t"), "stderr_")


def test_order_and_window_are_validated():
    with pytest.raises(ValueError):
        LSIFilter("a*t + b", "t", window_size=5, order=5)  # no room
    with pytest.raises(ValueError):
        EACFilter("a*t + b", "t", order=1)  # fewer coefficients than params
    with pytest.raises(ValueError):
        LSIFilter("a*t", "t", window_size=2)


def test_innovation_and_nis_are_whitened():
    """On a static line with white noise the whitened innovation has
    ``order + 1`` components and, once the estimate has settled, the
    normalized innovation squared averages near its degrees of freedom."""
    rng = np.random.default_rng(4)
    t = np.linspace(0, 30, 1500)
    y = 2.0 + 0.5 * t + rng.normal(0, 0.1, t.size)
    flt = LSIFilter("c0 + c1*t", "t", p0=[2.0, 0.5], window_size=40, order=4,
                    q_diag=[1e-6, 1e-6], adaptive_window=False)
    assert np.all(np.isnan(flt.innovation_)) and np.isnan(flt.nis_)
    nis = []
    for i, (ti, yi) in enumerate(zip(t, y)):
        flt.partial_fit(ti, yi)
        if i > 300:
            nis.append(flt.nis_)
    assert flt.innovation_.shape == (5,)
    assert 0.5 * 5 < float(np.mean(nis)) < 2.0 * 5


def test_drift_runs_through_the_shared_detector():
    from dtfit.streaming import DriftDetector
    flt, _ = _feed_step(1.0, 3.0)
    assert isinstance(flt.detector, DriftDetector)
    assert flt.detector.dim == flt.basis.n_coef
    assert flt.n_drifts_ == flt.detector.n_drifts_ >= 1


def test_noise_var_fixed_is_used_verbatim():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 300)
    y = 1.0 + 0.3 * t + rng.normal(0, 0.05, t.size)
    fixed = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
                      noise_var=0.05 ** 2, adaptive_window=False)
    free = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
                     adaptive_window=False)
    for ti, yi in zip(t, y):
        fixed.partial_fit(ti, yi)
        free.partial_fit(ti, yi)
    assert fixed.noise_var == 0.05 ** 2 and free.noise_var is None
    assert abs(fixed.params_["c1"] - 0.3) < 0.02
    assert abs(free.params_["c1"] - 0.3) < 0.02


def test_noise_var_fixed_changes_the_gain():
    """A ``noise_var`` far from the window's own residual variance changes
    the gain: fixing it that wrong makes ``P`` settle much larger than the
    self-estimating filter's, which is only possible if ``s2`` really comes
    from ``noise_var`` and not from the window's residual, as
    ``test_noise_var_fixed_is_used_verbatim`` alone cannot show (a
    ``noise_var`` close to the true variance makes both filters agree
    regardless of which one is actually used)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 300)
    y = 1.0 + 0.3 * t + rng.normal(0, 0.05, t.size)
    fixed = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
                      noise_var=100.0, adaptive_window=False)
    free = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=20, order=3,
                     adaptive_window=False)
    for ti, yi in zip(t, y):
        fixed.partial_fit(ti, yi)
        free.partial_fit(ti, yi)
    assert np.all(np.diag(fixed.P) > 5.0 * np.diag(free.P))


def test_irregular_sampling_is_imaged_at_the_actual_positions():
    """The window image is built on the samples' own positions: a jittered
    grid recovers the parameters as well as a uniform one."""
    rng = np.random.default_rng(5)
    n = 600
    t = np.sort(np.cumsum(rng.uniform(0.02, 0.08, n)))
    y = 3.0 * np.sin(1.5 * t) + rng.normal(0, 0.1, n)
    flt = LSIFilter("A*sin(w*t)", "t", p0=[2.0, 1.4], window_size=50, order=5,
                    q_diag=[1e-3, 1e-4], adaptive_window=False)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    assert abs(flt.params_["A"] - 3.0) < 0.3
    assert abs(flt.params_["w"] - 1.5) < 0.1


def test_window_image_uses_the_actual_positions_not_a_uniform_grid():
    """``_window_ops`` images a clustered window at its own normalized
    positions, not at a uniform grid of the same length: the two bases
    differ pointwise, which the parameter-recovery tolerance above is too
    loose to show (the innovation ``Phi^T(y - f)`` uses the same ``Phi`` on
    both sides, so a wrong ``u`` re-weights rather than misfits)."""
    rng = np.random.default_rng(1)
    n = 40
    t = np.sort(np.cumsum(rng.uniform(0.02, 2.0, n)))
    flt = LSIFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=n, order=5,
                    adaptive_window=False)
    Phi_actual, _, _ = flt._window_ops(t)
    Phi_uniform = flt.basis.evaluate(np.linspace(-1.0, 1.0, n))
    assert not np.allclose(Phi_actual, Phi_uniform, atol=1e-6)


def test_stream_rejection_leaves_the_window_untouched():
    """A sample outside the attached stream's domain must not enter the
    filter's own window either: the stream is fed before the window is
    mutated, so its rejection leaves both untouched."""
    from dtfit import ImageStream

    flt = LSIFilter("a*t", "t", window_size=10,
                    stream=ImageStream("legendre", 4, domain=(0.0, 5.0)))
    with pytest.raises(ValueError, match="domain"):
        flt.partial_fit(10.0, 1.0)
    assert flt._t == [] and flt._y == []


CT_BASELINE_RMSE = 1.310   # mean over seeds 0-4; see test_coordinated_turn


def _coordinated_turn(seed, n=300, dt=0.2, speed=15.0, omega=0.1, sigma=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    heading = omega * t
    x = speed / omega * np.sin(heading)
    y = speed / omega * (1.0 - np.cos(heading))
    return t, x, y, x + rng.normal(0, sigma, n), y + rng.normal(0, sigma, n)


def _ct_forecast_rmse(seed, horizon=1):
    t, tx, ty, zx, zy = _coordinated_turn(seed)
    expr = "c0 + c1*t + c2*t**2 + c3*t**3"
    fx = LSIFilter(expr, "t", p0=[zx[0], 0.0, 0.0, 0.0], window_size=15,
                   order=5, q_diag=[1e-2] * 4)
    fy = LSIFilter(expr, "t", p0=[zy[0], 0.0, 0.0, 0.0], window_size=15,
                   order=5, q_diag=[1e-2] * 4)
    err = []
    for i in range(t.size - horizon):
        fx.partial_fit(t[i], zx[i])
        fy.partial_fit(t[i], zy[i])
        if i >= 30:
            px = float(fx.coast(np.array([t[i + horizon]]))[0])
            py = float(fy.coast(np.array([t[i + horizon]]))[0])
            dx = px - tx[i + horizon]
            dy = py - ty[i + horizon]
            err.append(dx ** 2 + dy ** 2)
    return float(np.sqrt(np.mean(err)))


def test_coordinated_turn_forecast_gate():
    """The synthetic coordinated-turn benchmark: a constant-speed turn at
    5 Hz with 1 m noise per axis, tracked per axis by a cubic on a
    15-sample adaptive window, one-step forecast by ``coast``. The RMSE over
    five seeds must not exceed the pinned baseline by more than 5 percent
    (the adaptive-window regression of the spec, on synthetic data)."""
    rmse = float(np.mean([_ct_forecast_rmse(s) for s in range(5)]))
    assert rmse < 3.0            # a broken filter forecasts worse than noise
    assert rmse <= 1.05 * CT_BASELINE_RMSE


def test_result_is_a_batch_fit_on_the_window():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 800)
    y = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    flt = LSIFilter("c0 + c1*t", "t", p0=[0.0, 0.0], window_size=200,
                    order=4, adaptive_window=False)
    with pytest.raises(ValueError, match="min_window"):
        flt.result()
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    res = flt.result()
    assert res.n_obs == 200 and res.image_order == 4
    assert res.basis_name == "legendre"
    assert set(res.params) == {"c0", "c1"}
    assert abs(res.params["c1"] - 0.5) < 0.02
    assert res.cov is not None and res.cov.shape == (2, 2)
    se = res.stderr()
    assert all(np.isfinite(v) and v > 0 for v in se.values())
    assert res.x_range == (float(t[-200]), float(t[-1]))


def test_result_for_regressor_and_callable_models():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 20, 500)
    Sx = 0.5 * t ** 2 * np.sin(0.3 * t)
    y = 3.0 - 0.8 * t + Sx + rng.normal(0, 0.3, t.size)
    flt = LSIFilter("c0 + c1*t + Sx", "t", regressors="Sx", p0=[0.0, 0.0],
                    window_size=30, order=4, adaptive_window=False)
    for i in range(t.size):
        flt.partial_fit(t[i], y[i], regressors={"Sx": Sx[i]})
    res = flt.result()
    assert abs(res.params["c1"] + 0.8) < 0.1 and res.cov is not None

    def line(x, c0, c1):
        return c0 + c1 * x

    y2 = 2.0 + 0.5 * t + rng.normal(0, 0.05, t.size)
    flc = EACFilter(line, "t", p0=[0.0, 0.0], window_size=100, order=3,
                    adaptive_window=False)
    for ti, yi in zip(t, y2):
        flc.partial_fit(ti, yi)
    resc = flc.result()
    assert resc.basis_name == "block" and abs(resc.params["c1"] - 0.5) < 0.02


def test_result_covariance_covers_the_filter_error():
    """Over replicates the ratio of the filter's RMS parameter error to
    the mean standard error ``result()`` reports lies within a factor of
    two at the default process noise, static and tracking alike (the
    spec's uncertainty gate). With a small process noise the filter
    integrates information across overlapping windows: on the static
    case the reported error is conservative, but under drift it is
    anti-conservative, bounded here rather than shrunk to the measured
    ratio."""
    def ratio(drift, q, seeds=12):
        errs, ses = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            t = np.linspace(0, 30, 900)
            c1 = 0.5 + (0.01 * t if drift else 0.0)
            y = 2.0 + c1 * t + rng.normal(0, 0.1, t.size)
            kw = {} if q is None else {"q_diag": [q, q]}
            flt = LSIFilter("c0 + c1*t", "t", p0=[2.0, 0.5], window_size=60,
                            order=4, **kw)
            for ti, yi in zip(t, y):
                flt.partial_fit(ti, yi)
            res = flt.result()
            k = len(flt._t)
            tm = float(np.mean(t[-k:]))
            # the window's local slope
            slope = 0.5 + 0.02 * tm if drift else 0.5
            errs.append(flt.params_["c1"] - slope)   # the filter's error
            ses.append(res.stderr()["c1"])
        return float(np.sqrt(np.mean(np.square(errs)))) / float(np.mean(ses))

    for drift in (False, True):
        r = ratio(drift, None)
        assert 0.5 < r < 2.0, (drift, r)
    assert ratio(False, 1e-4) < 1.0
    assert ratio(True, 1e-4) < 2.0


def test_stream_hook_accumulates_every_sample():
    from dtfit import ImageStream
    rng = np.random.default_rng(2)
    t = np.linspace(0, 10, 400)
    y = 1.3 * np.exp(0.2 * t) + rng.normal(0, 0.05, t.size)
    direct = ImageStream("legendre", 6, domain=(0.0, 10.0))
    direct.update(t, y)
    attached = ImageStream("legendre", 6, domain=(0.0, 10.0))
    flt = LSIFilter("a*exp(b*t)", "t", p0=[1.0, 0.1], window_size=40,
                    order=4, stream=attached)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    def same(a, b):
        assert a.n == b.n and a.basis == b.basis and a.domain == b.domain
        for name in ("S", "G", "sumsq", "sumy", "wsum"):
            np.testing.assert_allclose(getattr(a, name), getattr(b, name),
                                       rtol=1e-10, atol=1e-10)

    assert flt.stream is attached
    # to rounding: the sums run in another order
    same(attached.image(), direct.image())
    with pytest.warns(RuntimeWarning):
        flt.partial_fit(10.0, float("nan"))
    # a skipped sample never reaches the stream
    same(attached.image(), direct.image())
