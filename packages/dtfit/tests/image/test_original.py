import numpy as np
import pytest

from dtfit.image import Original


def test_original_validates_and_sorts():
    o = Original([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
    assert np.array_equal(o.x, [1.0, 2.0, 3.0])
    assert np.array_equal(o.y, [10.0, 20.0, 30.0])
    assert o.n == 3 and o.domain == (1.0, 3.0)
    assert np.array_equal(o.w, [1.0, 1.0, 1.0])
    assert o.weighted is False


def test_original_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        Original([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="1-D"):
        Original(np.ones((3, 2)), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        Original([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        Original([1.0, 2.0], [1.0, 2.0], sigma=[1.0, 0.0])
    with pytest.raises(ValueError, match="nan_policy"):
        Original([1.0, 2.0], [1.0, 2.0], nan_policy="drop")


def test_nan_policy_omit_drops_pairs():
    o = Original(
        [1.0, 2.0, 3.0], [1.0, float("nan"), 3.0], nan_policy="omit"
    )
    assert o.n == 2 and np.array_equal(o.x, [1.0, 3.0])


def test_sigma_becomes_inverse_variance_weight():
    o = Original([0.0, 1.0], [0.0, 1.0], sigma=[1.0, 2.0])
    assert np.allclose(o.w, [1.0, 0.25]) and o.weighted is True


def test_window_and_domain():
    x = np.linspace(0, 10, 11)
    o = Original(x, x**2, domain=(0.0, 12.0))
    assert o.domain == (0.0, 12.0)
    w = o.window(2, 5)
    assert w.n == 3
    assert np.array_equal(w.x, [2.0, 3.0, 4.0])
    assert w.domain == (2.0, 4.0)


def test_ties_are_allowed_but_span_is_required():
    o = Original([1.0, 1.0, 2.0], [5.0, 6.0, 7.0])
    assert o.n == 3 and o.domain == (1.0, 2.0)
    with pytest.raises(ValueError):
        Original([2.0, 2.0], [1.0, 3.0])


def test_scalar_sigma_broadcasts():
    o = Original([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], sigma=2.0)
    assert np.allclose(o.w, [0.25, 0.25, 0.25])


def test_explicit_weight_follows_the_sort():
    o = Original([3.0, 1.0, 2.0], [30.0, 10.0, 20.0], w=[1.0, 2.0, 3.0])
    assert np.array_equal(o.w, [2.0, 3.0, 1.0])


def test_w_and_sigma_together_rejected():
    with pytest.raises(ValueError):
        Original([1.0, 2.0], [1.0, 2.0], w=[1.0, 1.0], sigma=[1.0, 1.0])


def test_infinite_weight_rejected():
    with pytest.raises(ValueError):
        Original([1.0, 2.0], [1.0, 2.0], w=[1.0, float("inf")])


def test_original_owns_its_data():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    w_in = np.array([1.0, 2.0, 3.0])
    o = Original(x, y, w=w_in)
    x[0] = -1.0
    y[0] = -1.0
    w_in[0] = -1.0
    assert o.x[0] == 1.0 and o.y[0] == 10.0 and o.w[0] == 1.0

    w = o.window(0, 2)
    w.y[0] = -1.0
    assert o.y[0] == 10.0


def test_pandas_input_is_coerced():
    pd = pytest.importorskip("pandas")
    s = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3))
    o = Original(s.index.to_numpy(dtype=float), s)
    assert np.array_equal(o.y, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Original(
            [0.0, 1.0, 2.0],
            pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}),
        )


def test_diagnostics_reports_white_residuals_for_the_right_model():
    x = np.linspace(0.0, 4.0, 400)
    rng = np.random.default_rng(0)
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + 0.05 * rng.standard_normal(x.size)
    d = Original(x, y).diagnostics("a*exp(-b*t) + c", [2.0, 0.7, 0.3], "t")
    assert set(d) == {"residuals", "durbin_watson", "lag1_autocorr",
                      "normality_p", "mean", "std"}
    assert d["residuals"].shape == (400,)
    assert abs(d["durbin_watson"] - 2.0) < 0.25
    assert abs(d["lag1_autocorr"]) < 0.15
    assert d["normality_p"] > 0.01
    assert d["std"] == pytest.approx(0.05, rel=0.2)


def test_diagnostics_flags_a_wrong_model():
    x = np.linspace(0.0, 4.0, 400)
    rng = np.random.default_rng(0)
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + 0.05 * rng.standard_normal(x.size)
    d = Original(x, y).diagnostics("a + b*t", [2.0, -0.4], "t")
    assert d["durbin_watson"] < 0.5 and d["lag1_autocorr"] > 0.9


def test_diagnostics_matches_residual_diagnostics_on_a_fit():
    from dtfit.diagnostics import residual_diagnostics
    from dtfit.image import fit

    x = np.linspace(0.0, 4.0, 300)
    rng = np.random.default_rng(1)
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + 0.05 * rng.standard_normal(x.size)
    o = Original(x, y)
    res = fit("a*exp(-b*t) + c", o, "t", order=10, p0=[2.0, 0.7, 0.3])
    mine = o.diagnostics("a*exp(-b*t) + c", res.coeffs, "t")
    theirs = residual_diagnostics(res, x, y)
    np.testing.assert_allclose(
        mine["residuals"], theirs["residuals"], rtol=1e-12, atol=1e-15
    )
    for key in ("durbin_watson", "lag1_autocorr", "normality_p", "std"):
        assert mine[key] == pytest.approx(theirs[key], rel=1e-12)


def test_diagnostics_takes_a_callable_model():
    def model(t, a, b):
        return a * np.exp(-b * t)

    x = np.linspace(0.0, 4.0, 200)
    rng = np.random.default_rng(2)
    o = Original(x, 2.0 * np.exp(-0.7 * x) + 0.05 * rng.standard_normal(200))
    d = o.diagnostics(model, [2.0, 0.7])
    assert abs(d["mean"]) < 0.02 and d["std"] == pytest.approx(0.05, rel=0.2)
    assert abs(d["durbin_watson"] - 2.0) < 0.25
