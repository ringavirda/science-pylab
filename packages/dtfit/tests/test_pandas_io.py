"""Optional-pandas interop: Series and single-column DataFrame inputs,
pandas-out predictions, and the ``dtfit._pandas`` helpers.

The module-level ``importorskip`` keeps a pandas-free environment green. The
ndarray paths are covered elsewhere.
"""

import numpy as np
import pytest

from dtfit import fit_lsi, fit_eac
from dtfit.stochastic import fit_stochastic
from dtfit.sklearn import NonlineRegressor
from dtfit._pandas import (
    HAS_PANDAS,
    as_series,
    capture_index,
    extend_index,
    is_dataframe,
    is_series,
    to_1d_array,
)

pd = pytest.importorskip("pandas")


@pytest.fixture
def exp_xy():
    """Exponential decay with known parameters."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 2.0, 80)
    y = 2.5 * np.exp(-1.2 * x) + rng.normal(0.0, 0.02, x.size)
    return x, y


def test_has_pandas_flag_true_when_installed():
    # the module-level importorskip already ran; the flag can only be True here
    assert HAS_PANDAS is True
    s = pd.Series([1.0, 2.0, 3.0])
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert is_series(s) and not is_series(df) and not is_series(np.arange(3))
    assert is_dataframe(df) and not is_dataframe(s) and not is_dataframe([1, 2])


def test_to_1d_array_series_and_single_col_dataframe():
    s = pd.Series([1.0, 2.0, 3.0])
    df1 = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
    np.testing.assert_array_equal(to_1d_array(s), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(to_1d_array(df1), np.array([1.0, 2.0, 3.0]))
    assert to_1d_array(s).dtype == float
    assert to_1d_array(df1).ndim == 1


def test_to_1d_array_multicol_dataframe_raises():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match=r"must be 1-D.*2 columns"):
        to_1d_array(df, "data_x")


def test_to_1d_array_ndarray_bit_identical():
    x = np.linspace(0.0, 1.0, 17)
    out = to_1d_array(x)
    np.testing.assert_array_equal(out, x)
    np.testing.assert_array_equal(to_1d_array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))


def test_capture_index_and_as_series_roundtrip():
    idx = pd.Index(["p", "q", "r"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    got = capture_index(s)
    assert got is not None and list(got) == ["p", "q", "r"]
    assert capture_index(np.arange(3)) is None
    out = as_series(np.array([4.0, 5.0, 6.0]), got)
    assert isinstance(out, pd.Series) and list(out.index) == ["p", "q", "r"]
    # a None index means a non-pandas caller: plain ndarray back
    plain = as_series(np.array([1.0, 2.0]), None)
    assert isinstance(plain, np.ndarray)


def test_extend_index_datetime_daily():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    fut = extend_index(idx, 3)
    assert fut is not None
    assert list(fut) == list(pd.date_range("2024-01-11", periods=3, freq="D"))


def test_extend_index_range():
    idx = pd.RangeIndex(start=0, stop=10, step=2)  # 0,2,4,6,8
    fut = extend_index(idx, 3)
    assert list(fut) == [10, 12, 14]


def test_extend_index_integer_index_constant_step():
    idx = pd.Index([100, 110, 120])
    fut = extend_index(idx, 2)
    assert list(fut) == [130, 140]


def test_extend_index_non_inferable_freq_returns_none():
    # irregular datetime spacing: pd.infer_freq has nothing to infer
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-08"])
    assert extend_index(idx, 2) is None
    # a string index is not continuable at all
    assert extend_index(pd.Index(["a", "b", "c"]), 2) is None
    # degenerate arguments: no horizon, no index, empty index
    assert extend_index(pd.RangeIndex(0, 5), 0) is None
    assert extend_index(None, 3) is None
    assert extend_index(pd.RangeIndex(0, 0), 3) is None


@pytest.mark.parametrize("fit", [fit_lsi, fit_eac])
def test_fitters_accept_series_matches_ndarray(fit, exp_xy):
    x, y = exp_xy
    r_arr = fit(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_ser = fit(pd.Series(x), pd.Series(y), "a*exp(b*x)", "x", p0=[1.0, -0.5])
    np.testing.assert_array_equal(r_ser.coeffs, r_arr.coeffs)


@pytest.mark.parametrize("fit", [fit_lsi, fit_eac])
def test_fitters_accept_single_col_dataframe(fit, exp_xy):
    x, y = exp_xy
    r_arr = fit(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    r_df = fit(
        pd.DataFrame({"t": x}), pd.DataFrame({"v": y}),
        "a*exp(b*x)", "x", p0=[1.0, -0.5],
    )
    np.testing.assert_array_equal(r_df.coeffs, r_arr.coeffs)


@pytest.mark.parametrize("fit", [fit_lsi, fit_eac])
def test_fitters_reject_multicol_dataframe(fit, exp_xy):
    x, y = exp_xy
    bad = pd.DataFrame({"a": x, "b": x})
    with pytest.raises(ValueError, match=r"must be 1-D.*2 columns"):
        fit(bad, pd.Series(y), "a*exp(b*x)", "x", p0=[1.0, -0.5])


def test_predict_series_returns_aligned_series(exp_xy):
    x, y = exp_xy
    r = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    idx = pd.date_range("2024-01-01", periods=x.size, freq="D")
    xs = pd.Series(x, index=idx)

    y_arr = r.predict(x)
    y_ser = r.predict(xs)
    assert isinstance(y_arr, np.ndarray)
    assert isinstance(y_ser, pd.Series)
    assert list(y_ser.index) == list(idx)
    np.testing.assert_allclose(y_ser.to_numpy(), y_arr)


def test_predict_series_return_std_pair(exp_xy):
    x, y = exp_xy
    r = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    idx = pd.RangeIndex(start=5, stop=5 + x.size)
    xs = pd.Series(x, index=idx)

    y_arr, std_arr = r.predict(x, return_std=True)
    y_ser, std_ser = r.predict(xs, return_std=True)
    assert isinstance(y_ser, pd.Series) and isinstance(std_ser, pd.Series)
    assert list(y_ser.index) == list(idx) and list(std_ser.index) == list(idx)
    np.testing.assert_allclose(y_ser.to_numpy(), y_arr)
    np.testing.assert_allclose(std_ser.to_numpy(), std_arr)


def test_predict_single_col_dataframe_returns_series(exp_xy):
    x, y = exp_xy
    r = fit_eac(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    df = pd.DataFrame({"t": x})
    out = r.predict(df)
    assert isinstance(out, pd.Series)
    np.testing.assert_allclose(out.to_numpy(), r.predict(x))


def test_predict_ndarray_still_ndarray(exp_xy):
    x, y = exp_xy
    r = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1.0, -0.5])
    assert isinstance(r.predict(x), np.ndarray)
    yv, sv = r.predict(x, return_std=True)
    assert isinstance(yv, np.ndarray) and isinstance(sv, np.ndarray)


def test_extend_index_datetime_inferred_freq():
    # regularly spaced but freq-less. index.freq is None here, leaving
    # pd.infer_freq as the only source of the frequency; drop that call and
    # nothing else in the suite notices.
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
    assert idx.freq is None  # forces the right operand of `index.freq or ...`
    fut = extend_index(idx, 2)
    assert list(fut) == list(pd.date_range("2024-01-04", periods=2, freq="D"))


def test_multivariate_x_raises_clear_error_everywhere():
    import dtfit as dt
    rng = np.random.default_rng(0)
    X2 = rng.normal(size=(40, 2))          # genuinely multivariate
    y = rng.normal(size=40)
    entries = [
        lambda: dt.fit_lsi(X2, y, "a*x", "x"),
        lambda: dt.fit_eac(X2, y, "a*x", "x"),
        lambda: dt.fit("a*x", dt.Original(X2, y), "x"),
        lambda: dt.auto_forecast(X2, y, horizon=3),
        lambda: fit_stochastic(X2),
        lambda: NonlineRegressor("a*x", "x").fit(X2, y),
        lambda: dt.fit_lsi(pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0)}),
                           y, "a*x", "x"),
    ]
    for entry in entries:
        with pytest.raises(ValueError) as exc:
            entry()
        msg = str(exc.value)
        # each message must state the 1-D scope and name the escape hatch
        assert "one-dimensional" in msg or "single input feature" in msg
        assert "compose 1-D models" in msg


def test_column_vector_is_accepted_not_flagged_multivariate():
    # an (n, 1) array or DataFrame is a column vector, not multivariate:
    # to_1d_array squeezes it, and only a second column is rejected.
    x = np.linspace(0.0, 2.0, 60)
    col = to_1d_array(x.reshape(-1, 1), "x")
    assert col.shape == (60,)
    np.testing.assert_array_equal(col, x)
    col_df = to_1d_array(pd.DataFrame({"t": x}), "x")
    assert col_df.shape == (60,)
