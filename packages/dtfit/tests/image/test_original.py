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
    with pytest.raises(ValueError):
        Original([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        Original(np.ones((3, 2)), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        Original([1.0, float("nan")], [1.0, 2.0])
    with pytest.raises(ValueError):
        Original([1.0, 2.0], [1.0, 2.0], sigma=[1.0, 0.0])
    with pytest.raises(ValueError):
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
