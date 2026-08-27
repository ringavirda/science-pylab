"""Shared p0/bounds normalizers: ``dtfit.methods.normalize_p0`` and
``normalize_bounds`` are the one place every batch fitter's input forms get
canonicalized."""

import numpy as np
import pytest

from dtfit.methods import normalize_bounds, normalize_p0

NAMES = ["a", "b", "w"]


# normalize_p0
def test_p0_none_passes_through():
    assert normalize_p0(None, NAMES) is None


def test_p0_positional_sequence():
    out = normalize_p0([1.0, 2.0, 3.0], NAMES)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
    assert out is not None and out.dtype == float


def test_p0_dict_reordered_to_sorted_names():
    out = normalize_p0({"w": 3.0, "a": 1.0, "b": 2.0}, NAMES)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_p0_wrong_length_raises():
    with pytest.raises(ValueError, match="length 3"):
        normalize_p0([1.0, 2.0], NAMES)


def test_p0_dict_missing_name_lists_valid_names():
    with pytest.raises(ValueError, match=r"\['a', 'b', 'w'\]"):
        normalize_p0({"a": 1.0, "b": 2.0}, NAMES)


def test_p0_dict_unknown_name_raises():
    with pytest.raises(ValueError, match=r"unknown \['z'\]"):
        normalize_p0({"a": 1.0, "b": 2.0, "w": 3.0, "z": 4.0}, NAMES)


# normalize_bounds
def test_bounds_none_passes_through():
    assert normalize_bounds(None, NAMES) is None


def test_bounds_pair_list():
    out = normalize_bounds([(0, 1), (-1, 1), (2, 5)], NAMES)
    assert out == [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]


def test_bounds_partial_dict_fills_infinite():
    out = normalize_bounds({"b": (0.0, 2.0)}, NAMES)
    assert out is not None
    assert out[0] == (-np.inf, np.inf)
    assert out[1] == (0.0, 2.0)
    assert out[2] == (-np.inf, np.inf)


def test_bounds_dict_unknown_name_raises():
    with pytest.raises(ValueError, match=r"unknown parameters \['z'\]"):
        normalize_bounds({"z": (0, 1)}, NAMES)


def test_bounds_scipy_tuple_arrays():
    out = normalize_bounds(([0, -1, 2], [1, 1, 5]), NAMES)
    assert out == [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)]


def test_bounds_scipy_tuple_scalars_broadcast():
    out = normalize_bounds((0, 10), NAMES)
    assert out == [(0.0, 10.0)] * 3


def test_bounds_two_param_ambiguity_reads_as_pairs():
    # The documented tie-break: at n == 2 a 2-tuple of two 2-sequences reads as
    # per-parameter (lo, hi) pairs, not as scipy's (lo_array, hi_array).
    out = normalize_bounds(([0, 1], [2, 3]), ["a", "b"])
    assert out == [(0.0, 1.0), (2.0, 3.0)]


def test_bounds_scalar_scipy_form_for_two_params():
    # Scalars are not pairs, so the scipy reading applies even for n == 2.
    out = normalize_bounds((0, [2, 3]), ["a", "b"])
    assert out == [(0.0, 2.0), (0.0, 3.0)]


def test_bounds_lo_above_hi_names_parameter():
    with pytest.raises(ValueError, match="parameter 'b'"):
        normalize_bounds([(0, 1), (3, 2), (0, 1)], NAMES)
    with pytest.raises(ValueError, match="parameter 'w'"):
        normalize_bounds({"w": (5, 4)}, NAMES)


def test_bounds_wrong_count_raises():
    with pytest.raises(ValueError, match="bounds"):
        normalize_bounds([(0, 1)], NAMES)
    with pytest.raises(ValueError, match="length-3"):
        normalize_bounds(([0, 1, 2], [3, 4, 5], [6, 7, 8]), ["a", "b"])


def test_bounds_lo_equals_hi_rejected_with_param_name():
    """A degenerate lo == hi box is rejected up front, with the parameter
    named. Left through, it crashes scipy's trf with an opaque message on some
    solver paths while DE tolerates it on others."""
    with pytest.raises(ValueError, match=r"'a'.*strictly less"):
        normalize_bounds([(2.0, 2.0), (0.0, 10.0)], ["a", "b"])
    with pytest.raises(ValueError, match=r"'b'.*strictly less"):
        normalize_bounds({"b": (1.0, 1.0)}, ["a", "b"])
