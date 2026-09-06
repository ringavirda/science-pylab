import numpy as np
import pytest

from dtfit.image import Grid


def test_uniform_grid_is_recognised_and_regenerated():
    x = np.linspace(0.0, 5.0, 101)
    g = Grid.of(x)
    assert g.kind == "uniform" and g.n == 101
    assert np.allclose(g.positions(), x)
    assert g.to_dict() == {"kind": "uniform", "n": 101, "x0": 0.0, "x1": 5.0}


def test_explicit_grid_keeps_positions():
    x = np.sort(np.random.default_rng(0).uniform(0, 1, 50))
    g = Grid.of(x)
    assert g.kind == "explicit"
    assert np.array_equal(g.positions(), x)
    assert np.array_equal(Grid.from_dict(g.to_dict()).positions(), x)


def test_merge_contiguous_uniform_stays_uniform():
    a = Grid.of(np.linspace(0.0, 1.0, 11))
    b = Grid.of(np.linspace(1.1, 2.0, 10))
    m = a.merge(b)
    assert m.kind == "uniform" and m.n == 21
    assert np.allclose(m.positions(), np.linspace(0.0, 2.0, 21))


def test_merge_non_contiguous_becomes_explicit():
    a = Grid.of(np.linspace(0.0, 1.0, 11))
    b = Grid.of(np.linspace(1.5, 2.0, 6))
    m = a.merge(b)
    assert m.kind == "explicit" and m.n == 17
    assert m.positions()[10] == 1.0 and m.positions()[11] == 1.5


def test_merge_rejects_overlap():
    a = Grid.of(np.linspace(0.0, 1.0, 11))
    with pytest.raises(ValueError):
        a.merge(Grid.of(np.linspace(0.5, 2.0, 16)))


def test_merge_small_scale_uniform_uses_relative_tolerance():
    a = Grid.of(np.linspace(0.0, 1e-8, 11))
    contiguous = Grid.of(np.linspace(1.1e-8, 2.1e-8, 11))
    m = a.merge(contiguous)
    expected = np.concatenate([a.positions(), contiguous.positions()])
    assert m.kind == "uniform"
    assert np.allclose(m.positions(), expected, rtol=1e-15, atol=0.0)

    gapped = Grid.of(np.linspace(2.1e-8, 3.1e-8, 11))
    m2 = a.merge(gapped)
    assert m2.kind == "explicit"
    assert np.array_equal(
        m2.positions(), np.concatenate([a.positions(), gapped.positions()])
    )


def test_positions_of_explicit_grid_is_a_copy():
    x = np.array([0.0, 1.0, 2.0, 5.0])
    g = Grid.of(x)
    p = g.positions()
    p[0] = -99.0
    assert g.positions()[0] == 0.0


def test_equality_and_hash_survive_roundtrip():
    a = Grid.of(np.linspace(0.0, 5.0, 101))
    assert Grid.from_dict(a.to_dict()) == a
    assert hash(a) == hash(Grid.from_dict(a.to_dict()))

    x = np.array([0.0, 1.0, 2.0, 5.0])
    b = Grid.of(x)
    assert Grid.from_dict(b.to_dict()) == b
    assert hash(b) == hash(Grid.from_dict(b.to_dict()))


def test_of_rejects_empty_array():
    with pytest.raises(ValueError):
        Grid.of(np.array([]))
