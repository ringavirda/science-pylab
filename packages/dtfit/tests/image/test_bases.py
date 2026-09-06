import numpy as np
import pytest

from dtfit.image import LegendreBasis, BlockBasis, make_basis, u_of


def test_u_maps_domain_to_unit_interval():
    u = u_of(np.array([2.0, 3.0, 4.0]), 2.0, 4.0)
    assert np.allclose(u, [-1.0, 0.0, 1.0])


def test_u_of_rejects_degenerate_domain():
    with pytest.raises(ValueError):
        u_of(np.array([1.0]), 1.0, 1.0)


def test_legendre_evaluate_shape_and_orthogonality():
    b = LegendreBasis(6)
    u = np.linspace(-1, 1, 20001)
    Phi = b.evaluate(u)
    assert Phi.shape == (20001, 7) and b.n_coef == 7 and b.order == 6
    gram = Phi.T @ Phi * (2.0 / 20001)
    expected = np.diag(2.0 / (2 * np.arange(7) + 1))
    assert np.allclose(gram, expected, atol=2e-3)


def test_block_evaluate_partitions_the_interval():
    b = BlockBasis(4)
    u = np.linspace(-1, 1, 12)
    Phi = b.evaluate(u)
    assert Phi.shape == (12, 4) and b.n_coef == 4
    assert np.array_equal(Phi.sum(axis=1), np.ones(12))
    assert np.array_equal(Phi.sum(axis=0), [3, 3, 3, 3])


def test_make_basis_by_name_and_passthrough():
    assert isinstance(make_basis("legendre", 3), LegendreBasis)
    assert isinstance(make_basis("block", 5), BlockBasis)
    b = LegendreBasis(2)
    assert make_basis(b, None) is b
    with pytest.raises(ValueError):
        make_basis("haar", 3)
    with pytest.raises(ValueError):
        make_basis("legendre", 0)


def test_basis_serialisation_roundtrip():
    b = LegendreBasis(4)
    d = b.to_dict()
    assert make_basis(d["name"], d["order"]).to_dict() == d
