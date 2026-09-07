import numpy as np
import pytest

from dtfit.image import Image, Original, fit
from dtfit.image.transfer import assemble, block_transfer, legendre_transfer


def _series(n=600):
    x = np.linspace(0.0, 10.0, n)
    y = 5.0 / (1.0 + np.exp(-1.5 * (x - 5.0)))
    y = y + 0.05 * np.random.default_rng(0).standard_normal(n)
    return x, y


def _close(a, b, rtol):
    scale = float(np.max(np.abs(b))) or 1.0
    assert np.allclose(a, b, rtol=rtol, atol=rtol * scale)


def test_legendre_transfer_maps_coarse_basis_onto_local():
    from dtfit.image import LegendreBasis, u_of

    A = legendre_transfer((2.0, 5.0), (0.0, 10.0), 12, 8)
    assert A.shape == (13, 9)
    x = np.linspace(2.0, 5.0, 40)
    phi_l = LegendreBasis(12).evaluate(u_of(x, 2.0, 5.0))
    phi_c = LegendreBasis(8).evaluate(u_of(x, 0.0, 10.0))
    _close(phi_l @ A, phi_c, 1e-12)


def test_legendre_assembly_matches_whole_interval_image():
    x, y = _series()
    blocks = []
    for k in range(3):
        sl = slice(200 * k, 200 * (k + 1))
        blocks.append(Image.of(Original(x[sl], y[sl]), "legendre", 12))
    whole = Image.of(Original(x, y, domain=(x[0], x[-1])), "legendre", 8)
    got = assemble(blocks, order=8)
    assert got.domain == whole.domain and got.n == whole.n
    _close(got.S, whole.S, 1e-10)
    _close(got.G, whole.G, 1e-10)
    assert got.sumsq == pytest.approx(whole.sumsq, rel=1e-12)
    a = fit("L/(1 + exp(-k*(x - x0)))", got, "x", p0=[5.0, 1.5, 5.0])
    b = fit("L/(1 + exp(-k*(x - x0)))", whole, "x", p0=[5.0, 1.5, 5.0])
    assert np.allclose(a.coeffs, b.coeffs, atol=1e-8)


def test_transfer_to_same_domain_and_order_is_identity():
    x, y = _series(200)
    img = Image.of(Original(x, y), "legendre", 10)
    back = img.transfer(img.domain, 10)
    _close(back.S, img.S, 1e-13)
    _close(back.G, img.G, 1e-13)
    assert back.grid == img.grid


def test_transfer_rejects_coarse_order_above_local_and_domain_outside():
    x, y = _series(200)
    img = Image.of(Original(x, y), "legendre", 6)
    with pytest.raises(ValueError, match="order"):
        img.transfer((0.0, 20.0), 8)
    with pytest.raises(ValueError, match="domain"):
        img.transfer((1.0, 3.0), 4)


def test_block_assembly_aggregates_windows():
    x, y = _series(600)
    edges = np.linspace(0.0, 10.0, 4)
    blocks = []
    for k in range(3):
        sel = (x >= edges[k]) & (x <= edges[k + 1]) if k == 2 else (
            (x >= edges[k]) & (x < edges[k + 1]))
        blocks.append(Image.of(
            Original(x[sel], y[sel], domain=(edges[k], edges[k + 1])),
            "block", 4))
    whole = Image.of(Original(x, y, domain=(0.0, 10.0)), "block", 6)
    got = assemble(blocks, domain=(0.0, 10.0), order=6)
    _close(got.S, whole.S, 1e-12)
    _close(got.G, whole.G, 1e-12)
    with pytest.raises(ValueError, match="union of fine"):
        assemble(blocks, domain=(0.0, 10.0), order=5)


def test_block_transfer_matrix_is_zero_one_aggregation():
    A = block_transfer((0.0, 2.0), (0.0, 6.0), 4, 3)
    assert A.shape == (4, 3)
    assert np.array_equal(A.sum(axis=1), np.ones(4))
    assert np.array_equal(A[:, 0], np.ones(4))


def test_assemble_rejects_mixed_bases_and_empty():
    x, y = _series(200)
    a = Image.of(Original(x, y), "legendre", 6)
    b = Image.of(Original(x, y), "block", 6)
    with pytest.raises(ValueError, match="basis"):
        assemble([a, b])
    with pytest.raises(ValueError, match="no images"):
        assemble([])
