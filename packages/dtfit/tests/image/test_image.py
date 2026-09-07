import numpy as np
import pytest

from dtfit.image import Original, Image, LegendreBasis, u_of
from dtfit.image.image import huber_weights


def _orig(n=200, seed=0, weighted=False):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + 0.05 * rng.standard_normal(n)
    return Original(x, y, sigma=(1.0 + x) if weighted else None)


def test_projections_and_gram_match_direct_formulas():
    o = _orig(weighted=True)
    img = Image.of(o, "legendre", 6)
    Phi = LegendreBasis(6).evaluate(u_of(o.x, *o.domain))
    assert np.allclose(img.S, Phi.T @ (o.w * o.y))
    assert np.allclose(img.G, Phi.T @ (o.w[:, None] * Phi))
    assert img.n == 200 and np.isclose(img.sumsq, np.sum(o.w * o.y ** 2))
    assert np.isclose(img.sumy, np.sum(o.w * o.y)) and np.isclose(img.wsum, o.w.sum())
    assert img.order == 6 and img.n_coef == 7 and img.weighted


def test_unweighted_image_stores_no_weights():
    img = Image.of(_orig(), "legendre", 6)
    assert img.w is None and not img.weighted


def test_additivity_under_merge():
    o = _orig(n=301)
    whole = Image.of(o, "legendre", 12)
    a = Image.of(Original(o.x[:150], o.y[:150], domain=o.domain), "legendre", 12)
    b = Image.of(Original(o.x[150:], o.y[150:], domain=o.domain), "legendre", 12)
    m = a.merge(b)
    assert np.allclose(m.S, whole.S, atol=1e-12) and np.allclose(m.G, whole.G, atol=1e-10)
    assert m.n == whole.n and np.isclose(m.sumsq, whole.sumsq)
    assert m.grid.kind == "uniform" and m.grid.n == 301


def test_merge_requires_same_basis_and_domain():
    o = _orig()
    a = Image.of(o, "legendre", 6)
    with pytest.raises(ValueError):
        a.merge(Image.of(o, "legendre", 7))
    with pytest.raises(ValueError):
        a.merge(Image.of(Original(o.x, o.y, domain=(0.0, 5.0)), "legendre", 6))


def test_nesting_is_exact():
    o = _orig()
    direct_S = Image.of(o, "legendre", 12).S
    direct_G = Image.of(o, "legendre", 12).G
    assert np.allclose(Image.of(o, "legendre", 48).truncate(12).S, direct_S, rtol=1e-13, atol=1e-13 * np.max(np.abs(direct_S)))
    assert np.allclose(Image.of(o, "legendre", 48).truncate(12).G, direct_G, rtol=1e-13, atol=1e-13 * np.max(np.abs(direct_G)))
    with pytest.raises(ValueError):
        Image.of(o, "block", 8).truncate(4)


def test_beta_reconstruct_and_serialisation():
    o = _orig()
    img = Image.of(o, "legendre", 8)
    beta = np.linalg.solve(img.G, img.S)
    assert np.allclose(img.beta, beta)
    Phi = LegendreBasis(8).evaluate(u_of(o.x, *o.domain))
    assert np.allclose(img.reconstruct(o.x), Phi @ beta)
    back = Image.from_dict(img.to_dict())
    assert np.array_equal(back.S, img.S) and np.array_equal(back.G, img.G)
    assert back.basis == img.basis and back.grid.to_dict() == img.grid.to_dict()


def test_simulate_shape_and_seed():
    img = Image.of(_orig(), "legendre", 8)
    x1, y1 = img.simulate(0.05, rng=np.random.default_rng(3))
    x2, y2 = img.simulate(0.05, rng=np.random.default_rng(3))
    assert x1.shape == (200,) and np.array_equal(y1, y2)
    assert np.std(y1 - img.reconstruct(x1)) == pytest.approx(0.05, rel=0.3)


def test_of_model_equals_image_of_sampled_model():
    o = _orig()
    img = Image.of_model("a*exp(-b*x) + c", [2.0, 0.7, 0.3], o.grid, "legendre", 6, var="x", domain=o.domain)
    x = o.grid.positions()
    direct = Image.of(Original(x, 2.0 * np.exp(-0.7 * x) + 0.3, domain=o.domain), "legendre", 6)
    assert np.allclose(img.S, direct.S) and np.allclose(img.G, direct.G)


def test_robust_weights_downweight_outliers_only():
    o = _orig()
    y = o.y.copy()
    y[[20, 100, 150]] += 5.0
    Phi = LegendreBasis(8).evaluate(u_of(o.x, *o.domain))
    w = huber_weights(Phi, y, np.ones(200))
    assert np.all(w[[20, 100, 150]] < 0.2)
    clean = np.delete(w, [20, 100, 150])
    assert np.mean(clean > 0.9) > 0.8
    assert np.median(clean) == 1.0
    img = Image.of(Original(o.x, y), "legendre", 8, robust=True)
    assert img.robust and img.w is not None and img.wsum < 200.0


def test_block_image_sums_windows():
    o = _orig(n=40)
    img = Image.of(o, "block", 4)
    assert np.allclose(img.S, o.y.reshape(4, 10).sum(axis=1))
    assert np.allclose(np.diag(img.G), [10, 10, 10, 10]) and np.allclose(img.G - np.diag(np.diag(img.G)), 0)
