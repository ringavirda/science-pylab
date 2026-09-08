import numpy as np
import pytest

from dtfit.image import Original, Image, LegendreBasis, u_of
from dtfit.image.image import gram_whitener, huber_weights


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
    assert np.isclose(img.sumy, np.sum(o.w * o.y))
    assert np.isclose(img.wsum, o.w.sum())
    assert img.order == 6 and img.n_coef == 7 and img.weighted


def test_unweighted_image_stores_no_weights():
    img = Image.of(_orig(), "legendre", 6)
    assert img.w is None and not img.weighted


def test_additivity_under_merge():
    o = _orig(n=301)
    whole = Image.of(o, "legendre", 12)
    a = Image.of(
        Original(o.x[:150], o.y[:150], domain=o.domain), "legendre", 12
    )
    b = Image.of(
        Original(o.x[150:], o.y[150:], domain=o.domain), "legendre", 12
    )
    m = a.merge(b)
    assert np.allclose(m.S, whole.S, atol=1e-12)
    assert np.allclose(m.G, whole.G, atol=1e-10)
    assert m.n == whole.n and np.isclose(m.sumsq, whole.sumsq)
    assert m.grid.kind == "uniform" and m.grid.n == 301


def test_merge_of_replicate_sample_sets_matches_direct_image():
    rng = np.random.default_rng(0)
    domain = (0.0, 4.0)

    def f(x):
        return 2.0 * np.exp(-0.7 * x) + 0.3

    x1 = np.sort(rng.uniform(*domain, 90))
    x2 = np.sort(rng.uniform(*domain, 70))
    y1 = f(x1) + 0.01 * rng.standard_normal(x1.size)
    y2 = f(x2) + 0.01 * rng.standard_normal(x2.size)
    a = Image.of(Original(x1, y1, domain=domain), "legendre", 8)
    b = Image.of(Original(x2, y2, domain=domain), "legendre", 8)
    merged = a.merge(b)
    whole = Image.of(
        Original(
            np.concatenate([x1, x2]), np.concatenate([y1, y2]),
            domain=domain,
        ),
        "legendre", 8,
    )
    assert np.allclose(merged.S, whole.S)
    assert np.allclose(merged.G, whole.G)
    assert merged.n == whole.n


def test_merge_of_interleaved_chunks_matches_whole_series():
    o = _orig(n=200)
    even = Image.of(
        Original(o.x[::2], o.y[::2], domain=o.domain), "legendre", 8
    )
    odd = Image.of(
        Original(o.x[1::2], o.y[1::2], domain=o.domain), "legendre", 8
    )
    whole = Image.of(o, "legendre", 8)
    m = even.merge(odd)
    assert np.allclose(m.S, whole.S)
    assert np.allclose(m.G, whole.G)
    assert m.n == whole.n


def test_merge_of_uniform_and_explicit_grid_positions():
    a = Image.of(
        Original(np.linspace(0.0, 1.0, 11), np.zeros(11)), "legendre", 2
    )
    xb = np.array([0.05, 0.3, 0.5, 0.95])
    b = Image.of(
        Original(xb, np.zeros(4), domain=(0.0, 1.0)), "legendre", 2
    )
    m = a.merge(b)
    assert m.grid.kind == "explicit"
    expected = np.sort(
        np.concatenate([a.grid.positions(), xb]), kind="stable"
    )
    assert np.array_equal(m.grid.positions(), expected)


def test_merge_requires_same_basis_and_domain():
    o = _orig()
    a = Image.of(o, "legendre", 6)
    with pytest.raises(ValueError):
        a.merge(Image.of(o, "legendre", 7))
    with pytest.raises(ValueError):
        a.merge(Image.of(Original(o.x, o.y, domain=(0.0, 5.0)), "legendre", 6))


def test_nesting_matches_direct_image():
    o = _orig()
    direct = Image.of(o, "legendre", 12)
    truncated = Image.of(o, "legendre", 48).truncate(12)
    assert np.allclose(
        truncated.S, direct.S,
        rtol=1e-13, atol=1e-13 * np.max(np.abs(direct.S)),
    )
    assert np.allclose(
        truncated.G, direct.G,
        rtol=1e-13, atol=1e-13 * np.max(np.abs(direct.G)),
    )
    with pytest.raises(ValueError):
        Image.of(o, "block", 8).truncate(4)


def test_beta_reconstruct_and_serialisation():
    o = _orig()
    img = Image.of(o, "legendre", 8)
    Phi = LegendreBasis(8).evaluate(u_of(o.x, *o.domain))
    ref = np.linalg.lstsq(Phi, o.y, rcond=None)[0]
    assert np.allclose(img.beta, ref, atol=1e-10)
    assert np.allclose(img.reconstruct(o.x), Phi @ img.beta)
    back = Image.from_dict(img.to_dict())
    assert np.array_equal(back.S, img.S) and np.array_equal(back.G, img.G)
    assert back.basis == img.basis
    assert back.grid.to_dict() == img.grid.to_dict()


def test_simulate_shape_and_seed():
    img = Image.of(_orig(), "legendre", 8)
    x1, y1 = img.simulate(sigma=0.05, rng=np.random.default_rng(3))
    x2, y2 = img.simulate(sigma=0.05, rng=np.random.default_rng(3))
    assert x1.shape == (200,) and x2.shape == y2.shape
    assert np.array_equal(x1, img.grid.positions())
    assert np.array_equal(y1, y2)
    assert np.std(y1 - img.reconstruct(x1)) == pytest.approx(0.05, rel=0.3)


def test_simulate_on_a_fresh_grid():
    img = Image.of(_orig(), "legendre", 8)
    x, y = img.simulate(50, sigma=0.0)
    assert x.shape == (50,) and y.shape == (50,)
    assert x[0] == img.domain[0] and x[-1] == img.domain[1]
    assert np.allclose(np.diff(x), x[1] - x[0])
    assert np.array_equal(y, img.reconstruct(x))


def test_simulate_defaults_sigma_to_the_noise_level():
    o = _orig(n=600)
    img = Image.of(o, "legendre", 40)
    level = img.noise_sigma()
    assert level is not None
    _, y = img.simulate(rng=np.random.default_rng(0))
    spread = float(np.std(y - img.reconstruct(img.grid.positions())))
    assert spread == pytest.approx(level, rel=0.2)


def test_simulate_rejects_a_float_sample_count():
    img = Image.of(_orig(), "legendre", 8)
    with pytest.raises(TypeError, match="sample count"):
        img.simulate(0.05)
    with pytest.raises(ValueError, match="n >= 2"):
        img.simulate(1)
    with pytest.raises(ValueError, match="non-negative"):
        img.simulate(sigma=-1.0)


def test_simulate_without_a_readable_noise_level():
    img = Image.of(_orig(), "legendre", 6)
    with pytest.raises(ValueError, match="tail orders"):
        img.simulate()
    block = Image.of(_orig(), "block", 8)
    with pytest.raises(ValueError, match="tail orders"):
        block.simulate()


def test_of_model_equals_image_of_sampled_model():
    o = _orig()
    img = Image.of_model(
        "a*exp(-b*x) + c", [2.0, 0.7, 0.3], o.grid, "legendre", 6,
        var="x", domain=o.domain,
    )
    x = o.grid.positions()
    direct = Image.of(
        Original(x, 2.0 * np.exp(-0.7 * x) + 0.3, domain=o.domain),
        "legendre", 6,
    )
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
    assert np.allclose(np.diag(img.G), [10, 10, 10, 10])
    assert np.allclose(img.G - np.diag(np.diag(img.G)), 0)


def test_of_requires_an_order():
    o = _orig()
    with pytest.raises(ValueError):
        Image.of(o, "legendre", None)


def test_of_rejects_too_few_samples():
    o = Original([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        Image.of(o, "legendre", 6)


def test_roundtrip_equality_and_hash():
    rng = np.random.default_rng(1)
    x = np.array([0.0, 0.3, 1.0, 1.4, 2.5, 3.1, 3.9])
    y = 2.0 * np.exp(-0.7 * x) + 0.3 + 0.02 * rng.standard_normal(x.size)
    w = rng.uniform(0.5, 2.0, x.size)
    o = Original(x, y, w=w, domain=(0.0, 4.0))
    img = Image.of(o, "legendre", 3, robust=True)
    back = Image.from_dict(img.to_dict())
    assert back == img
    assert hash(back) == hash(img)


def test_gram_whitener_factors_a_gram_singular_to_rounding():
    # samples on the last tenth of the domain at order 40 leave the
    # Legendre Gram singular far below the 1e-14 jitter; the whitener
    # still returns a factor, and its square is G up to that jitter
    x = np.linspace(0.9, 1.0, 400)
    img = Image.of(Original(x, np.cos(3.0 * x), domain=(0.0, 1.0)),
                   "legendre", 40)
    s = np.linalg.svd(img.G, compute_uv=False)
    assert s[0] / max(s[-1], 1e-300) > 1e16
    L = gram_whitener(img.G)
    scale = 1e-8 * np.trace(img.G) / img.G.shape[0]
    assert np.tril(L).shape == img.G.shape
    assert np.abs(L @ L.T - img.G).max() <= scale * 1.01


def test_fit_from_a_numerically_singular_image_returns():
    from dtfit.image import fit

    x = np.linspace(0.9, 1.0, 400)
    y = 2.0 * np.exp(-1.5 * x)
    img = Image.of(Original(x, y, domain=(0.0, 1.0)), "legendre", 40)
    res = fit("a * exp(-b * x)", img, "x", p0=[1.0, 1.0])
    assert np.isfinite(res.params["a"]) and np.isfinite(res.params["b"])
