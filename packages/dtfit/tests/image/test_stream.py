import numpy as np
import pytest

from dtfit.image import Image, ImageStream, Original, fit


def _series(n=1000, seed=0):
    x = np.linspace(0.0, 20.0, n)
    y = 3.0 * np.exp(-0.2 * x) + 0.5
    return x, y + 0.02 * np.random.default_rng(seed).standard_normal(n)


def _close(a, b, rtol):
    scale = float(np.max(np.abs(b))) or 1.0
    assert np.allclose(a, b, rtol=rtol, atol=rtol * scale)


def _feed(stream, x, y, cuts, w=None):
    start = 0
    for c in [*cuts, x.size]:
        if w is None:
            stream.update(x[start:c], y[start:c])
        else:
            stream.update(x[start:c], y[start:c], w[start:c])
        start = c
    return stream


def test_uniform_accumulator_equals_batch_image():
    x, y = _series()
    ref = Image.of(Original(x, y, domain=(0.0, 20.0)), "legendre", 12)
    s = _feed(ImageStream("legendre", 12, domain=(0.0, 20.0)), x, y,
              [1, 2, 137, 500, 999])
    got = s.image()
    assert got.n == ref.n and got.grid.kind == "uniform"
    _close(got.S, ref.S, 1e-12)
    _close(got.G, ref.G, 1e-12)
    assert got.sumsq == pytest.approx(ref.sumsq, rel=1e-13)
    a = fit("a*exp(-b*x) + c", got, "x", p0=[3.0, 0.2, 0.5])
    b = fit("a*exp(-b*x) + c", ref, "x", p0=[3.0, 0.2, 0.5])
    assert np.allclose(a.coeffs, b.coeffs, atol=1e-9)


def test_explicit_grid_with_weights_equals_batch_image():
    x, y = _series(400)
    rng = np.random.default_rng(1)
    xr = np.sort(rng.uniform(0.0, 20.0, 400))
    xr[0], xr[-1] = 0.0, 20.0
    w = rng.uniform(0.5, 2.0, 400)
    ref = Image.of(Original(xr, y, w, domain=(0.0, 20.0)), "legendre", 8)
    s = _feed(ImageStream("legendre", 8, domain=(0.0, 20.0),
                          grid="explicit"), xr, y, [50, 51, 300], w=w)
    got = s.image()
    assert got.grid.kind == "explicit" and got.weighted
    _close(got.S, ref.S, 1e-12)
    _close(got.G, ref.G, 1e-12)
    assert np.allclose(got.grid.positions(), ref.grid.positions())


def test_checkpoint_and_resume_reproduce_the_uninterrupted_image():
    x, y = _series()
    whole = _feed(ImageStream("legendre", 10, domain=(0.0, 20.0)), x, y,
                  [300, 700]).image()
    s = ImageStream("legendre", 10, domain=(0.0, 20.0))
    s.update(x[:300], y[:300])
    state = s.checkpoint()
    import json
    state = json.loads(json.dumps(state))
    t = ImageStream("legendre", 10, domain=(0.0, 20.0)).resume(state)
    t.update(x[300:700], y[300:700])
    t.update(x[700:], y[700:])
    got = t.image()
    assert np.array_equal(got.S, whole.S) and np.array_equal(got.G, whole.G)
    assert got.n == whole.n and got.grid == whole.grid


def test_resume_rejects_a_different_configuration():
    s = ImageStream("legendre", 10, domain=(0.0, 20.0))
    x, y = _series(100)
    s.update(x, y)
    with pytest.raises(ValueError, match="configuration"):
        ImageStream("legendre", 8, domain=(0.0, 20.0)).resume(s.checkpoint())


def test_channels_equal_per_channel_images():
    x, _ = _series(500)
    rng = np.random.default_rng(2)
    Y = np.column_stack([
        2.0 * np.exp(-0.3 * x) + rng.standard_normal(500) * 0.01,
        np.sin(0.7 * x) + rng.standard_normal(500) * 0.01,
        0.1 * x,
    ])
    s = ImageStream("legendre", 9, domain=(0.0, 20.0), channels=3)
    s.update(x[:200], Y[:200])
    s.update(x[200:], Y[200:])
    imgs = s.images()
    assert len(imgs) == 3
    for c in range(3):
        ref = Image.of(Original(x, Y[:, c], domain=(0.0, 20.0)),
                       "legendre", 9)
        _close(imgs[c].S, ref.S, 1e-12)
        _close(imgs[c].G, ref.G, 1e-12)
        assert imgs[c].sumsq == pytest.approx(ref.sumsq, rel=1e-13)
    assert np.array_equal(s.image(1).S, imgs[1].S)


def test_merge_of_contiguous_uniform_streams_equals_whole():
    x, y = _series()
    a = ImageStream("legendre", 6, domain=(0.0, 20.0))
    a.update(x[:400], y[:400])
    b = ImageStream("legendre", 6, domain=(0.0, 20.0))
    b.update(x[400:], y[400:])
    ref = Image.of(Original(x, y, domain=(0.0, 20.0)), "legendre", 6)
    got = a.merge(b).image()
    _close(got.S, ref.S, 1e-12)
    assert got.grid == ref.grid
    c = ImageStream("legendre", 6, domain=(0.0, 20.0))
    c.update(x[500:], y[500:])
    with pytest.raises(ValueError, match="contiguous"):
        a.merge(c)


def test_input_errors():
    s = ImageStream("legendre", 6, domain=(0.0, 20.0))
    x, y = _series(50)
    with pytest.raises(ValueError, match="weights"):
        s.update(x, y, np.ones(50))
    with pytest.raises(ValueError, match="uniform"):
        s.update(np.array([0.0, 1.0, 3.0]), np.zeros(3))
    with pytest.raises(ValueError, match="domain"):
        s.update(np.array([25.0]), np.zeros(1))
    with pytest.raises(ValueError, match="finite"):
        s.update(np.array([0.0, 1.0]), np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="no samples"):
        s.image()
    with pytest.raises(ValueError, match="domain"):
        ImageStream("legendre", 6)
    with pytest.raises(NotImplementedError):
        ImageStream("legendre", 6, domain=(0.0, 1.0), block=10)


def test_backend_keyword_accepts_numpy_and_rejects_unknown():
    x, y = _series(100)
    s = ImageStream("legendre", 4, domain=(0.0, 20.0), backend="numpy")
    s.update(x, y)
    assert s.n == 100
    with pytest.raises(ValueError):
        ImageStream("legendre", 4, domain=(0.0, 20.0), backend="nope")
