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
    assert np.allclose(a.coeffs, b.coeffs, rtol=0, atol=1e-9)


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


def test_merge_rejects_uniform_streams_with_different_spacing():
    xa = np.arange(21) * 0.1
    xb = 2.1 + np.arange(38) * 0.05
    ya = np.zeros(xa.size)
    yb = np.zeros(xb.size)
    a = ImageStream("legendre", 3, domain=(0.0, 4.0))
    a.update(xa, ya)
    b = ImageStream("legendre", 3, domain=(0.0, 4.0))
    b.update(xb, yb)
    with pytest.raises(ValueError, match="contiguous"):
        a.merge(b)


def test_merge_of_two_single_sample_uniform_streams():
    a = ImageStream("legendre", 2, domain=(0.0, 1.0))
    a.update(np.array([0.2]), np.array([1.0]))
    b = ImageStream("legendre", 2, domain=(0.0, 1.0))
    b.update(np.array([0.5]), np.array([2.0]))
    got = a.merge(b).image()
    assert got.n == 2 and got.grid.kind == "uniform"
    assert np.allclose(got.grid.positions(), [0.2, 0.5])


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


def test_backend_keyword_accepts_numpy_and_rejects_unknown():
    x, y = _series(100)
    s = ImageStream("legendre", 4, domain=(0.0, 20.0), backend="numpy")
    s.update(x, y)
    assert s.n == 100
    with pytest.raises(ValueError):
        ImageStream("legendre", 4, domain=(0.0, 20.0), backend="nope")


def _blocks_series(n_blocks=100, per=20, seed=5):
    n = n_blocks * per
    x = np.linspace(0.0, float(n_blocks), n)
    rng = np.random.default_rng(seed)
    y = 1.0 + 0.5 * np.sin(0.3 * x) + 0.02 * rng.standard_normal(n)
    return x, y


def test_count_blocks_equal_batch_images_and_assemble():
    x, y = _blocks_series(10, 50)
    s = ImageStream("legendre", 6, domain=(0.0, 10.0), block=50)
    out = []
    for c in range(0, 500, 37):
        out.extend(s.update(x[c:c + 37], y[c:c + 37]))
    assert len(out) == 10 and len(s.fine_) == 10
    for k, img in enumerate(out):
        sl = slice(50 * k, 50 * (k + 1))
        ref = Image.of(Original(x[sl], y[sl]), "legendre", 6)
        assert img == ref
    whole = Image.of(Original(x, y, domain=(x[0], x[-1])), "legendre", 4)
    got = s.assemble(0.0, 10.0, order=4)
    _close(got.S, whole.S, 1e-10)
    _close(got.G, whole.G, 1e-10)
    assert got.n == 500


def test_length_blocks_have_fixed_domains():
    x, y = _blocks_series(8, 25)
    s = ImageStream("legendre", 4, domain=(0.0, 8.0), block=2.0)
    out = s.update(x, y)
    out += s.close()
    assert s.close() == []
    assert [img.domain for img in out] == [
        (0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
    assert sum(img.n for img in out) == 200


def test_retention_folds_old_blocks_and_keeps_assembly_exact():
    x, y = _blocks_series(100, 20)
    s = ImageStream("legendre", 4, domain=(0.0, 100.0), block=20,
                    keep_fine=8, fold=4)
    s.update(x, y)
    assert len(s.fine_) == 8 and len(s.coarse_) == 23
    assert all(img.n == 80 for img in s.coarse_)
    whole = Image.of(Original(x, y, domain=(x[0], x[-1])), "legendre", 4)
    got = s.assemble(0.0, 100.0)
    _close(got.S, whole.S, 1e-10)
    _close(got.G, whole.G, 1e-10)
    inside = s.blocks(50.0, 60.0)
    assert inside and all(
        b.domain[0] >= 50.0 - 1e-9 and b.domain[1] <= 60.0 + 1e-9
        for b in inside)


def test_block_checkpoint_resume_mid_block_matches():
    x, y = _blocks_series(6, 30)
    whole = ImageStream("legendre", 5, domain=(0.0, 6.0), block=30)
    ref = whole.update(x, y)
    s = ImageStream("legendre", 5, domain=(0.0, 6.0), block=30)
    got = s.update(x[:100], y[:100])
    state = s.checkpoint()
    import json
    state = json.loads(json.dumps(state))
    t = ImageStream("legendre", 5, domain=(0.0, 6.0), block=30).resume(state)
    got.extend(t.update(x[100:], y[100:]))
    assert len(got) == len(ref) == 6
    assert all(a == b for a, b in zip(got, ref))


def test_detect_previous_flags_a_level_shift():
    per, nb = 50, 80
    x = np.linspace(0.0, float(nb), per * nb)
    rng = np.random.default_rng(6)
    y = 2.0 + 0.05 * rng.standard_normal(x.size)
    y[per * 40:] += 3.0
    s = ImageStream("legendre", 3, domain=(0.0, float(nb)), block=per,
                    detect="previous")
    s.update(x, y)
    assert [f[0] for f in s.flags_][:1] == [40]
    assert len(s.flags_) <= 2


def test_detect_model_flags_departure_from_the_model():
    per, nb = 50, 60
    x = np.linspace(0.0, float(nb), per * nb)
    rng = np.random.default_rng(7)
    y = 2.0 + 0.05 * rng.standard_normal(x.size)
    y[per * 30:] += 3.0
    s = ImageStream("legendre", 3, domain=(0.0, float(nb)), block=per,
                    detect=("c + 0*x", [2.0], "x"))
    s.update(x, y)
    assert s.flags_ and s.flags_[0][0] == 30


def test_block_mode_errors():
    with pytest.raises(ValueError, match="channels"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), block=10, channels=2)
    with pytest.raises(ValueError, match="order \\+ 2"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), block=5)
    s = ImageStream("legendre", 4, domain=(0.0, 1.0), block=10)
    with pytest.raises(ValueError, match="no blocks"):
        s.assemble(0.0, 1.0)
    with pytest.raises(ValueError, match="detect"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), block=10, detect=3)
    with pytest.raises(ValueError, match="length"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), block=0.0)
    with pytest.raises(ValueError, match="int or a float"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), block=True)
    with pytest.raises(ValueError, match="detect needs block"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), detect="previous")
    with pytest.raises(ValueError, match="keep_fine"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), keep_fine=0)
    with pytest.raises(ValueError, match="fold"):
        ImageStream("legendre", 4, domain=(0.0, 1.0), fold=0)


def test_block_basis_rejects_count_blocks():
    with pytest.raises(ValueError, match="length block"):
        ImageStream("block", 4, domain=(0.0, 10.0), block=50)
    s = ImageStream("block", 4, domain=(0.0, 8.0), block=2.0,
                     keep_fine=2, fold=2)
    x, y = _blocks_series(8, 25)
    s.update(x, y)
    s.close()
    assert s.assemble(0.0, 8.0).n == 200


def test_close_on_accumulator_raises():
    s = ImageStream("legendre", 4, domain=(0.0, 1.0))
    with pytest.raises(ValueError, match="block mode"):
        s.close()


@pytest.mark.parametrize("basis,order,block,coarse", [
    ("legendre", 8, 1.0, 8),   # yearly blocks
    ("block", 2, 1.0, 1),      # yearly blocks into one coarse window
])
def test_partial_last_block_assembles_to_the_domain_end(
    basis, order, block, coarse
):
    """A domain that is not a whole number of block lengths ends inside its
    last block: close() finishes that block on the cut domain, and the
    assembly over the whole domain carries every sample and matches the
    direct image to rounding."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 10.5, 3832, endpoint=False)
    y = 0.5 * t + 0.3 * np.cos(2 * np.pi * t) + rng.normal(0, 0.01, t.size)
    dom = (0.0, 10.5)
    stream = ImageStream(
        basis, order, domain=dom, block=block, grid="explicit"
    )
    for _ in stream.update(t, y):
        pass
    last = stream.close()
    assert len(last) == 1 and last[0].domain == (10.0, 10.5)
    whole = stream.assemble(*dom, order=coarse)
    direct = Image.of(Original(t, y, domain=dom), basis, coarse)
    assert whole.n == direct.n == t.size
    np.testing.assert_allclose(whole.S, direct.S, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(whole.G, direct.G, rtol=1e-12, atol=1e-12)


def test_sparse_length_block_is_dropped_not_raised():
    x, y = _blocks_series(8, 25)
    gap = (x >= 2.0) & (x < 4.0)
    keep = ~gap | (np.cumsum(gap) <= 1)
    xs, ys = x[keep], y[keep]
    s = ImageStream("legendre", 4, domain=(0.0, 8.0), block=2.0)
    out = s.update(xs, ys)
    out += s.close()
    assert [img.domain for img in out] == [
        (0.0, 2.0), (4.0, 6.0), (6.0, 8.0)]


def test_merge_rejects_block_mode():
    a = ImageStream("legendre", 4, domain=(0.0, 8.0), block=2.0)
    b = ImageStream("legendre", 4, domain=(0.0, 8.0), block=2.0)
    x, y = _blocks_series(8, 25)
    a.update(x, y)
    b.update(x, y)
    with pytest.raises(ValueError, match="block mode"):
        a.merge(b)


def test_detect_previous_stays_accurate_above_order_four():
    per, nb = 50, 80
    x = np.linspace(0.0, float(nb), per * nb)
    rng = np.random.default_rng(6)
    y = 2.0 + 0.05 * rng.standard_normal(x.size)
    y[per * 40:] += 3.0
    for order in (6, 10):
        s = ImageStream("legendre", order, domain=(0.0, float(nb)),
                         block=per, detect="previous")
        s.update(x, y)
        assert [f[0] for f in s.flags_][:1] == [40]


def test_block_checkpoint_resume_with_detector_matches():
    per, nb = 50, 80
    x = np.linspace(0.0, float(nb), per * nb)
    rng = np.random.default_rng(6)
    y = 2.0 + 0.05 * rng.standard_normal(x.size)
    y[per * 40:] += 3.0
    whole = ImageStream("legendre", 3, domain=(0.0, float(nb)), block=per,
                         detect="previous")
    whole.update(x, y)
    s = ImageStream("legendre", 3, domain=(0.0, float(nb)), block=per,
                     detect="previous")
    s.update(x[:per * 45], y[:per * 45])
    state = s.checkpoint()
    import json
    state = json.loads(json.dumps(state))
    t = ImageStream("legendre", 3, domain=(0.0, float(nb)), block=per,
                     detect="previous").resume(state)
    t.update(x[per * 45:], y[per * 45:])
    assert t.flags_ == whole.flags_
    assert t._detector.n_tests_ == whole._detector.n_tests_
