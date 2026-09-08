"""The stochastic tier's stream: the accumulating and block forms of the
second-order image, and the block form against the filter."""

import json

import numpy as np
import pytest

from dtfit.stochastic import (
    SecondOrderImage, SecondOrderStream, StochasticFilter,
)


def ar1(n, phi, rng, sigma=1.0, burn=200):
    e = rng.normal(0.0, sigma, n + burn)
    x = np.empty(n + burn)
    x[0] = e[0]
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def walk(n, seed):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def test_accumulator_matches_the_whole_record_image():
    y = walk(2300, 3)
    st = SecondOrderStream(lag=32, nfreq=64, scales=5)
    for s in range(0, y.size, 197):
        assert st.update(y[s:s + 197]) == []
    whole = SecondOrderImage(32, 64, 5).update(y)
    assert st.n == y.size
    scale = float(np.max(np.abs(whole.acov())))
    assert np.max(np.abs(st.image().acov() - whole.acov())) < 1e-12 * scale
    assert st.image().trend() == pytest.approx(whole.trend(), rel=1e-12)


def test_block_mode_emits_blocks_and_assembles_to_the_whole_image():
    y = walk(2300, 3)
    st = SecondOrderStream(500, lag=32, nfreq=64, scales=5)
    emitted = []
    for s in range(0, y.size, 197):
        emitted.extend(st.update(y[s:s + 197]))
    assert len(emitted) == 4 and all(b.n == 500 for b in emitted)
    tail = st.close()
    assert len(tail) == 1 and tail[0].n == 300
    assembled = st.assemble(-1.0, 1e9)
    whole = SecondOrderImage(32, 64, 5).update(y)
    assert assembled.n == y.size
    scale = float(np.max(np.abs(whole.acov())))
    assert np.max(np.abs(assembled.acov() - whole.acov())) < 1e-11 * scale
    dscale = float(np.max(np.abs(whole.dft())))
    assert np.max(np.abs(assembled.dft() - whole.dft())) < 1e-11 * dscale
    assert assembled.trend()[0] == pytest.approx(whole.trend()[0], rel=1e-10)


def test_blocks_select_by_position_and_folding_stays_exact():
    y = walk(2000, 4)
    st = SecondOrderStream(100, lag=16, nfreq=32, scales=4, keep_fine=3,
                           fold=2)
    st.update(y)
    assert len(st.coarse_) == 9 and len(st.fine_) == 2
    whole = SecondOrderImage(16, 32, 4).update(y)
    asm = st.assemble(-1.0, 1e9)
    assert asm.n == 2000
    scale = float(np.max(np.abs(whole.acov())))
    assert np.max(np.abs(asm.acov() - whole.acov())) < 1e-11 * scale
    # positions are the sample indices here; a coarse block spans 200 of them
    assert [b.n for b in st.blocks(0.0, 199.0)] == [200]
    assert [b.n for b in st.blocks(0.0, 399.0)] == [200, 200]
    assert st.blocks(0.0, 99.0) == []      # the first coarse block is wider


def test_checkpoint_and_resume_reproduce_the_uninterrupted_stream():
    y = walk(2300, 3)
    a = SecondOrderStream(500, lag=32, nfreq=64, scales=5)
    cut = 0
    for s in range(0, 1200, 97):
        a.update(y[s:s + 97])
        cut = min(s + 97, y.size)
    b = SecondOrderStream(500, lag=32, nfreq=64, scales=5)
    b.resume(json.loads(json.dumps(a.checkpoint())))
    for s in range(cut, y.size, 211):
        a.update(y[s:s + 211])
        b.update(y[s:s + 211])
    a.close()
    b.close()
    ia, ib = a.assemble(-1.0, 1e9), b.assemble(-1.0, 1e9)
    assert ia.n == ib.n == y.size
    assert np.array_equal(ia.acov(), ib.acov())
    assert np.array_equal(ia.dft(), ib.dft())


def test_stream_rejects_the_wrong_mode_and_a_foreign_checkpoint():
    y = walk(600, 5)
    acc = SecondOrderStream(lag=8, nfreq=16, scales=3)
    blk = SecondOrderStream(200, lag=8, nfreq=16, scales=3)
    acc.update(y)
    blk.update(y)
    with pytest.raises(ValueError, match="block mode"):
        acc.close()
    with pytest.raises(ValueError, match="accumulator mode"):
        blk.image()
    with pytest.raises(ValueError, match="no blocks"):
        blk.assemble(1e8, 1e9)
    with pytest.raises(ValueError, match="no samples yet"):
        SecondOrderStream(lag=8, nfreq=16, scales=3).image()
    with pytest.raises(ValueError, match="configuration"):
        SecondOrderStream(100, lag=8, nfreq=16, scales=3).resume(
            blk.checkpoint())
    with pytest.raises(ValueError, match="unknown checkpoint version"):
        blk.resume({"version": 99})
    for kwargs in ({"block": 1}, {"keep_fine": 0}, {"fold": 0}):
        with pytest.raises(ValueError):
            SecondOrderStream(**kwargs)


def test_update_rejects_non_finite_and_non_1d_chunks_in_block_mode():
    # a partial block never reaches SecondOrderImage.update, so the stream
    # must check finiteness and shape itself
    st = SecondOrderStream(500, lag=8, nfreq=16, scales=3)
    with pytest.raises(ValueError, match="finite"):
        st.update(np.full(10, np.nan))
    with pytest.raises(ValueError, match="1-D"):
        st.update(np.zeros((4, 5)))
    with pytest.raises(ValueError, match="at least one sample"):
        st.update(np.zeros(0))


def test_close_drops_a_partial_block_of_a_single_sample():
    st = SecondOrderStream(500, lag=8, nfreq=16, scales=3)
    st.update(np.zeros(1))
    assert st.close() == []


def test_block_images_reach_the_filter_accuracy_on_an_ar1_coefficient():
    """At equal memory the block form and the exponentially weighted filter
    have the same steady-state error on the AR(1) coefficient; a block's
    estimate becomes available only once the block closes, so the filter,
    which updates every sample, resolves a change sooner than the block
    size."""
    n, block, halflife = 6000, 500, 173.0
    errs = {"filter": [], "blocks": []}
    delays = {"filter": [], "blocks": []}
    for s in range(8):
        rng = np.random.default_rng(s)
        y = np.concatenate([ar1(3000, 0.3, rng), ar1(3000, 0.8, rng)])
        est = {}
        flt = StochasticFilter(nlags=4, halflife=halflife, warmup=10 ** 9)
        track = np.full(n, np.nan)
        for t in range(n):
            flt.update(y[t])
            if t > 200:
                track[t] = flt.params_["ar1_phi"]
        est["filter"] = track
        track = np.full(n, np.nan)
        st = SecondOrderStream(block, lag=4, nfreq=8, scales=2)
        pos = 0
        for s0 in range(0, n, block):
            for img in st.update(y[s0:s0 + block]):
                g = img.acov()
                # the block's own estimate is only known once it closes, at
                # pos + img.n; it stays the read-out until the next block
                # closes
                track[pos + img.n:pos + 2 * img.n] = g[1] / g[0]
                pos += img.n
        est["blocks"] = track
        for key, e in est.items():
            resid = np.concatenate([e[2000:3000] - 0.3, e[4500:6000] - 0.8])
            errs[key].append(float(np.sqrt(np.nanmean(resid ** 2))))
            cross = np.where(e[3000:] > 0.55)[0]
            delays[key].append(int(cross[0]) if cross.size else n)
    assert np.mean(errs["blocks"]) < 0.05
    assert np.mean(errs["blocks"]) < np.mean(errs["filter"]) + 0.01
    assert np.median(delays["filter"]) < np.median(delays["blocks"])
