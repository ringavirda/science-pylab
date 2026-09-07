"""Fused map-reduce and GEMM-batched LSI: ``PartitionedBatchLSI``.

The accumulation is exact and additive. Chunked multi-channel updates must
land on the whole-array batched projection as well as on the per-channel
``PartitionedLSI``, with ``merge`` reducing partitions losslessly.
"""

import numpy as np
import pytest

from dtfit_experimental.scale import (
    PartitionedLSI, PartitionedBatchLSI, project_spectra,
)


@pytest.fixture
def channels():
    """Shared grid; several exp-growth channels with differing params + noise."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 4000)
    a = rng.uniform(0.5, 2.0, 8)
    b = rng.uniform(-0.3, 0.3, 8)
    Y = np.column_stack(
        [a[c] * np.exp(b[c] * x) for c in range(8)]
    ) + rng.normal(0, 0.02, (x.size, 8))
    return x, Y, a, b


def _chunked(x, Y, order=6, chunk=512, basis="legendre", domain=None):
    # The domain defaults to the data's own range. A distributed merge must
    # instead be given the global domain, or the workers project onto
    # different bases.
    dom = (float(x[0]), float(x[-1])) if domain is None else domain
    acc = PartitionedBatchLSI(
        "a*exp(b*t)", "t", domain=dom,
        n_channels=Y.shape[1], order=order, basis=basis)
    for i in range(0, x.size, chunk):
        acc.update(x[i:i + chunk], Y[i:i + chunk])
    return acc


@pytest.mark.parametrize("basis", ["legendre", "chebyshev", "fourier", "laguerre"])
def test_fused_spectra_equal_whole_array(channels, basis):
    x, Y, _, _ = channels
    fused = _chunked(x, Y, basis=basis).spectra()
    whole = project_spectra(x, Y, order=6, basis=basis, backend="numpy")
    np.testing.assert_allclose(fused, whole, rtol=1e-9, atol=1e-11)


def test_fused_spectrum_equals_per_channel_partitioned(channels):
    x, Y, _, _ = channels
    fused = _chunked(x, Y).spectra()
    for c in (0, 3, 7):
        p = PartitionedLSI("a*exp(b*t)", "t", domain=(0.0, 10.0), order=6)
        for i in range(0, x.size, 512):
            p.update(x[i:i + 512], Y[i:i + 512, c])
        np.testing.assert_allclose(fused[c], p.spectrum(), rtol=1e-9, atol=1e-11)


def test_merge_is_exact_associative_reduce(channels):
    """Two workers on disjoint sub-ranges merge to exactly the single-pass
    result, given the global domain and a shared boundary sample."""
    x, Y, _, _ = channels
    g = (float(x[0]), float(x[-1]))                  # global domain for both
    half = x.size // 2
    left = _chunked(x[:half + 1], Y[:half + 1], domain=g)   # share boundary sample
    right = _chunked(x[half:], Y[half:], domain=g)
    merged = left.merge(right).spectra()
    whole = _chunked(x, Y, domain=g).spectra()
    np.testing.assert_allclose(merged, whole, rtol=1e-9, atol=1e-11)


def test_fused_recovers_parameters(channels):
    x, Y, a, b = channels
    results = _chunked(x, Y).fit(p0=[1.0, 0.0])
    for c in range(Y.shape[1]):
        assert results[c].coeffs[0] == pytest.approx(a[c], abs=5e-2)
        assert results[c].coeffs[1] == pytest.approx(b[c], abs=5e-2)


def test_single_channel_1d_input(channels):
    x, Y, _, _ = channels
    acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=(0.0, 10.0), n_channels=1, order=6)
    for i in range(0, x.size, 512):
        acc.update(x[i:i + 512], Y[i:i + 512, 0])  # 1-D column -> (n, 1)
    assert acc.spectra().shape == (1, 7)


def test_channel_count_mismatch_raises(channels):
    x, Y, _, _ = channels
    acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=(0.0, 10.0), n_channels=3, order=6)
    with pytest.raises(ValueError, match="channels"):
        acc.update(x[:512], Y[:512])  # 8 channels != 3


def test_update_accepts_plain_lists(channels):
    # ``update`` coerces its arguments before touching them: list-of-lists
    # chunks accumulate exactly the ndarray-fed state. Nothing reads
    # ``.shape`` off the raw arguments.
    x, Y, _, _ = channels
    ref = PartitionedBatchLSI(
        "a*exp(b*t)", "t", domain=(0.0, 10.0), n_channels=Y.shape[1], order=6)
    alt = PartitionedBatchLSI(
        "a*exp(b*t)", "t", domain=(0.0, 10.0), n_channels=Y.shape[1], order=6)
    for i in range(0, x.size, 512):
        ref.update(x[i:i + 512], Y[i:i + 512])
        alt.update(list(x[i:i + 512]), [list(row) for row in Y[i:i + 512]])
    assert alt.n_samples == ref.n_samples == x.size
    np.testing.assert_array_equal(alt.spectra(), ref.spectra())
