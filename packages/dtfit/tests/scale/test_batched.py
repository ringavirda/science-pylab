"""GEMM-batched, backend-pluggable spectral projection.

The batched projection must equal the per-channel loop. The
``project_integral`` factoring must reproduce ``np.trapezoid`` to roundoff.
``PartitionedLSI`` is built on that same factoring, which is why the
equivalence is checked here for every basis.
"""

import numpy as np
import pytest

from dtfit import fit_lsi_batched
from dtfit.scale import project_spectra
from dtfit._core._backend import available_backends, resolve_backend
from dtfit._core._spectral import LaguerreBasis, make_basis, _trapz_weights
from sklearn.metrics import r2_score


@pytest.fixture
def channels():
    """A shared grid with several exp-growth channels of differing params."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 3.0, 500)
    params = [(1.0, 0.9), (1.5, 0.6), (0.8, 1.2), (2.0, 0.4)]
    Y = np.column_stack(
        [a * np.exp(b * x) + rng.normal(0, 0.02, x.size) for a, b in params]
    )
    return x, Y, params


@pytest.mark.parametrize("basis", ["legendre", "chebyshev", "fourier", "laguerre"])
def test_project_integral_matches_trapezoid(basis):
    x = np.linspace(0.0, 2.0, 257)
    y = np.sin(2.0 * x) + 0.3 * x
    b = make_basis(basis, 6, (float(x[0]), float(x[-1])))
    D, w = b._gemm_factors(x)
    gemm = D.T @ (w * y)
    trapz = np.trapezoid(y[:, None] * D, x[:, None] if basis != "laguerre" else None,
                         axis=0) if basis != "laguerre" else None
    if basis == "laguerre":
        # laguerre integrates over u, with e^-u already folded into D
        assert isinstance(b, LaguerreBasis)
        u = b._to_u(x)
        trapz = np.trapezoid(y[:, None] * D, u[:, None], axis=0)
    assert trapz is not None
    np.testing.assert_allclose(gemm, trapz, rtol=1e-10, atol=1e-12)


def test_trapz_weights_reproduce_numpy():
    x = np.array([0.0, 0.5, 1.5, 4.0, 4.1])
    y = np.array([1.0, 2.0, 0.5, 3.0, 1.0])
    assert _trapz_weights(x) @ y == pytest.approx(np.trapezoid(y, x))


@pytest.mark.parametrize("basis", ["legendre", "chebyshev", "fourier", "laguerre"])
def test_batched_spectra_equal_looped(channels, basis):
    x, Y, _ = channels
    b = make_basis(basis, 6, (float(x[0]), float(x[-1])))
    looped = np.array(
        [b.integral_to_spectrum(b.project_integral(x, Y[:, i]))
         for i in range(Y.shape[1])]
    )
    batched = project_spectra(x, Y, order=6, basis=basis, backend="numpy")
    np.testing.assert_allclose(batched, looped, rtol=1e-10, atol=1e-12)


def test_project_spectra_single_channel_is_1d(channels):
    x, Y, _ = channels
    s = project_spectra(x, Y[:, 0], order=6, basis="legendre")
    assert s.ndim == 1 and s.size == 7


def test_project_spectra_shape_validation(channels):
    x, _, _ = channels
    with pytest.raises(ValueError):
        project_spectra(x, np.ones((x.size + 3, 2)), order=4)


def test_fit_lsi_batched_recovers_all_channels(channels):
    x, Y, params = channels
    results = fit_lsi_batched(x, Y, "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
    assert isinstance(results, list) and len(results) == len(params)
    for r, (a, b) in zip(results, params):
        assert abs(r.coeffs[0] - a) < 0.2 and abs(r.coeffs[1] - b) < 0.2
        assert r2_score(Y[:, params.index((a, b))], np.asarray(r.model(x))) > 0.99


def test_fit_lsi_batched_single_channel_returns_one(channels):
    x, Y, params = channels
    r = fit_lsi_batched(x, Y[:, 0], "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
    assert not isinstance(r, list)
    assert abs(r.coeffs[1] - params[0][1]) < 0.2


def test_batched_matches_serial_fit_lsi(channels):
    import dtfit as dt

    x, Y, _ = channels
    batched = fit_lsi_batched(x, Y, "a*exp(b*t)", "t", order=6, p0=[1.0, 1.0])
    assert isinstance(batched, list)
    for i, r in enumerate(batched):
        serial = dt.fit_lsi(x, Y[:, i], "a*exp(b*t)", "t", p0=[1.0, 1.0])
        np.testing.assert_allclose(r.coeffs, serial.coeffs, rtol=0.05, atol=0.05)


def test_numpy_backend_always_available():
    assert "numpy" in available_backends()


def test_auto_backend_roundtrips():
    # "auto" picks a GPU backend where one is installed and numpy otherwise.
    # Either way the asarray/to_host roundtrip must be lossless.
    bk = resolve_backend("auto")
    assert bk.name in {"numpy", "cupy", "torch"}
    arr = bk.asarray(np.arange(4.0))
    np.testing.assert_array_equal(bk.to_host(arr), np.arange(4.0))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        resolve_backend("quantum")
