"""Benchmarks for the GEMM-batched projection case study.

``08_gpu_batched_projection.ipynb`` imports this module and owns the
presentation.

The data side of LSI and EAC is an integral ``beta_j = integral y*phi_j dx``,
and that integral factors exactly into one matrix product,
``beta = D^T (w * y)``, for a design matrix ``D`` and trapezoid weights ``w``.
Stack ``B`` channels that share a grid into the columns of ``Y`` and the whole
batch collapses to a single GEMM, ``S = D^T (w * Y)``. This module measures
what that reframing is worth.

:func:`bench_loop_vs_batched` puts one BLAS GEMM against a per-channel Python
loop and reports how far apart their spectra land, so a speedup bought by
cutting a corner would show up. :func:`bench_dtype` sweeps fp32 against fp64,
where the kernel being bandwidth-bound predicts the outcome, and
:func:`host_copy_bandwidth` supplies the CPU roofline those numbers are read
against. :func:`bench_resident_streamed` times each backend twice, once with
the arrays already on the device and once transferring them every call, which
is the difference between a GPU win and a PCIe bill.

The GPU rows come entirely from :func:`dtfit_experimental.available_backends`.
On a CUDA box the resident and streamed columns are measured; with no usable
GPU only the ``numpy`` backend appears and the notebook falls back to the CPU
numbers and the PCIe-bound model, so it completes either way.
"""

from __future__ import annotations

import time

import numpy as np

from dtfit._core._spectral import make_basis
from dtfit_experimental import available_backends, resolve_backend
from dtfit_experimental.scale import project_spectra

from dtfit_experimental.experiments.common import fmt

__all__ = [
    "ORDER",
    "gpu_name",
    "make_data",
    "host_copy_bandwidth",
    "bench_loop_vs_batched",
    "bench_dtype",
    "bench_resident_streamed",
    "has_gpu_backend",
    "available_backends",
    "fmt",
]

ORDER = 6  # k = order + 1 = 7 spectral coefficients.


def gpu_name() -> str:
    """Best-effort device name of the active GPU backend (for the report)."""
    try:  # pragma: no cover - only on a CUDA box
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # pragma: no cover
        try:
            import torch

            return torch.cuda.get_device_name(0)
        except Exception:
            return "the GPU"


def has_gpu_backend() -> bool:
    """True iff a non-``numpy`` (GPU) backend is present and usable.

    Every non-numpy backend gets a tiny GEMM. If none of them resolves and
    runs, the host is CPU-only and the notebook's GPU rows read "n/a".
    """
    for name in available_backends():
        if name == "numpy":
            continue
        try:  # pragma: no cover - depends on the host
            bk = resolve_backend(name, dtype="float32")
            a = bk.asarray(np.ones((2, 2), dtype="float32"))
            bk.to_host(a.T @ a)
            return True
        except Exception:  # pragma: no cover
            continue
    return False


def _time(fn, reps: int) -> float:
    """Best (min) wall time of ``fn`` over ``reps`` runs."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_data(N: int, B: int, seed: int = 0):
    """A bank of ``B`` channels ``y = exp(b*t)`` on a shared ``N``-point grid."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 10.0, N)
    rates = np.linspace(0.1, 0.5, B)
    Y = np.exp(np.outer(x, rates)) + rng.normal(0.0, 0.01, (N, B))
    return x, np.ascontiguousarray(Y)


def host_copy_bandwidth(nbytes: int = 256 << 20, reps: int = 5) -> float:
    """Host memory copy bandwidth in GB/s, the CPU roofline anchor."""
    a = np.ones(nbytes // 8, dtype=np.float64)
    b = np.empty_like(a)
    t = _time(lambda: np.copyto(b, a), reps)
    return 2.0 * a.nbytes / t / 1e9  # read + write


def bench_loop_vs_batched(N, Bs, reps):
    """Per-channel Python loop vs a single batched GEMM through NumPy/BLAS.

    Returns rows ``[B, loop_ms, batched_ms, speedup, Melem/s, GB/s,
    max|delta|]``, numbers formatted by :func:`fmt`. ``max|delta|`` is the
    largest spectrum difference between the two paths over a capped sample of
    channels; it is reported, not asserted on, and lands near machine epsilon
    when the batched path is doing the same arithmetic as the loop.
    """
    rows = []
    for B in Bs:
        x, Y = make_data(N, B)
        b = make_basis("legendre", ORDER, (float(x[0]), float(x[-1])))
        loop = lambda: [b.integral_to_spectrum(b.project_integral(x, Y[:, i]))  # noqa: E731
                        for i in range(B)]
        batch = lambda: project_spectra(x, Y, order=ORDER, backend="numpy")  # noqa: E731
        loop(); batch()  # warm
        t_loop = _time(loop, max(2, reps // 2))  # Slow but stable; fewer reps.
        t_batch = _time(batch, reps)
        # The batched result has to match the loop. Only a capped subset of
        # channels is checked; the full comparison is not cheap at large B.
        nchk = min(B, 32)
        sb = np.atleast_2d(np.asarray(batch()))[:nchk]
        sl = np.array([b.integral_to_spectrum(b.project_integral(x, Y[:, i]))
                       for i in range(nchk)])
        max_diff = float(np.max(np.abs(sb - sl)))
        gbps = N * B * 8 / t_batch / 1e9
        rows.append([B, fmt(t_loop * 1e3, "{:.2f}"), fmt(t_batch * 1e3, "{:.2f}"),
                     fmt(t_loop / t_batch, "{:.1f}"), fmt(N * B / t_batch / 1e6, "{:.0f}"),
                     fmt(gbps, "{:.1f}"), f"{max_diff:.1e}"])
    return rows


def bench_dtype(N, B, reps):
    """The same batched projection in fp64 and fp32.

    Returns rows ``[dtype, time_ms, Melem/s, GB/s]``. Halving the bytes per
    element should lift throughput roughly in proportion, because the kernel
    is bandwidth-bound rather than compute-bound.
    """
    x, Y0 = make_data(N, B)
    rows = []
    for dt in ["float64", "float32"]:
        Y = Y0.astype(dt)  # Stored in the target precision, not cast later.
        bk = resolve_backend("numpy", dtype=dt)
        run = lambda: project_spectra(x, Y, order=ORDER, backend=bk)  # noqa: E731
        run()
        t = _time(run, reps)
        rows.append([dt, fmt(t * 1e3, "{:.2f}"), fmt(N * B / t / 1e6, "{:.0f}"),
                     fmt(N * B * np.dtype(dt).itemsize / t / 1e9, "{:.1f}")])
    return rows


def bench_resident_streamed(N, B, reps):
    """The same GEMM on every available backend, timed two ways.

    Resident means the arrays already live on the device and only the matmul
    is timed. Streamed means they are transferred on every call, the real
    cost of host- or disk-resident data.

    Returns rows ``[backend, dtype, resident_ms, streamed_ms,
    streamed/resident, Melem/s(resident)]``. A CPU-only host yields ``numpy``
    alone, where the two timings coincide because nothing is transferred; a
    CUDA box fills in the GPU backends as well. A backend that fails to run is
    dropped, not raised, so a missing GPU cannot break the notebook.
    """
    x, Y0 = make_data(N, B)
    b = make_basis("legendre", ORDER, (float(x[0]), float(x[-1])))
    D, w = b._gemm_factors(x)
    Dw0 = np.ascontiguousarray(w[:, None] * D)  # Fold w into the small D once.
    rows = []
    for name in available_backends():
        try:
            bk = resolve_backend(name, dtype="float64")
        except Exception:  # pragma: no cover - backend not usable on this host
            continue
        for dt in ("float64", "float32"):
            try:
                bk = resolve_backend(name, dtype=dt)
                Dw = Dw0.astype(dt); Y = Y0.astype(dt)

                def streamed(bk=bk, Dw=Dw, Y=Y):  # Send Y, GEMM, fetch back.
                    return bk.to_host(bk.asarray(Dw).T @ bk.asarray(Y))

                Dd = bk.asarray(Dw); Yd = bk.asarray(Y)  # Transferred once.

                def resident(bk=bk, Dd=Dd, Yd=Yd):
                    return bk.to_host(Dd.T @ Yd)

                streamed(); resident()  # warm
                t_s = _time(streamed, reps)
                t_r = _time(resident, reps)
            except Exception:  # pragma: no cover - GPU absent / out of memory
                continue
            rows.append([name, dt, fmt(t_r * 1e3, "{:.2f}"), fmt(t_s * 1e3, "{:.2f}"),
                         fmt(t_s / t_r, "{:.1f}"), fmt(N * B / t_r / 1e6, "{:.0f}")])
    return rows
