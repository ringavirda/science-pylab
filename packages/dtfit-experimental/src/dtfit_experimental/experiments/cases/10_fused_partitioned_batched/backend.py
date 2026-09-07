"""Drivers and estimators for the fused multi-channel big-data case study.

``10_fused_partitioned_batched.ipynb`` imports this module and owns the
presentation.

The subject is ``PartitionedBatchLSI``, which fuses two big-data levers the
other cases exercise on their own. ``PartitionedLSI`` partitions by volume:
flat ``O(order)`` memory and an exact one-pass reduce, but one channel at a
time. ``project_spectra`` batches by channel: one matmul across all of
them, GPU-pluggable, but the whole volume resident at ``O(N)``. Each covers
the other's weakness, and the fused estimator is the claim that you can
have both at once. Both live in :mod:`dtfit_experimental.scale`.

:func:`fused_project` folds each chunk's GEMM into a ``(B, n_coef)``
accumulator; :func:`loop_project` runs the per-channel ``PartitionedLSI`` loop
that it replaces; the whole-array reference comes straight from dtfit.
:func:`perf_sweep` times all three against channel count while
:func:`mem_sweep` watches peak memory as the volume grows. :func:`gpu_sweep`
repeats the fused path on a GPU backend when one is present, and
``curve_fit`` runs alongside as the nonlinear accuracy reference.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from dtfit_experimental import available_backends, resolve_backend
from dtfit_experimental.scale import (
    PartitionedLSI, PartitionedBatchLSI, fit_lsi_batched, project_spectra,
)

try:  # Optional: the external nonlinear accuracy reference.
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

try:  # Optional: sklearn's r2_score, else the NumPy fallback below.
    from sklearn.metrics import r2_score as _sk_r2_score
except Exception:  # pragma: no cover
    _sk_r2_score = None

__all__ = [
    "DOMAIN", "ORDER", "HAVE_SCIPY", "HAVE_SKLEARN",
    "make_channels", "fused_project", "loop_project",
    "project_spectra", "fit_lsi_batched",
    "r2_score", "curvefit_all", "recon_r2", "struct_recon", "r2_window",
    "time_best", "gpu_name", "available_backends", "resolve_backend",
    "perf_sweep", "mem_sweep", "gpu_sweep",
]

DOMAIN = (0.0, 10.0)
ORDER = 6
HAVE_SCIPY = curve_fit is not None
HAVE_SKLEARN = _sk_r2_score is not None


def r2_score(y_true, y_pred):
    """Coefficient of determination, from sklearn when it is installed and
    from a plain NumPy fallback when it is not."""
    if _sk_r2_score is not None:
        return float(_sk_r2_score(y_true, y_pred))
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def gpu_name() -> str:
    """Human-readable name of CUDA device 0, or a generic placeholder."""
    try:  # pragma: no cover - only on a CUDA box
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # pragma: no cover
        return "the GPU"


def make_channels(n, b_ch, *, seed=0, noise=0.02, dtype=np.float64):
    """``b_ch`` exponential-growth channels ``a_c·exp(b_c·t)`` on a shared grid.

    Returns ``(x, Y, a, b)``: ``Y`` has shape ``(n, b_ch)``, and ``a`` and
    ``b`` are the true per-channel amplitudes and rates.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(DOMAIN[0], DOMAIN[1], n)
    a = rng.uniform(0.5, 2.0, b_ch)
    b = rng.uniform(-0.25, 0.25, b_ch)
    Y = a[None, :] * np.exp(np.outer(x, b)) + rng.normal(0, noise, (n, b_ch))
    return x, Y.astype(dtype), a, b


def time_best(fn, reps=3):
    """Best-of-``reps`` wall time (seconds) for ``fn``; returns ``(best, out)``."""
    best = float("inf")
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def fused_project(x, Y, chunk, backend, *, domain=DOMAIN):
    """Fused ``PartitionedBatchLSI`` projection.

    Each chunk's ``B``-channel partial integrals are one backend GEMM folded
    into a ``(B, n_coef)`` accumulator, giving flat memory over the volume and
    a single matmul over the channels in the same pass.
    """
    acc = PartitionedBatchLSI(
        "a*exp(b*t)", "t", domain=domain, n_channels=Y.shape[1],
        order=ORDER, backend=backend)
    for i in range(0, x.size, chunk):
        acc.update(x[i:i + chunk], Y[i:i + chunk])
    return acc


def loop_project(x, Y, chunk, *, domain=DOMAIN):
    """Per-channel ``PartitionedLSI`` loop, the thing the fused GEMM replaces.

    Memory stays flat, but the channels are walked in a Python loop.
    """
    accs = [PartitionedLSI("a*exp(b*t)", "t", domain=domain, order=ORDER)
            for _ in range(Y.shape[1])]
    for i in range(0, x.size, chunk):
        xs = x[i:i + chunk]
        for c, acc in enumerate(accs):
            acc.update(xs, Y[i:i + chunk, c])
    return accs


def curvefit_all(x, Y, *, p0=(1.0, 0.0), maxfev=4000):
    """Per-channel ``scipy.optimize.curve_fit`` of ``a·exp(b·t)``, the
    gold-standard nonlinear accuracy reference.

    Returns a ``(B, 2)`` array of ``(a, b)``, with a channel that failed to
    converge left as NaN. Raises if scipy is unavailable.
    """
    if curve_fit is None:  # pragma: no cover
        raise RuntimeError("scipy is required for curvefit_all")
    f = lambda t, a, b: a * np.exp(b * t)  # noqa: E731
    out = np.zeros((Y.shape[1], 2))
    for c in range(Y.shape[1]):
        try:
            out[c] = curve_fit(f, x, Y[:, c], p0=list(p0), maxfev=maxfev)[0]
        except Exception:
            out[c] = np.nan
    return out


def struct_recon(x, a, b):
    """Reconstruct the structured family ``a·exp(b·t)`` over ``x`` for all channels."""
    return a[None, :] * np.exp(np.outer(x, b))


def recon_r2(x, Y, a, b):
    """Mean per-channel reconstruction R² of ``a·exp(b·t)`` against ``Y``."""
    yhat = struct_recon(x, a, b)
    return float(np.mean([r2_score(Y[:, c], yhat[:, c]) for c in range(Y.shape[1])]))


def r2_window(Y, Yhat, idx):
    """Mean per-channel R² over the sample indices ``idx`` (in/out-of-window)."""
    return float(np.mean([r2_score(Y[idx, c], Yhat[idx, c]) for c in range(Y.shape[1])]))


def perf_sweep(n_perf, Bs, chunk, *, seed=1, noise=0.03):
    """Projection throughput vs channel count ``B``.

    For each ``B`` in ``Bs``, takes the best of three timings for projecting
    the whole ``n_perf``-sample stream three ways: the fused GEMM, the
    per-channel partitioned loop and the whole-array ``project_spectra``.
    ``curve_fit`` is timed at the smallest ``B`` alone, since it does not
    scale and timing it everywhere would dominate the run.

    Returns ``(perf, cf_perf)``. ``perf`` maps each method name to a list of
    seconds aligned with ``Bs``; ``cf_perf`` holds the curve_fit times.
    """
    xp = np.linspace(DOMAIN[0], DOMAIN[1], n_perf)
    rng = np.random.default_rng(seed)
    perf = {"fused (GEMM, numpy)": [], "per-channel PartitionedLSI loop": [],
            "whole-array project_spectra": []}
    cf_perf = []
    b_min = min(Bs)
    for B in Bs:
        a = rng.uniform(0.5, 2.0, B); b = rng.uniform(-0.25, 0.25, B)
        Yb = a[None, :] * np.exp(np.outer(xp, b)) + rng.normal(0, noise, (n_perf, B))
        perf["fused (GEMM, numpy)"].append(
            time_best(lambda: fused_project(xp, Yb, chunk, "numpy"))[0])
        perf["per-channel PartitionedLSI loop"].append(
            time_best(lambda: loop_project(xp, Yb, chunk))[0])
        perf["whole-array project_spectra"].append(
            time_best(lambda: project_spectra(xp, Yb, order=ORDER, backend="numpy"))[0])
        if B <= b_min and HAVE_SCIPY:  # Smallest B only; see the docstring.
            f = lambda t, aa, bb: aa * np.exp(bb * t)  # noqa: E731
            cf_perf.append(time_best(
                lambda: [curve_fit(f, xp, Yb[:, c], p0=[1.0, 0.0], maxfev=4000)[0]
                         for c in range(B)], reps=1)[0])
    return perf, cf_perf


def mem_sweep(vols, chunk, *, b_mem=128, noise=0.03):
    """Peak ``tracemalloc`` memory over ``b_mem`` channels as the volume grows.

    The fused path generates and consumes chunk by chunk and never
    materializes the full ``Y``; the whole-array path has to hold ``Y`` in RAM.
    Showing that gap is the point of the sweep. Returns
    ``(mem_fused, mem_whole)``, lists of peak MB aligned with ``vols``.
    """
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 2.0, b_mem); b = rng.uniform(-0.25, 0.25, b_mem)
    mem_fused, mem_whole = [], []
    for nv in vols:
        def stream_fused(nv=nv):
            acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=DOMAIN,
                                      n_channels=b_mem, order=ORDER)
            rg = np.random.default_rng(2)
            xs_full = np.linspace(DOMAIN[0], DOMAIN[1], nv)
            for i in range(0, nv, chunk):
                xs = xs_full[i:i + chunk]
                Ys = a[None, :] * np.exp(np.outer(xs, b)) + rg.normal(0, noise, (xs.size, b_mem))
                acc.update(xs, Ys)
            return acc

        def whole_array(nv=nv):
            xs = np.linspace(DOMAIN[0], DOMAIN[1], nv)
            Y = a[None, :] * np.exp(np.outer(xs, b)) + np.random.default_rng(2).normal(0, noise, (nv, b_mem))
            return project_spectra(xs, Y, order=ORDER, backend="numpy")

        tracemalloc.start()
        stream_fused()
        mem_fused.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
        tracemalloc.start()
        whole_array()
        mem_whole.append(tracemalloc.get_traced_memory()[1] / 1e6)
        tracemalloc.stop()
    return mem_fused, mem_whole


def gpu_sweep(n_perf, b_gpu, chunk):
    """Fused projection on numpy (CPU) against the first available GPU backend.

    Returns ``(gpu, rows)``: the backend name, or ``None`` when no GPU backend
    is present or usable, and a list of
    ``[dtype, t_cpu_ms, t_gpu_ms, speedup]``.
    """
    backends = available_backends()
    gpu = next((bk for bk in ("cupy", "torch") if bk in backends), None)
    rows = []
    if not gpu:
        return None, rows
    try:
        for dt, dname in [("float64", "fp64"), ("float32", "fp32")]:
            xg, Yg, _, _ = make_channels(
                n_perf, b_gpu, seed=3,
                dtype=np.float32 if dt == "float32" else np.float64)
            bk_cpu = resolve_backend("numpy", dtype=dt)
            bk_gpu = resolve_backend(gpu, dtype=dt)
            t_cpu = time_best(lambda: fused_project(xg, Yg, chunk, bk_cpu))[0]
            t_gpu = time_best(lambda: fused_project(xg, Yg, chunk, bk_gpu))[0]
            rows.append([dname, t_cpu * 1e3, t_gpu * 1e3, t_cpu / t_gpu])
    except Exception:  # pragma: no cover - GPU present but no usable device
        return None, []
    return gpu, rows
