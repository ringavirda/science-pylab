"""Leg 1's measurements: how fast the reduction runs, how much memory it
holds, what float32 accumulation costs, and whether the GPU can be used at
all.

The GPU check runs a real matrix product. ``dtfit._core._backend`` reports
``cupy`` as available whenever the module imports, which on this machine it
does even though cuBLAS is missing, so an availability flag is not enough.

Two memory numbers are reported per row, and they measure different
things: ``peak_mib`` is ``tracemalloc`` (traced Python and numpy
allocations of this process only, blind to a process pool's children) and
``peak_rss_mib`` is this process's own resident set. Under the
``forkserver`` and ``spawn`` start methods a pool's workers are children
of the forkserver or spawn process, never of the caller, so
``RUSAGE_CHILDREN`` stays at zero here and ``peak_rss_mib`` on a pooled
row is the parent only, same as on an unpooled one.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dtfit.image import Image, ImageStream, Original

from dtfit_experimental.experiments.domains.common import peak_memory

from . import isd_reduce, ngl_reduce
from .isd_reduce import DAY_POSITIONS, DIURNAL_ORDER

try:                                         # POSIX only
    import resource
except ImportError:                          # pragma: no cover - Windows
    resource = None                          # type: ignore[assignment]

THROUGHPUT_COLUMNS = [
    "dataset", "route", "workers", "n_files", "samples", "seconds",
    "samples_per_second", "peak_mib", "peak_rss_mib", "raw_bytes",
    "coef_bytes", "gram_bytes", "grid_bytes", "image_bytes",
    "reduction_ratio", "host", "backend", "note",
]
GEMM_COLUMNS = [
    "dataset", "backend", "order", "channels", "days", "samples",
    "repeats", "seconds", "elements_per_second", "gather_seconds",
    "note",
]


def machine_row() -> dict[str, Any]:
    """Who ran the measurement: host, architecture, cores, interpreter and
    numpy version, for a caller to log alongside a measurement. Only
    ``host`` currently flows into :data:`THROUGHPUT_COLUMNS` and
    :data:`GEMM_COLUMNS`, so that field is what tells the PC and the Pi
    rows apart in a table; the rest is for a caller that prints full
    provenance directly."""
    return {
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_count": int(os.cpu_count() or 1),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "numpy": np.__version__,
    }


def peak_rss_mib() -> float:
    """Peak resident set of this process plus its exited children, MiB.

    ``ru_maxrss`` is kibibytes on Linux and bytes on macOS. The children
    term only counts a child of *this* process: with the ``forkserver``
    or ``spawn`` start method a :class:`~concurrent.futures.
    ProcessPoolExecutor`'s workers are children of the forkserver or
    spawn process instead, so ``RUSAGE_CHILDREN`` stays zero and this
    reports the caller alone; it needs an explicit ``fork`` context to
    pick up a pool's workers. Returns ``nan`` where :mod:`resource` is
    unavailable (Windows), which the tables report as an empty field
    rather than as a zero.
    """
    if resource is None:
        return float("nan")
    total = float(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    unit = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return total / unit


def gpu_probe(backend: str = "cupy") -> tuple[bool, str]:
    """Whether ``backend`` can actually multiply, and what it said.

    Runs a 64 by 64 float64 matrix product on the device: an import check
    is not enough, since a wheel built for another CUDA major version
    imports and sees the device but fails in cuBLAS. Returns
    ``(False, "<ExceptionType>: <message>")`` on any failure, never
    raises.
    """
    try:
        if backend == "cupy":
            import cupy as cp

            a = cp.arange(64 * 64, dtype=cp.float64).reshape(64, 64)
            value = float(cp.asnumpy(a.T @ a).sum())
            if not np.isfinite(value):
                return False, "cupy: the product was not finite"
            return True, f"cupy {cp.__version__}"
        if backend == "torch":
            import torch

            if not torch.cuda.is_available():
                return False, "torch: no CUDA device"
            a = torch.arange(
                64 * 64, dtype=torch.float64, device="cuda"
            ).reshape(64, 64)
            value = float((a.T @ a).sum().item())
            if not np.isfinite(value):
                return False, "torch: the product was not finite"
            return True, f"torch {torch.__version__}"
        return False, f"unknown backend {backend!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def disk_read_rate(
    paths: Sequence[Any], *, block: int = 1 << 20
) -> dict[str, Any]:
    """Raw read throughput over ``paths``, parsing nothing.

    Read next to a reduction rate this says whether the reduction is
    I/O-bound. Measure it on files the reduction does not also read: the
    page cache would otherwise serve the second reader from memory.
    """
    paths = list(paths)
    total = 0
    started = time.perf_counter()
    for path in paths:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(block)
                if not chunk:
                    break
                total += len(chunk)
    seconds = time.perf_counter() - started
    return {
        "n_files": len(paths), "bytes": total, "seconds": seconds,
        "mb_per_second": (
            round(total / seconds / 1e6, 2) if seconds > 0 else 0.0
        ),
    }


def float32_error(
    t: np.ndarray, y: np.ndarray, order: int, *, chunk: int = 10_000
) -> dict[str, Any]:
    """The accumulation cost of a float32 ``S``, chunk by chunk, on one
    series.

    Accumulates ``S`` chunk by chunk in float32 and in float64 and
    compares both against the direct float64 image of the whole series.
    The basis is evaluated once in float64 and cast to each
    accumulator's dtype before the chunk sum, so this isolates the
    accumulation error: it does not cover a producer that also evaluates
    the basis in float32, nor ``G = Phi^T Phi``, whose conditioning at
    high order is where float32 bites hardest. Returns the maximum
    relative differences of ``S`` (``rel_S_float32``, ``rel_S_float64``)
    with ``n``, ``order`` and the chunk size. ``ImageStream`` accumulates
    in float64 whatever its backend, so the float32 arm is built here
    rather than asked of it.
    """
    from dtfit.image import make_basis, u_of

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    domain = (float(t[0]), float(t[-1]))
    basis = make_basis("legendre", int(order))
    direct = Image.of(Original(t, y, domain=domain), basis, int(order))
    out: dict[str, Any] = {
        "n": int(t.size), "order": int(order), "chunk": int(chunk),
    }
    for name, dtype in (("float64", np.float64), ("float32", np.float32)):
        S = np.zeros(basis.n_coef, dtype=dtype)
        for start in range(0, t.size, chunk):
            sl = slice(start, start + chunk)
            Phi = basis.evaluate(u_of(t[sl], *domain)).astype(dtype)
            S = S + Phi.T @ y[sl].astype(dtype)
        scale = float(np.max(np.abs(direct.S)))
        out[f"rel_S_{name}"] = float(
            np.max(np.abs(S.astype(float) - direct.S)) / scale
        )
    return out


def channel_gemm_rate(
    Y: np.ndarray,
    *,
    order: int = DIURNAL_ORDER,
    backend: str = "numpy",
    repeats: int = 3,
) -> dict[str, Any]:
    """Time the channel-form projection of a resident batch.

    ``Y`` has shape ``(24, channels)`` on the day grid
    :data:`~.isd_reduce.DAY_POSITIONS`. The read is excluded on purpose:
    this measures the GEMM, and :func:`disk_read_rate` measures the other
    half. The fastest of ``repeats`` runs is reported, with the images of
    the last one.
    """
    Y = np.asarray(Y, dtype=float)
    best = float("inf")
    images: list[Image] = []
    for _ in range(max(1, int(repeats))):
        stream = ImageStream(
            "legendre", int(order), domain=(0.0, 1.0), grid="explicit",
            channels=Y.shape[1], backend=backend,
        )
        started = time.perf_counter()
        stream.update(DAY_POSITIONS[: Y.shape[0]], Y)
        images = stream.images()
        best = min(best, time.perf_counter() - started)
    samples = int(Y.size)
    return {
        "backend": backend, "order": int(order),
        "channels": int(Y.shape[1]), "samples": samples,
        "repeats": int(repeats), "seconds": best,
        "elements_per_second": (
            round(samples / best, 1) if best > 0 else 0.0
        ),
        "images": images,
    }


def _reduce_twice(
    queue: Any,
    dataset: str,
    files: list[str],
    out_dir: str,
    workers: int,
    steps_by_station: dict[str, Sequence[float]],
    fields: tuple[str, ...],
) -> None:
    """The child of :func:`reduce_rate`: the reduction untraced for its
    wall time, then traced for its allocation peak; puts
    ``(rows, seconds, peak_mib, peak_rss_mib)`` on ``queue``."""
    paths = [Path(f) for f in files]
    if dataset == "ngl":
        def work() -> list[dict[str, Any]]:
            return ngl_reduce.reduce_many(
                paths, Path(out_dir), steps_by_station, workers=workers,
            )
    else:
        def work() -> list[dict[str, Any]]:
            return isd_reduce.reduce_many_years(
                paths, Path(out_dir), fields=fields, workers=workers,
            )
    started = time.perf_counter()
    rows = work()
    seconds = time.perf_counter() - started
    _, peak = peak_memory(work)
    queue.put((rows, seconds, peak, peak_rss_mib()))


def reduce_rate(
    paths: Sequence[Any],
    out_dir: Any,
    *,
    dataset: str,
    workers: int = 1,
    steps_by_station: Mapping[str, Sequence[float]] | None = None,
    fields: Sequence[str] = ("TMP",),
) -> dict[str, Any]:
    """Reduce a set of files and report the rate, the peak memory and the
    size reduction.

    ``dataset`` is ``"ngl"`` or ``"isd"``. The reduction runs in a fresh
    spawned interpreter that holds nothing but the packages and the
    work, twice: once untraced for ``seconds``, once under
    :func:`peak_memory` for ``peak_mib``, the traced allocation
    high-water mark of that process. ``peak_rss_mib`` is the child's own
    resident peak, read by :func:`peak_rss_mib` inside it before it
    exits; both cover the reducing process alone, not a pool's workers
    (see :func:`peak_rss_mib` for why), so the two ``cpu-1`` rows of a
    small and a large file set are the memory gate.

    Raises:
        ValueError: an unknown ``dataset``.
    """
    if dataset not in ("ngl", "isd"):
        raise ValueError(f"dataset must be 'ngl' or 'isd', got {dataset!r}")
    files = [Path(p) for p in paths]
    out_dir = Path(out_dir)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_reduce_twice, args=(
        queue, dataset, [str(f) for f in files], str(out_dir), workers,
        dict(steps_by_station or {}), tuple(fields),
    ))
    child.start()
    rows, seconds, peak, rss = queue.get()
    child.join()
    good = [r for r in rows if not r["error"]]
    samples = sum(int(r["n"]) for r in good)
    raw = sum(int(r["raw_bytes"]) for r in good)
    coef = sum(int(r["coef_bytes"]) for r in good)
    gram = sum(int(r["gram_bytes"]) for r in good)
    grid = sum(int(r["grid_bytes"]) for r in good)
    image_bytes = coef + gram + grid
    machine = machine_row()
    return {
        "dataset": dataset, "route": f"cpu-{workers}", "workers": workers,
        "n_files": len(files), "samples": samples,
        "seconds": round(seconds, 3),
        "samples_per_second": (
            round(samples / seconds, 1) if seconds > 0 else 0.0
        ),
        "peak_mib": round(peak, 2),
        "peak_rss_mib": None if np.isnan(rss) else round(rss, 2),
        "raw_bytes": raw, "coef_bytes": coef, "gram_bytes": gram,
        "grid_bytes": grid, "image_bytes": image_bytes,
        "reduction_ratio": (
            round(raw / image_bytes, 3) if image_bytes else None
        ),
        "host": machine["host"], "backend": "numpy",
        "note": f"{len(rows) - len(good)} failed",
    }
