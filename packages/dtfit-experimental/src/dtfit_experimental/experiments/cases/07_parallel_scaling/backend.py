"""Benchmark code for the parallel-scaling case study.

``07_parallel_scaling.ipynb`` imports this module and owns the presentation.

The question is how much of a multi-core box dtfit can actually turn into
throughput, measured as speedup against the rank of parallelism ``P``. Three
routes are timed. Each hits a different ceiling; the ceilings are the
interesting part.

:func:`kernel_scaling` runs the numpy Simpson kernel under P threads. The
numpy path holds the GIL, so this leg measures Python-level threading rather
than a compute-bound near-linear win; :func:`amdahl_serial_fraction` fits the
serial fraction implied by its curve.
:func:`fitmany_scaling` fans independent fits across loky processes. The fits
themselves are embarrassingly parallel, but per-task dispatch and the per-fit
SymPy lambdify put a ceiling on fine-grained work. :func:`mapreduce_scaling`
threads a ``PartitionedLSI`` stream (adaptation #1): NumPy drops the GIL on the
bulk array ops so the partitions genuinely overlap, yet the workload is
memory-bandwidth-bound and the speedup flattens there instead.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dtfit import fit_many
from dtfit.image import FittingProblem
from dtfit._core import _kernels
from dtfit_experimental.scale import PartitionedLSI

__all__ = [
    "N_CORES", "PHYS",
    "amdahl_serial_fraction", "kernel_scaling", "fitmany_scaling",
    "mapreduce_scaling",
]

N_CORES = os.cpu_count() or 8
PHYS = N_CORES // 2


def amdahl_serial_fraction(Ps, speedups):
    """Fit Amdahl's serial fraction ``s`` to an observed speedup curve: the
    ``s`` for which ``1 / (s + (1 - s)/P)`` best matches the measurements."""
    from scipy.optimize import least_squares
    Ps = np.asarray(Ps, float)
    sp = np.asarray(speedups, float)
    sol = least_squares(lambda s: 1.0 / (s + (1 - s) / Ps) - sp, 0.05,
                        bounds=(0, 1))
    return float(sol.x[0])


def kernel_scaling(Ps, rep_per_thread=4000):
    """Weak-scaling throughput of the numpy Simpson kernel under P threads.

    Each of ``P`` threads runs ``rep_per_thread`` ``simpson_windows`` calls
    on cache-resident data. The numpy path holds the GIL, so P threads do not
    overlap the compute. Returns
    ``({P: throughput-multiplier}, {P: wall-time})``.
    """
    x = np.ascontiguousarray(np.linspace(0, 10, 40_000))
    y = np.ascontiguousarray(np.sin(x))
    starts = np.arange(0, 39_000, 200, dtype=np.intp)
    stops = starts + 200

    def work(_):
        acc = 0.0
        for _ in range(rep_per_thread):
            acc += _kernels.simpson_windows(y, x, starts, stops).sum()
        return acc

    def run(P):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=P) as ex:
            list(ex.map(work, range(P)))
        return time.perf_counter() - t0

    run(1)  # warm
    times = {P: run(P) for P in Ps}
    # Weak scaling: P threads issue P times the calls, so the throughput
    # multiplier is P * t(1) / t(P) rather than t(1) / t(P).
    return {P: (P * times[1]) / times[P] for P in Ps}, times


def fitmany_scaling(Ps, n_problems):
    """Strong-scaling speedup of ``fit_many`` (EAC) across loky processes.

    Builds ``n_problems`` independent ``a*exp(b*t)`` fits and times them at
    each worker count ``P``. Returns ``({P: speedup vs P=1}, {P: wall-time})``.
    """
    rng = np.random.default_rng(0)
    probs = []
    for i in range(n_problems):
        x = np.linspace(0, 1.5, 400)
        y = (1 + 0.2 * (i % 5)) * np.exp((0.6 + 0.05 * (i % 7)) * x) + rng.normal(0, 0.03, 400)
        probs.append(FittingProblem(x=x, y=y, expr="a*exp(b*t)", var="t",
                                method="eac", kwargs={"p0": [1.0, 1.0]}))
    fit_many(probs[:16], n_jobs=max(Ps), backend="loky")  # Pay the spawn cost.
    times = {P: _timed(lambda: fit_many(probs, n_jobs=P, backend="loky")) for P in Ps}
    return {P: times[1] / times[P] for P in Ps}, times


def mapreduce_scaling(Ps, total):
    """Strong-scaling speedup of a threaded ``PartitionedLSI`` map-reduce.

    A ``total``-sample exponential stream is split into ``P`` partitions, one
    thread each, relying on NumPy to drop the GIL over the bulk ops. Returns
    ``({P: speedup vs P=1}, {P: wall-time})``.
    """
    def work(args):
        t0, t1, n, seed = args
        rng = np.random.default_rng(seed)
        acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
        CH = 2_000_000
        nc = max(1, n // CH)
        span = (t1 - t0) / nc
        for c in range(nc):
            x = np.linspace(t0 + c * span, t0 + (c + 1) * span, CH, endpoint=False)
            y = np.exp(0.2 * x) + rng.normal(0, 0.05, CH)
            acc.update(x, y)
        return acc._s

    def run(P):
        parts = [(10.0 * i / P, 10.0 * (i + 1) / P, total // P, i) for i in range(P)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=P) as ex:
            list(ex.map(work, parts))
        return time.perf_counter() - t0

    run(2)  # warm
    times = {P: run(P) for P in Ps}
    return {P: times[1] / times[P] for P in Ps}, times


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
