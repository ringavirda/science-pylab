# GIL-released compiled kernels

> **Status (2026-09):** the compiled kernel was removed in the image-core
> migration; the projections it accelerated now run on numpy. The sections
> below describe the study as it was run.

**Verdict: WORKS -- the clean parallelization win.** Releasing the GIL around the
pure-C numeric loops lets a thread pool drive every physical core at near-linear
efficiency.

Source: [`_core/_native.c`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/_core/_native.c)
(`simpson_windows`, `simpson_windows_rows`, `legendre_project`).
Tested in: [Parallel scaling (7)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/07_parallel_scaling/07_parallel_scaling.ipynb).

## What it is

The three hot kernels are compiled C over raw pointers. Each is refactored so that
**argument validation, bounds checks and output allocation happen while the GIL is
held** (they touch Python objects), then the **pure compute loop is wrapped in
`Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`** (it touches no Python object).
A thread pool can then run many kernel calls *truly* concurrently.

## Measured results

Each of P threads runs a fixed batch of native Simpson calls on cache-resident
data (Exp 7):

| threads P | throughput x | efficiency % |
|---|---|---|
| 1 | 1.00 | 100 |
| 2 | 1.98 | 99 |
| 4 | 3.84 | 96 |
| 8 | 7.57 | 95 |
| 16 | 9.35 | 58 |

Peak **9.3x at P=16**, **>95% efficiency through 8 cores**, Amdahl serial
fraction **s ~= 0.041**.

## Why it works (and why it's safe)

1. **The GIL is the only thing serializing the threads.** The Simpson/Legendre
   loops are arithmetic over `double*` buffers -- no reference counting, no Python
   object access, no allocation inside the loop. The GIL was being held purely as
   a side-effect of being called from CPython, not because the work needs it.
   Dropping it for the loop removes the one global lock; the cores then run
   independently.

2. **Correctness is preserved because the dangerous parts stay locked.** Anything
   that can raise (bounds errors), allocate (the output array), or touch Python
   refcounts is done *before* `Py_BEGIN_ALLOW_THREADS`. The released region only
   reads input buffers and writes its own output slice. Native-vs-fallback
   equivalence tests confirm results are bit-for-bit unchanged after the refactor.

3. **The workload is compute-bound on cache-resident data.** The benchmark keeps
   the data in cache, so the kernels are limited by arithmetic throughput, not
   memory -- which is the regime where adding cores helps linearly. (Contrast the
   *streaming* reduce, which is memory-bound and caps at ~2x -- see
   [01_map_reduce_partitioned.md](Cases-Analysis-01-Map-Reduce-Partitioned) and
   [09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters).)

## Why it tapers past P=8->16

The machine has **16 physical cores / 32 logical** (SMT). Through 8 threads,
efficiency is >95% -- each thread gets a physical core. From 8->16 the efficiency
drops to 58% because threads start sharing physical cores' execution resources
(SMT gives far less than a 2x boost for compute-bound FP code) and contend for
shared cache/memory bandwidth. The Amdahl serial fraction of 0.041 says ~96% of
the work is perfectly parallel -- the taper is hardware, not algorithm.

## Why this is the *right* parallelism lever for dtfit

It is in-process (no IPC, no pickling, no spawn cost), so it has none of the
overhead that sinks the process-pool path ([08_fit_many_parallelism.md](Cases-Analysis-08-Fit-Many-Parallelism)).
For dtfit's workloads -- many small integral/projection calls inside an optimizer
or a streaming loop -- threaded GIL-released kernels are the cheapest way to use
all cores, and they compose with the map-reduce reduce (threads over partitions).

## Related

- Enables the threaded map-reduce in [01_map_reduce_partitioned.md](Cases-Analysis-01-Map-Reduce-Partitioned).
- The opposite outcome (overhead-bound) is [08_fit_many_parallelism.md](Cases-Analysis-08-Fit-Many-Parallelism).
