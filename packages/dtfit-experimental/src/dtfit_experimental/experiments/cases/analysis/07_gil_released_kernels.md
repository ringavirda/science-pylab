# GIL-released kernels (historical) / GIL-bound numpy kernel (current)

**Verdict: INVERTED by the image-core migration.** The clean parallelization win this
page used to report came from a compiled C kernel (`dtfit._core._native`) that has
since been removed -- it had no production caller once the batch fit and the filters
moved onto numpy-based projections (image-core step 7). The numpy replacement that
took its place, `_kernels.simpson_windows`, is a plain Python loop over
`scipy.integrate.simpson` calls on small window slices. It does not release the GIL
for this workload, so a thread pool buys nothing: throughput is flat to slightly
*worse* as threads are added.

Source: [`../../../../../../dtfit/src/dtfit/_core/_kernels.py`](../../../../../../dtfit/src/dtfit/_core/_kernels.py)
(`simpson_windows`, `simpson_windows_rows`, `legendre_project`).
Tested in: [Parallel scaling (7)](../07_parallel_scaling/07_parallel_scaling.ipynb).

## What it was

The three hot kernels used to be compiled C over raw pointers, refactored so that
argument validation, bounds checks and output allocation happened while the GIL was
held (they touch Python objects), and the pure compute loop ran inside
`Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` (it touched no Python object). A
thread pool could then run many kernel calls truly concurrently.

That kernel is gone. `dtfit._core._native` had no caller left in the library once the
image-core migration routed the batch fit and the streaming filters through numpy
projections, so it was removed rather than kept as unused C. The numpy bodies that
replaced it (`simpson_windows`, `simpson_windows_rows`, `legendre_project`) are the
ones the library actually uses now.

## What it is now

`simpson_windows` builds each window's integral with a Python `for`/list-comprehension
loop over `scipy.integrate.simpson`, one call per window. Each call is short and
mostly Python-level bookkeeping (slicing, dispatch) around a small numeric kernel, so
the interpreter holds the GIL for most of the wall time. There is no equivalent of the
old `Py_BEGIN_ALLOW_THREADS` region: the loop itself, not just the arithmetic inside
one `simpson()` call, is what needs to run concurrently for threading to pay off, and
it does not.

## Measured results

**Historical (compiled C kernel, removed):** each of P threads ran a fixed batch of
native Simpson calls on cache-resident data.

| threads P | throughput x | efficiency % |
|---|---|---|
| 1 | 1.00 | 100 |
| 2 | 1.98 | 99 |
| 4 | 3.84 | 96 |
| 8 | 7.57 | 95 |
| 16 | 9.35 | 58 |

Peak 9.3x at P=16, >95% efficiency through 8 cores, Amdahl serial fraction s = 0.041.
These numbers describe a kernel that no longer exists; they are kept here for the
record and as the ceiling a compiled kernel could recover.

**Current (numpy `simpson_windows`, same benchmark shape, Exp 7):**

| threads P | throughput x | efficiency % |
|---|---|---|
| 1 | 1.00 | 100 |
| 2 | 0.95 | 47 |
| 4 | 0.93 | 23 |
| 8 | 0.93 | 12 |

Adding threads makes it flat to slightly *slower*, not faster: pure threading and
contention overhead with no real concurrency underneath. Efficiency collapses because
the numerator (throughput) is pinned near 1x while the denominator (P) grows -- there
is no scaling for it to measure.

## Why it inverted

1. **The GIL was the only thing serializing the old threads, and it was dropped on
   purpose.** The C loops were arithmetic over `double*` buffers with no reference
   counting or Python object access inside the hot loop, so releasing the GIL around
   them was safe and let the cores run independently.

2. **The numpy replacement never drops that lock for this workload.** Each
   `simpson_windows` call does ~200 small `scipy.integrate.simpson` calls in a Python
   loop; the slicing, dispatch and small-array overhead between calls all run under
   the GIL. Threads take turns rather than overlapping, and the thread-pool bookkeeping
   on top makes it a hair slower than running the same work on one thread.

3. **This was a workload the C kernel was built for, not something numpy accidentally
   lost.** The compute-bound, cache-resident regime that made GIL release pay off
   assumed a genuinely compiled inner loop. Swapping in a Python-level loop over a
   numpy/scipy call removes that assumption; the benchmark did not change, the kernel
   under it did.

## What still holds elsewhere

The GIL-release story is not dead everywhere, only for this specific kernel:

- The threaded map-reduce in [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md)
  still scales, because its bulk array ops (large-array numpy arithmetic, not a
  Python loop over many small calls) do drop the GIL. That benchmark is unaffected by
  this removal.
- Bringing the compiled-kernel win back for `simpson_windows` would need either a real
  C/Cython loop over the whole window batch (not a Python loop calling scipy per
  window) or vectorizing the windows into fewer, larger numpy calls so the per-call
  Python overhead stops dominating. Neither has been done; this page records the
  current, unaccelerated state.

## Related

- Map-reduce over partitions, unaffected by this: [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
- The opposite overhead-bound outcome (process pool): [08_fit_many_parallelism.md](08_fit_many_parallelism.md).
