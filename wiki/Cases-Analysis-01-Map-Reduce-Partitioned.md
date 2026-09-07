# #1 -- Map-reduce / partitioned LSI & EAC

> **Status (2026-09):** The map-reduce estimators of the study, `PartitionedLSI`
> and `PartitionedEAC`, live in `dtfit_experimental.scale` for the notebooks; the
> use is covered by `ImageStream` (accumulator, `merge`). The sections below
> describe the study as it was run.

**Verdict: WORKS.** The only adaptation that cleared the >=2-domain gate, and the
deepest result in the suite -- it is not a heuristic but the linearity of
integration turned into an architecture.

Source: [`scale/_partitioned.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/scale/_partitioned.py).
Tested in: [Big-data (2)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/02_big_data_streaming/02_big_data_streaming.ipynb),
[Parallel (7)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/07_parallel_scaling/07_parallel_scaling.ipynb),
[GPU (8)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/08_gpu_batched_projection/08_gpu_batched_projection.ipynb),
[Fused multi-channel (10)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb).

## The fused multi-channel extension (`PartitionedBatchLSI`, Exp 10)

The partition (volume) reduce here handles **one channel at a time**. Fusing it
with the channel-batching GEMM ([10_gemm_batched_projection.md](Cases-Analysis-10-GEMM-Batched-Projection))
gives `PartitionedBatchLSI` -- *flat memory over volume **and** one matmul over
channels in one pass*, exact to machine precision, ~19x faster than looping this
estimator per channel. It is the same additive-projection fact extended along the
channel axis. **Full treatment, benchmarks and the surrogate-trap comparison:
[12_fused_streaming_batched.md](Cases-Analysis-12-Fused-Streaming-Batched).**

## What it is

LSI's empirical spectrum coefficient is an integral `beta_j = integral y.phi_j dx`, and an
EAC window area is `integral y dx`. Both are accumulated chunk-by-chunk into an
`O(order)` state vector and reduced by **plain addition**:

```
acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
for x_chunk, y_chunk in stream:     # one pass, fixed memory
    acc.update(x_chunk, y_chunk)
result = acc.fit(p0=[1.0, 1.0])     # solve the spectral match once, at the end
```

Workers each build an accumulator over their partition; `merge()` sums the
partial states. The model-side solve happens **once**, on the reduced `O(order)`
spectrum -- it never touches the raw data again.

## Measured results

**Big-data scaling (Exp 2):** streamed **10.5 GB (1.31 billion samples) in 119 s
at a flat 544 MB**, with throughput constant (~11 Msample/s) across an 8x volume
range:

| volume | samples | time (s) | peak mem (MB) |
|---|---|---|---|
| 0.5 GB | 62 M | 5.63 | 544 |
| 1 GB | 124 M | 11.30 | 544 |
| 2 GB | 250 M | 22.61 | 544 |
| 4 GB | 500 M | 45.92 | 544 |

Flat memory + linear time -> O(1)/sample; the linear fit extrapolates to **1 TB in
~192 min at the same 544 MB**.

**Exactness (unit tests):** the sequential reduce equals a single whole-domain
projection to **rtol 1e-9**, and the parallel `merge` is **associative** when
partitions share boundary samples.

**Parallel reduce (Exp 7):** threaded map-reduce reaches **2.06x at P=16** --
limited by memory bandwidth, not the algorithm.

## Why it works (the mechanism, in depth)

The reduce is **exact, not approximate**, and that is the whole point. Integration
is a linear functional, so over a partition `D = union D_k`:

```
integral_D y.phi_j dx  =  sum_k integral_{D_k} y.phi_j dx
```

The right-hand side is a sum of per-partition partial integrals. Three properties
follow, and each maps to a systems benefit:

- **Additivity -> distributable.** Partial sums combine by `+`, which is
  associative and commutative, so partitions can be processed in any order, on any
  number of workers/nodes, and reduced with a tree-reduction. There is no
  approximation error from splitting -- unlike, say, averaging per-chunk parameter
  estimates, which is *not* equivalent to a global fit.
- **`O(order)` state -> flat memory.** The accumulator is `order+1` numbers
  (~=7), independent of how many samples have streamed through. That is why peak
  memory is flat at 544 MB whether you process 0.5 GB or 10.5 GB -- the 544 MB is
  Python/NumPy working set for one chunk, not the data.
- **One solve at the end -> cost is in the reduce, not the fit.** The nonlinear
  spectral match is `order`-dimensional and runs once, so it is negligible; the
  whole cost is the single streaming pass that builds the spectrum.

The one subtlety the implementation gets right: a naive trapezoid over disjoint
chunks would **drop the interval connecting two chunks**. `update` carries the
previous chunk's last sample into the next (and `merge` assumes partitions share
their boundary sample), so the connecting interval belongs to exactly one
partition -- which is what makes the reduce *exact* rather than additive-up-to-a-
boundary-error.

## Why the parallel speedup is "only" ~2x

The 2.06x at P=16 (Exp 7) is **not** an algorithmic limit -- it is the roofline.
The per-chunk projection is a low-arithmetic-intensity reduction (~`order` FLOPs
per element), so a streaming pass saturates memory bandwidth long before it
saturates the cores. More threads cannot move bytes faster than the memory
controller. This is the same ceiling that caps the GPU on streamed data
([11_gpu_backend.md](Cases-Analysis-11-GPU-Backend)): the bottleneck is data movement, never
the additions.

## When to use / when not

- **Use** for any dataset too large for RAM, for distributed/multi-node fitting,
  and as the back-end for streaming aggregation -- it is the canonical big-data
  path.
- **Requires** the global domain fixed up front (so every chunk projects onto the
  same basis) and chunks fed in (or merged in) domain order.
- **Not** a substitute for an online *tracker* -- for time-varying parameters with
  drift, use the streaming filters ([09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters)).

## Related

- The projection it accumulates is now a GEMM -- see
  [10_gemm_batched_projection.md](Cases-Analysis-10-GEMM-Batched-Projection).
- Pairs with the GIL-released kernels for the threaded reduce
  ([07_gil_released_kernels.md](Cases-Analysis-07-GIL-Released-Kernels)).
