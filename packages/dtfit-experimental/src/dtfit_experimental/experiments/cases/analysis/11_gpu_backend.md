# Pluggable GPU backend (CuPy)

**Verdict: CONDITIONAL WIN — measured on an RTX 5080 (Blackwell sm_120).** The
GPU is decisive when data is **device-resident** (fp32: ~16× the CPU,
bandwidth-saturated) and gives **~nothing** for a single streaming pass over
host/disk data (PCIe-bound ≈ CPU). Use fp32 on consumer GPUs — their fp64 is
throttled.

Source: [`../../../../../../dtfit/src/dtfit/_core/_backend.py`](../../../../../../dtfit/src/dtfit/_core/_backend.py),
[`_batched.py`](../../../../../../dtfit/src/dtfit/scale/_batched.py).
Tested in: [GPU/batched throughput (8)](../08_gpu_batched_projection/08_gpu_batched_projection.ipynb),
[Fused multi-channel big data (10)](../10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb).
Install: `pip install cupy-cuda13x` (the `gpu` extra).

## What it is

A thin backend abstraction (`resolve_backend("auto"|"numpy"|"cupy"|"torch")`) that
only has to move arrays on/off a device; the projection itself uses generic
`@` / `*` / `.T`, so the GEMM-batched projection
([10_gemm_batched_projection.md](10_gemm_batched_projection.md)) runs unchanged on
NumPy/BLAS or on cuBLAS. **GPU support is a backend choice, not a rewrite.**

## Measured results (RTX 5080, 16 GB, sm_120; CUDA 13.2 driver)

Resident (data already on device) vs streamed (transferred per call), B=2048:

| backend | dtype | resident (ms) | streamed (ms) | streamed/resident | Melem/s (resident) |
|---|---|---|---|---|---|
| numpy | fp64 | 57 | 57 | 1.0× | 7,165 |
| numpy | fp32 | 31 | 36 | 1.2× | 13,071 |
| cupy | fp64 | 32 | 211 | 6.7× | 12,967 |
| **cupy** | **fp32** | **1.9** | 94 | **49×** | **212,956** |

- **fp32 resident: ~16× the fp32 CPU, ~30× the fp64 CPU.**
- fp32 resident achieves **~853 GB/s ≈ 88% of the card's GDDR7 peak**.
- fp64 resident: only **~2×** the CPU.
- streamed (any dtype): PCIe-bound (~17 GB/s) ≈ CPU memory bandwidth.

## Why it works only conditionally (the roofline argument)

The projection is a **low-arithmetic-intensity reduction** (~7 FLOPs per element).
For such an op, runtime is set by *where the data lives and how fast you can read
it*, not by compute. Three measured facts all follow from this:

1. **Resident fp32 wins by exactly the bandwidth ratio.** 853 GB/s on the GPU vs
   ~54 GB/s on the CPU is ~16×, and the measured speedup is ~16×. This is the
   textbook signature of a bandwidth-bound kernel: the GPU's only relevant
   advantage is its memory bandwidth, and the speedup equals the bandwidth ratio —
   no more, no less. (The kernel is *not* compute-limited or launch-limited at
   B=2048; it saturates HBM.)

2. **Streamed collapses to PCIe.** If `Y` must be copied host→device every call,
   every byte crosses PCIe (~16–32 GB/s) to do ~7 FLOPs. There is nothing to
   amortize the transfer against, so throughput is capped at PCIe bandwidth —
   which is *comparable to or below* CPU memory bandwidth. Hence streamed is 49×
   slower than resident and ≈ the CPU. **A single pass over out-of-core data sees
   no GPU benefit** — the transfer *is* the computation's cost.

3. **Consumer fp64 is hardware-throttled (~1/64).** GeForce cards gate double
   precision, so fp64 resident is only ~2× the CPU (compute-limited by the
   throttle, not bandwidth-limited). Switching to fp32 removes that gate and the
   kernel jumps to bandwidth-saturation. **On consumer GPUs, run the projection in
   fp32** (with safe chunked/partitioned accumulation to protect the sum).

## What this means for dtfit's big-data story

The GPU is an **accelerator for resident / many-channel projection**, not a
big-data solution. The primary big-data lever remains the exact partition-and-
reduce estimator ([01_map_reduce_partitioned.md](01_map_reduce_partitioned.md)),
which streams arbitrary volume in O(order) state — and *that* is precisely the
out-of-core, streamed regime where the GPU adds nothing. The two are
complementary: reduce to bring data down to a resident working set, accelerate the
resident batched projections on the GPU.

## Confirmed again by the fused estimator (Exp 10)

`PartitionedBatchLSI` ([12_fused_streaming_batched.md](12_fused_streaming_batched.md))
is backend-pluggable, so its per-chunk GEMM was run on the GPU directly — and it
**did not help** (measured 0.6–0.9× the CPU). This is the same roofline result
from the other side: a *streaming* estimator transfers each chunk host→device, so
it lives permanently in the PCIe-bound "streamed" column above. **Flat-memory
streaming and GPU-resident speed are mutually exclusive** — you cannot keep the
working set tiny *and* keep it resident. The GPU accelerates the resident batched
projection; the streaming reduce is a CPU job.

## When to use

- **Use** when the data is (or can be kept) **resident** on the GPU — generated on
  device, or reused across many projections/fits — and especially with **many
  channels** (large B) in **fp32**.
- **Don't bother** for a single streaming pass over host/disk data (PCIe-bound),
  or for fp64 on a consumer card (throttled). A data-center GPU with full-rate
  fp64 and faster interconnect (NVLink/PCIe 5) would shift these thresholds.

## Related

- Rides on the CPU-side GEMM reframe: [10_gemm_batched_projection.md](10_gemm_batched_projection.md).
- The complementary streamed big-data path: [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
