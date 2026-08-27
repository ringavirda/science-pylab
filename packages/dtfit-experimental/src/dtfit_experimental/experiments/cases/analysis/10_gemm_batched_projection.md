# GEMM-batched projection (CPU / BLAS)

**Verdict: WORKS.** Expressing the LSI/EAC projection as one matrix product over
many channels turns a Python per-channel loop into a single BLAS GEMM — up to
~340× faster on CPU, exact to machine precision, and the form that also unlocks
the GPU.

Source: [`../../../../../../dtfit/src/dtfit/scale/_batched.py`](../../../../../../dtfit/src/dtfit/scale/_batched.py),
[`_spectral.py`](../../../../../../dtfit/src/dtfit/_core/_spectral.py) (`_gemm_factors`).
Tested in: [GPU/batched throughput (8)](../08_gpu_batched_projection/08_gpu_batched_projection.ipynb),
[Fused multi-channel big data (10)](../10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb).

## What it is

The projection `β_j = ∫ y·φ_j dx` factors exactly as a matrix product
`β = Dᵀ·(w⊙y)` — design matrix `D`, trapezoid quadrature weights `w` folded in.
Stacking B channels that share a sampling grid into the columns of `Y` makes the
whole batch **one GEMM**: `S = (w⊙D)ᵀ·Y`. Exposed as `fit_lsi_batched(x, Y, …)`
and `project_spectra(x, Y, …)`.

## Measured results

Projecting B channels (N = 200 k samples each), per-channel loop vs batched GEMM
(Exp 8):

| channels B | loop (ms) | batched (ms) | speedup × | max\|Δ\| |
|---|---|---|---|---|
| 1 | 8.4 | 9.4 | 0.9 | 8e-17 |
| 8 | 67.6 | 10.6 | 6.4 | 2e-14 |
| 64 | 617 | 14.1 | ~44 | 8e-15 |
| 512 | 5390 | 28.8 | ~187 | 2e-15 |
| 2048 | 25408 | 75.0 | **~340** | 2e-15 |

Peak ~6000 Melem/s; **bit-exact vs the loop** (max\|Δ\| ≈ machine epsilon).
fp32 ~1.6× fp64 (fewer bytes moved). Throughput plateaus near the host
memory-copy rate (~50 GB/s) → **bandwidth-bound**.

## Why it works (the reasoning)

Three compounding effects, all genuine and all exact:

1. **Build the basis once, not per channel.** The naive way to project many
   channels is to loop `project_integral` per channel, which rebuilds the
   `(N×k)` Vandermonde *every call*. Batching builds `D` once and reuses it across
   all B channels — most of the 340× at large B is this amortization.

2. **One BLAS GEMM instead of B GEMVs.** A single `Dᵀ·Y` hands the work to
   multithreaded, cache-blocked, vectorized BLAS, which is far more efficient than
   B separate small matrix-vector products dispatched from Python (each with
   interpreter + dispatch overhead).

3. **Fold the weights into the *small* matrix.** The critical implementation
   detail: weight `D` (which is `N×k`, k≈7) rather than `Y` (which is `N×B`). The
   factoring `Dᵀ·(w⊙Y) = (w⊙D)ᵀ·Y` is algebraically identical but avoids
   materializing a multi-GB `(N×B)` temporary — without this, end-to-end
   throughput was ~7× lower (a temporary that dominated the runtime). This was
   found *by benchmarking*, not by theory.

It is **exact** because `Dᵀ·(w⊙y)` is the same trapezoid sum as `np.trapezoid`,
just reassociated — so the promoted `PartitionedLSI`, which now shares this
factoring, is numerically unchanged (verified to 1e-9 in tests).

## The ceiling: bandwidth, not FLOPs

The op is a low-arithmetic-intensity reduction (~k ≈ 7 FLOPs per element read).
So once dispatch overhead is amortized, throughput is capped by memory bandwidth,
plateauing at the host copy rate. This is *why* fp32 helps (half the bytes) and
*why* the GPU only helps when data is resident — the same roofline governs both
([11_gpu_backend.md](11_gpu_backend.md)).

## Fused with the partition: streaming multi-channel (Exp 10)

The whole-array form here needs `Y` in RAM (O(N)). Because the projection is *also*
additive over the domain, it composes with the map-reduce partition into
`PartitionedBatchLSI`: each chunk's `B`-channel projection is one GEMM folded into
an accumulator, giving **flat memory over volume *and* the channel-batched GEMM in
one pass**. Exp 10 measured it ~19× faster than looping `PartitionedLSI` per
channel (same flat memory), trailing this whole-array GEMM only by the per-chunk
overhead — the price of bounded memory. It also surfaced the **surrogate trap**: a
vectorised polynomial `lstsq` (the fast external alternative) matches in-window R²
but extrapolates to negative R² and recovers no physical parameters, where the
structured fit extrapolates and yields the physics. Full treatment:
[12_fused_streaming_batched.md](12_fused_streaming_batched.md).

## When to use

- **Use** whenever you fit/project **many channels or windows on a shared grid**
  (multi-channel sensors, LTSF-style panels, batched re-projection) — the more
  channels, the bigger the win. For volumes that exceed RAM, use the streaming
  fusion ([12](12_fused_streaming_batched.md)) instead.
- **Little benefit** for a single channel (B=1 is ~parity with the loop — the
  GEMM has nothing to amortize over).

## Related

- The same factoring is what `PartitionedLSI` accumulates:
  [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
- Fused with the partition for streaming multi-channel:
  [12_fused_streaming_batched.md](12_fused_streaming_batched.md).
- The GPU backend rides on this GEMM form: [11_gpu_backend.md](11_gpu_backend.md).
