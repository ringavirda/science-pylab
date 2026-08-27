# Fused streaming + GEMM-batched LSI (`PartitionedBatchLSI`)

**Verdict: WORKS — the streaming multi-channel estimator.** It fuses the two
big-data levers that were separate before — the *volume* partition of
`PartitionedLSI` (flat O(channels·order) memory, exact one-pass reduce) and the
*channel* GEMM of `project_spectra` (one matmul over channels) — into a single
estimator that is **exact to machine precision**, **~19× faster than the
per-channel partitioned loop at the same flat memory**, and **backend-pluggable**.
It trails the whole-array single GEMM (the price of bounded memory) and the GPU
does *not* help its streaming path — both honest, both expected.

Source: [`../../../../../../dtfit/src/dtfit/scale/_partitioned.py`](../../../../../../dtfit/src/dtfit/scale/_partitioned.py)
(`PartitionedBatchLSI`), built on the new
[`_spectral.py`](../../../../../../dtfit/src/dtfit/_core/_spectral.py) primitive
`Basis.project_integral_batched` (raw additive integrals).
Tested in: [Fused multi-channel big data (10)](../10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb).
Extends [#1 map-reduce](01_map_reduce_partitioned.md) and the
[GEMM-batched projection](10_gemm_batched_projection.md).

## What it is

Before, the two big-data axes were handled by *separate* tools:

* `PartitionedLSI` reduced a huge stream in flat memory — but **one channel at a
  time**;
* `project_spectra` batched **many channels** into one GEMM — but needed the
  **whole volume in RAM** (O(N)).

`PartitionedBatchLSI` does both in one pass: each chunk's `B`-channel partial
integrals are a single backend GEMM `S = (w⊙D)ᵀ·Y_chunk`, folded into a
`(B, n_coef)` accumulator; `merge` reduces accumulators across workers; `fit`
solves each channel's small spectral match. The fusion is **exact** because the
projection is *linear across channels* and *additive over the domain* — the same
two facts that make the base estimator and the map-reduce work.

## Measured results (Exp 10)

**Exact, by construction.** The fused spectra match both references to machine
precision (it is the same projection, reorganised):

| check | max\|Δ\| |
|---|---|
| fused vs whole-array `project_spectra` | ~6e-15 → exact |
| fused vs per-channel `PartitionedLSI` | ~5e-15 → exact |

Recovered parameters equal the gold-standard per-channel `curve_fit` (median
\|Δa\|, \|Δb\| ~1e-4, noise-limited).

**Performance — the win and its honest cost.** Projecting a 40 k-sample stream of
`B` channels:

| | vs per-channel `PartitionedLSI` loop | vs whole-array `project_spectra` |
|---|---|---|
| throughput (B=1024) | **~19× faster** | ~6× slower |
| memory (10⁶×128) | same (flat ~29 MB) | **~3 GB → won't fit** |

So among the **streaming** (bounded-memory) options it is a large, free win —
one GEMM instead of a Python per-channel loop. It trails the whole-array single
GEMM because chunking pays per-chunk overhead (rebuilding the design, boundary
handling) — but the whole-array batch is **O(N) memory** and does not scale. The
gap is the price of bounded memory, tunable via chunk size.

**GPU does not accelerate the streaming path** (measured 0.6–0.9× on the RTX
5080): per-chunk projection is a fresh host→device transfer each chunk, i.e.
exactly the PCIe-bound *streamed* regime of [11_gpu_backend.md](11_gpu_backend.md).
Flat-memory streaming and GPU-resident speed are **mutually exclusive** — the GPU
helps only when `Y` is already resident (Exp 8's ~16× fp32), which means *not*
streaming. The fused estimator is therefore the **CPU streaming** tool;
GPU-resident batch projection is the separate operating point.

## The surrogate trap (vs external standard approaches)

The most instructive comparison is against the methods a practitioner would
actually use instead. Fit `B` channels on the first 70% of the domain, predict the
held-out last 30%:

| approach | recovers | in-window R² | extrapolation R² | streaming? |
|---|---|---|---|---|
| fused `PartitionedBatchLSI` (exp) | **physical a, b** | 0.891 | **0.726** | yes — flat mem |
| per-channel `curve_fit` (NLLS) | physical a, b | 0.891 | 0.726 | no — O(N) |
| vectorised polynomial `lstsq` (deg 6) | surrogate coeffs | 0.891 | **−0.292** | no — O(N) |
| per-channel `polyfit` loop | surrogate coeffs | 0.891 | −0.292 | no — O(N) |

**In-window R² is identical for every method** — the fast batched polynomial
*looks* exactly as good if you only check the fit window. That is the trap. The
discriminator is **extrapolation**: the structured (exp) fits hold R²≈0.73 while
the polynomial surrogate collapses to **−0.29** (worse than predicting the mean —
a degree-6 polynomial diverges outside its window) and recovers **no physical
parameters**. The one external method that *does* recover the physics
(`curve_fit`) neither batches nor streams, and is ~3× slower at identical
accuracy. So:

> The vectorised polynomial `lstsq` is the fastest, but it solves a different,
> easier problem (a non-extrapolating surrogate). `curve_fit` recovers the physics
> but does not scale. **The fused estimator is the only one delivering
> nonlinear-physical + batched + streaming together** — which is the operational
> meaning of "dtfit recovers a *known* nonlinear-in-parameters model."

## Why it works (the reasoning)

1. **Linearity across channels → one GEMM.** Stacking channels into the columns of
   `Y` makes the per-chunk projection a single matrix product, handing the work to
   BLAS/cuBLAS instead of a Python loop. This is the ~19× over the loop.
2. **Additivity over the domain → exact streaming reduce.** The raw integrals
   `s_{c,j} = ∫ y_c·φ_j` sum over a partition, so chunk-wise accumulation (and
   cross-worker `merge`) equals a single whole-domain projection — bit-for-bit.
   The new `project_integral_batched` returns the *un-normalized* integrals
   precisely so they remain additive (the per-coefficient norm is applied once, at
   the end).
3. **The whole-array gap is overhead, not algorithm.** Both do the same GEMM
   FLOPs; chunking just rebuilds the small design and copies boundary samples once
   per chunk. Bigger chunks close the gap at the cost of memory — the knob that
   trades the two.

## When to use

- **Use** for a *massive multi-channel* dataset on a **shared sampling grid**
  (panel / sensor-array / multivariate streams) **too big for RAM**, where you
  want every channel's structured parametric fit in one pass. It is the streaming
  multi-channel estimator: the per-channel loop's flat memory with most of the
  batched GEMM's speed.
- **Don't bother** when the data *fits* in RAM (the whole-array batch, CPU or
  GPU-resident per Exp 8, is faster), or for a single channel (no channel axis to
  batch — use `PartitionedLSI`).

## Honest limits & status

- Needs a **shared sampling grid** and the **global domain fixed up front** (every
  chunk/worker projects onto the same basis); heterogeneous grids fall back to
  independent fits.
- A 10⁹-element reduction may want a compensated (Kahan) accumulator; the bounded
  per-chunk GEMM is fine at fp32.
- A **single-domain** result so far, so by the suite's ≥2-domain promotion gate it
  stays in `dtfit.adaptations` (experimental), **not yet promoted** to the
  top-level API — unlike its parent `PartitionedLSI`.

## Related

- Parent estimator (promoted): [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
- The channel-batching GEMM it reuses: [10_gemm_batched_projection.md](10_gemm_batched_projection.md).
- Why the GPU does not help its streaming path: [11_gpu_backend.md](11_gpu_backend.md).
