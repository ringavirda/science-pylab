# dtfit feature analysis

Per-feature deep-dives into every optimization and structural adaptation tested
across the experiment suite (`experiments/01_...` through `10_...`). Each file
answers the same four questions -- **what it is, where it was tested (with the
measured numbers), why it works or doesn't, and the verdict** -- and pushes on the
*mechanistic* "why" rather than just tabulating outcomes.

This complements [`../REPORTS.md`](Cases-Reports): REPORTS.md is the per-experiment
index + the promotion matrix; this folder is organised **per feature**, tracing
each one across all the experiments that exercised it.

## Structural adaptations (new ways to compose LSI / EAC)

| file | adaptation | verdict |
|---|---|---|
| [01_map_reduce_partitioned.md](Cases-Analysis-01-Map-Reduce-Partitioned) | #1 map-reduce / partitioned LSI*EAC | In `dtfit_experimental.scale` -- covered by `ImageStream` (accumulator, `merge`) |
| [02_pluggable_basis.md](Cases-Analysis-02-Pluggable-Basis) | #2 pluggable orthogonal basis | Experimental -- vocabulary, not power |
| [03_overlapping_ensemble.md](Cases-Analysis-03-Overlapping-Ensemble) | #3 overlapping-window ensemble | **Promoted** (`dtfit.ensemble_fit`) -- the whole-window-rejection complement to `fit_eac`'s robust image |
| [04_joint_multichannel.md](Cases-Analysis-04-Joint-Multichannel) | #4 joint shared-parameter fit | Experimental -- loss where tested |
| [05_stagewise_boosting.md](Cases-Analysis-05-Stagewise-Boosting) | #5 stage-wise boosting | Experimental -- win, 1 domain |
| [06_adaptive_window_eac.md](Cases-Analysis-06-Adaptive-Window-EAC) | #6 adaptive-window EAC | Retired -- `fit_eac` places equal windows |

## Performance & parallelization optimizations

| file | optimization | verdict |
|---|---|---|
| [07_gil_released_kernels.md](Cases-Analysis-07-GIL-Released-Kernels) | GIL-released C kernels (threads) | **Win** |
| [08_fit_many_parallelism.md](Cases-Analysis-08-Fit-Many-Parallelism) | `fit_many` process/thread fan-out | Mixed -- coarse-only |
| [09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters) | recursive O(1)/sample filters | **Win** |
| [10_gemm_batched_projection.md](Cases-Analysis-10-GEMM-Batched-Projection) | GEMM-batched projection (CPU/BLAS) | In `dtfit_experimental.scale` -- channel axis of `ImageStream` |
| [11_gpu_backend.md](Cases-Analysis-11-GPU-Backend) | pluggable GPU backend (CuPy) | Conditional win |
| [12_fused_streaming_batched.md](Cases-Analysis-12-Fused-Streaming-Batched) | fused streaming + GEMM-batched LSI (`PartitionedBatchLSI`) | In `dtfit_experimental.scale` -- `ImageStream` with `channels` |

The streaming filters' **embedded footprint** (Exp 9 -- fixed sub-KB C struct,
MCU-deployable) and the **fused maneuver detector** (Exp 5) are covered in
[09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters).

## The two facts that explain every result

Almost every verdict below is a corollary of two structural properties of dtfit:

1. **dtfit recovers a *known* nonlinear-in-parameters model.** It wins where
   structure is known and physical (control ID, transients, exponential growth,
   clean cycles, structured trajectory forecasting) and loses where the
   predictable signal is *global stochastic structure learned across many
   records* (LTSF) or the series is irregular. The operational payoff -- made
   concrete by the **surrogate trap** in Exp 10 -- is that a structured fit
   **extrapolates and yields physical parameters**, where a polynomial surrogate
   with *identical in-window R^2* extrapolates to negative R^2 and gives no physics.
   "Recovers a known model" is not cosmetic; it is what survives outside the fit
   window.
2. **dtfit's computation is integration -- linear and additive.** This makes the
   map-reduce reduce *exact* (#1), makes the kernels embarrassingly parallel once
   the GIL is released, and makes the projection a GEMM -- but the same low
   arithmetic intensity is why the GPU helps only on resident data, and why
   outlier robustness is a reweighting of the image rather than an ensemble
   over windows.

A useful pattern emerges: **the adaptations that worked exploit the math
(additivity, structural composition); the ones that disappointed tried to add
statistical power the data did not contain.**

## Data-quality caveats (read before citing)

- **GPS (Exp 5) -- fixed, made realistic, then improved.** It had two problems:
  dtfit metrics came out `nan` (a `nan`-poisoning filter bug, now fixed by a
  non-finite guard), and the trajectory handed dtfit the *exact* generating model
  (best-case, ~22x "win"). The experiment was redesigned around a genuinely
  **maneuvering** target tracked with a **generic** CA model. Honest result
  (mean over 12 noise realizations): dtfit is **competitive with, slightly behind,
  the gold-standard Kalman** (smoothing 1.30 vs 1.18 km) -- the structural-model
  advantage only appears when the model is genuinely known. A new **fused
  multi-axis maneuver detector** roughly *doubles* maneuver detection (~0.6 -> ~1.2
  of 3 onsets), but acting on it improves tracking only marginally (and *hurts* the
  already-optimal Kalman) -- the ceiling is measurement SNR, not the algorithm. See
  [09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters). GPS does **not** exercise the
  joint fit (#4) -- see [04_joint_multichannel.md](Cases-Analysis-04-Joint-Multichannel).
- **Embedded footprint (Exp 9 -- new).** The streaming filter's deployable state is
  a fixed sub-KB C struct (~= 636 B float32 for a 3-axis tracker) that fits common
  microcontrollers; compute is never the bottleneck at GPS rates. Folded into
  [09_streaming_filters.md](Cases-Analysis-09-Streaming-Filters).
- **Fused streaming + batched LSI (Exp 10 -- new).** `PartitionedBatchLSI` fuses the
  volume partition (#1) and the channel GEMM into one exact streaming estimator;
  ~19x over the per-channel partitioned loop at flat memory, trails the (O(N))
  whole-array GEMM, and the GPU does not help its streaming path. See
  [12_fused_streaming_batched.md](Cases-Analysis-12-Fused-Streaming-Batched). It is a
  **single-domain** result; it lives in `dtfit_experimental.scale`, covered by
  `ImageStream` with `channels`.
- #5 is a **single-domain win** -- promising, not promotion-eligible under
  the >=2-domain gate.
- The `fit_many` negative is **platform-specific** (Windows process spawn).
