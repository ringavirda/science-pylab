# Streaming filters — recursive O(1)/sample tracking

**Verdict: WORKS — the real-time win, in its niche.** Constant per-sample cost at
bounded, **embeddable** memory with drift detection — capabilities batch methods
structurally lack (shown cleanly in Exp 1 & 2). On a *realistic maneuvering* GPS
target with a generic model it is **competitive with, not better than, the
gold-standard Kalman** (see below); a **fused multi-axis maneuver detector** added
here roughly doubles maneuver detection, but the measurement-SNR ceiling means
that does not translate into beating the Kalman. The streaming filter's value is
real-time/streaming operation, change detection, and a tiny fixed footprint that
fits a microcontroller (Exp 9) — not raw accuracy superiority.

Source: [`../../../../../../dtfit/src/dtfit/streaming/`](../../../../../../dtfit/src/dtfit/streaming/)
(`EACFilter`, `LSIFilter`, `FilterBank`).
Tested in: [Control (1)](../01_control_systems/01_control_systems.ipynb),
[Big-data (2)](../02_big_data_streaming/02_big_data_streaming.ipynb),
[GPS (5)](../05_gps_trajectory/05_gps_trajectory.ipynb),
[Embedded footprint (9)](../09_embedded_footprint/09_embedded_footprint.ipynb).

## What it is

Online counterparts of the batch fits: a recursive estimator that updates its
parameter estimate **per sample** in bounded memory, with a NIS-based drift
detector that flags structural breaks. `FilterBank` runs K of them in parallel
(one per channel / axis / satellite stream).

## Measured results

**Constant cost, bounded memory (Exp 2):**

| updater | cost | scaling | memory |
|---|---|---|---|
| **EACFilter (online)** | **103.1 µs/sample** | **O(1)/sample** | **11.7 MB (bounded)** |
| batch re-fit, 10 k samples | 2.8 ms/refit | O(N) | O(N) |
| batch re-fit, 50 k samples | 5.0 ms/refit | O(N) | O(N) |
| batch re-fit, 250 k samples | 20.3 ms/refit | O(N) | O(N) |

**Drift tracking (Exp 1):** the filter tracked a mid-run damping jump
(ζ: 0.08 → 0.30) and **flagged 1 structural break** — online, in a single pass.

## Why it works (the mechanism)

1. **Recursive update = O(1) state, O(1) time.** The filter keeps only the running
   sufficient statistics (areas / spectral accumulators) and updates them with each
   new sample, then re-solves a tiny `order`-dimensional system. Nothing grows with
   the history length, so per-sample cost and memory are flat — the defining
   property of a real-time estimator.

2. **Re-fitting is quadratic; tracking is linear.** Tracking a stream by *batch
   re-fitting* a growing window costs O(N) per refit, and doing that every step is
   **O(N²)** with O(N) memory — it falls over as the stream grows (the table shows
   2.8 → 20.3 ms as the window grows). The recursive filter is O(1)/sample by
   construction. This is the structural reason batch methods (and batch NNs/ARIMA
   that need the whole array) cannot do real-time tracking at scale.

3. **Drift detection comes for free from the residual.** The filter already
   computes a normalized innovation (how surprised it is by each sample); a
   sustained spike is a structural break. So it *adapts* to regime change rather
   than silently averaging across it — something a single batch fit cannot express.

## Relationship to the map-reduce estimator

These are complementary, not competing:
- **`PartitionedLSI`** ([01_map_reduce_partitioned.md](01_map_reduce_partitioned.md))
  is for a *stationary* parameter over a huge dataset — exact one-pass reduce.
- **Streaming filters** are for a *time-varying* parameter — they deliberately
  forget old data (so they can follow drift) and flag breaks.
Choose by whether the thing you're estimating is constant (reduce) or moving
(filter).

## The GPS result (resolved, then improved)

In the GPS experiment (5) the `FilterBank` is used two ways:
- **Per-satellite range smoothing — a documented negative:** smoothing each
  pseudorange stream independently *before* multilateration **degraded the fix
  from 3.37 km to ~350 km**. Why: multilateration needs the satellite ranges at one
  instant to be mutually *consistent*; smoothing each stream with its own lag
  destroys that synchrony, so the geometry solve diverges. The lesson — smooth at
  the **trajectory level, not the raw-range level** — is correct and valuable.
- **Per-axis trajectory tracking — competitive with Kalman (realistic test):**
  the experiment was *re-made realistic* — the target now flies a genuinely
  maneuvering path (coordinated turns with changing heading/speed/climb), and
  dtfit is fitted with a **generic constant-acceleration model**, not the exact
  generating functions. Averaged over 12 noise realizations (a single seed proved
  unrepresentative): dtfit in-track smoothing **1.30 km** vs Kalman **1.18 km** —
  **dtfit on par with, slightly behind, the gold-standard Kalman; neither
  dominates.** Both streaming measurements were tried: **EAC (area) and LSI
  (Legendre spectrum) come out essentially tied** (EAC 1.30 vs LSI 1.33 km
  smoothing; LSI a touch better on detection, 1.4 vs 1.2/3). LSI's edge is
  resolving frequency/phase/shape, and a CA quadratic over a short window has none
  — so the richer spectral measurement buys nothing here (it would help on an
  oscillatory target, not a smooth trajectory).

  > **An earlier version of this experiment reported a ~22× forecast win — that
  > was an artifact** of handing dtfit the exact per-axis generating functions
  > (`linear / sine / exponential`). Removing that best-case setup removed the
  > gap. The real lesson: dtfit's structural-model advantage is decisive **only
  > when the model is genuinely known** (see Exp 1, where the physics *is* the
  > model); as a generic kinematic tracker it matches, not beats, the Kalman.

### The fused maneuver detector (the improvement)

The original per-axis area-innovation detector caught essentially **none** of the
maneuvers (~0.6/3). The diagnosis was twofold: (i) the integrated full-window
*area* statistic smooths the brief onset transient away, and (ii) per single axis
the onset is only ~1.4–3.2× the fix noise — and the noisiest axis (z) also has the
largest onset, so a per-axis detector is unreliable. **The fix exploits that a
maneuver moves all axes at once:** fuse the three axes' one-step forecast
residuals into a single χ²(3) NIS statistic and run a CUSUM on it. Fused, the
maneuver reaches **~4× baseline and is consistent across all onsets**, so the
detector catches **~1.2/3 — roughly double**. (The first onset, t=3, sits on the
filter's own convergence transient and stays hard — an honest startup/SNR limit.)

This is built on two small, reusable library primitives added to both filters:
`last_residual_` (the exposed one-step innovation) and a public `inflate()`
(covariance re-arming hook for an external detector). The fused detector itself
lives in the experiment as a prototype — demonstrated on one domain, so it has
**not** cleared the ≥2-domain promotion gate.

**Does acting on detection help tracking?** Only in a narrow regime — the most
important honest finding. A *gentle* covariance nudge (×3) on a flag turns the
better detection into a small tracking gain for dtfit (1.34 → 1.30 km), because
plain dtfit is slightly *under*-reactive and has headroom. The **identical** nudge
applied to the Kalman *hurts* it (1.18 → 1.22 km): it is already near-optimal, so
extra inflation just adds variance. Aggressive re-arming (×100) hurts both. **The
ceiling is measurement SNR, not the algorithm**: these acceleration-level
maneuvers sit barely above the ~3 km fix noise, so neither sharper detection nor
adaptation overtakes the fixed Kalman, and the adaptive Kalman cannot beat the
fixed one. Reliable maneuver detection from position alone is near the information
limit — real systems fuse an independent sensor (IMU/gyro), where the maneuver is
obvious in acceleration but buried in position.

### Embedded footprint (Exp 9)

The streaming filter is the only embeddable part of dtfit, and Experiment 9 sizes
it for the microcontroller you would attach to a GPS module. The deployable state
is a **fixed-size, no-malloc C struct** — `2W + n² + 2n + 8` words — that does
*not* grow with stream length: a 3-axis tracker is **≈ 636 B (float32)**, fitting
comfortably on a Cortex-M0+/M4/ESP32 and feasible even on a 2 KB AVR. At a few
thousand FLOPs per epoch against a 1–10 Hz GPS rate, **compute is never the
bottleneck** — memory and float discipline are. Honest caveat: NumPy does not run
on an MCU, so the per-sample latency (~34 µs desktop) is a reference for the
algorithm shape while the deployable artifact is a hand-coded C recurrence (the
integration kernels are already C in `dtfit._native`); the memory verdict, which
usually decides feasibility, is exact.

### A library robustness bug found along the way (the durable fix)

The earlier matched-model version of this experiment surfaced a real
streaming-filter bug worth keeping on record. Its climb axis used
`z0 + c·(1 − e^(−t/τ))`, which has a **singularity at τ → 0**; the online estimate
wandered toward it, `exp(−t/τ)` overflowed, and the non-finite innovation was
committed straight into the EKF state — after which **every** later `predict()`
returned `nan` (one bad sample permanently poisoned the filter).

The **durable fix** (still in place, regression-tested): both `EACFilter`
and `LSIFilter` now **reject a non-finite update** — if the
innovation/Jacobian or the candidate `(p, P)` is not finite, the sample is skipped
and the last good estimate kept. A streaming EKF should never be able to
NaN-poison itself, regardless of which model it is fitting. (The matched-model
version that triggered it has since been replaced wholesale by the realistic
generic-model design above, so the singular `exp(−t/τ)` form is no longer used
here at all.)

## When to use

- **Use** for real-time tracking, drift/anomaly detection, any stream where
  parameters move over time, and **resource-constrained / embedded** deployment
  (the fixed sub-KB footprint of Exp 9).
- **For maneuver/change detection over several streams**, fuse the per-stream
  innovations (`last_residual_`) into one χ²(K) statistic rather than testing each
  stream alone — and re-arm gently (`inflate()` with a small factor): aggressive
  re-arming, or any re-arming of an already-optimal estimator, costs more variance
  than it saves.
- **Don't** use for a stationary parameter over a fixed dataset (use the exact
  reduce), and **don't** apply per-stream smoothing upstream of a solver that
  needs cross-stream synchrony (the GPS multilateration lesson).

## Related

- Stationary-parameter counterpart: [01_map_reduce_partitioned.md](01_map_reduce_partitioned.md).
- Joint multi-channel fit (#4) — *not* exercised by GPS (axes share no parameter);
  its weak-data retest remains open: [04_joint_multichannel.md](04_joint_multichannel.md).
