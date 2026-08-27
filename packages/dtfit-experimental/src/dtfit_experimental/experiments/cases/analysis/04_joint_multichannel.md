# #4 — Joint shared-parameter multi-channel fit

**Verdict: EXPERIMENTAL — a loss where it was tested.** On cleanly-identifiable
channels the dedicated per-channel solver already wins; coupling buys *parameter
parsimony and enforced consistency*, not accuracy. The case where it *should*
help (genuinely weak per-channel data) was not cleanly demonstrated.

Source: [`../../../joint.py`](../../../joint.py).
Tested in: [Control MIMO (1)](../01_control_systems/01_control_systems.ipynb),
[GPS (5)](../05_gps_trajectory/05_gps_trajectory.ipynb) (intended retest — inconclusive, see caveat).

## What it is

Stack several channels' spectra/areas into **one system** with **shared**
parameters (common across channels) plus **private** parameters (per channel).
E.g. a MIMO plant whose outputs share one natural frequency `ω` but each have
their own amplitude and damping. The promise: coupling makes a shared parameter
*more observable* than fitting each channel alone.

## Measured results

3-output plant sharing one `ω` (truth `ω=3.0`; amplitudes 1, 2, 3) — Exp 1:

| estimator | shared ω | ω err % | amplitudes |
|---|---|---|---|
| **joint (#4)** | 3.327 | **10.90** | 1.12, 2.21, 3.30 |
| independent EAC (mean) | 2.996 | **0.12** | (per-channel) |

→ Independent per-channel EAC recovered `ω` to **0.12%**; the joint fit to only
**10.9%** — two orders of magnitude worse on the shared parameter it was supposed
to nail.

## Why it loses here (the reasoning)

The value of parameter coupling is entirely a question of **identifiability**, and
this test was in the wrong regime for it:

1. **Coupling only adds information when per-channel information is scarce.**
   Pooling channels to estimate a shared `ω` helps when *no single channel*
   constrains `ω` well (short records, low SNR, a parameter that is nearly
   unidentifiable alone). Here every channel is a clean, well-sampled damped
   sinusoid that *already* pins `ω` to 0.12%. There is no information deficit for
   the coupling to fill, so it can only match the per-channel solver at best.

2. **The joint objective is coarser.** The implementation matches **window areas**
   across channels in one least-squares system; that area-matching is a blunter
   criterion than the dedicated bounded solver each channel gets on its own. With
   nothing to gain from pooling, the coarser objective simply loses accuracy —
   the 10.9% is the cost of the coarser fit, not a benefit of coupling.

3. **What it *does* buy is real but wasn't the bar.** One shared `ω` for all
   channels is *guaranteed consistent* (you cannot get three slightly different
   frequencies for one physical resonance) and halves the parameter count. That is
   genuinely valuable for interpretability and for downstream control design — but
   the experiment scored *accuracy*, where it had no edge.

## The unfinished test

The weak-per-channel-data regime — where coupling *should* help — has **still not
been tested**, and this is a subtler gap than it first looks. The GPS experiment
(5) was nominally the retest, but two things are true:

1. Its dtfit metrics used to be `nan`; that bug is now **fixed** (see
   [09_streaming_filters.md](09_streaming_filters.md)). So GPS produces real
   numbers again.
2. **But GPS does not actually exercise `fit_joint`.** Its per-axis trajectory
   path uses *independent* `EACFilter`s (one per axis), not a joint
   shared-parameter fit — the axes don't share a parameter, so there is nothing
   to couple. The experiment's mention of #4 is aspirational, not realized.

So the verdict stands: **#4 has only been measured where it loses (clean,
strongly-identifying channels), and never where it should win (genuinely weak,
parameter-sharing channels).** A fair test would construct several short/noisy
channels that share one parameter (e.g. low-SNR oscillators with a common
frequency, short records) and compare `fit_joint` against per-channel fits — the
mixed-effects regime where pooling is provably more efficient.

## When it would help

- Channels that **share a physical parameter** *and* are each individually
  **weakly identifying** it (short/noisy records) — the classic
  random/mixed-effects setting, where pooling is provably more efficient.
- Whenever **enforced consistency** matters more than the last digit of accuracy
  (one frequency for one resonance, one decay constant for one material).

## Related

- The clean per-channel result it was measured against is in the
  [control report](../01_control_systems/01_control_systems.ipynb).
- Its intended weak-data retest depends on fixing the GPS `nan`
  ([09_streaming_filters.md](09_streaming_filters.md)).
