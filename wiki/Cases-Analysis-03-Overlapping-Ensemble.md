# #3 -- Overlapping-window ensemble + robust aggregation

> **Status (2026-09):** The overlapping-window ensemble is not part of
> `dtfit`. The whole-window-rejection idea it tested helps a contiguous
> burst; the shipped robustness tool for that case is the basis-agnostic
> robust image (`robust=True`), best on the Legendre basis for a burst. The
> sections below describe the study as it was run.

**Verdict: not promoted.** The experiment showed that whole-window rejection
helps a contiguous burst, but the median-of-coefficients aggregation
collapses once *many* overlapping windows are corrupted, and plain **LSI is
both simpler and more robust** at high contamination. It is a specialist
finding, not a default, and it does not ship.

Tested in: [Noise & robustness (3)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/03_noise_robustness/03_noise_robustness.ipynb).

## What it is

Fit the model on many **overlapping sub-windows** of the time axis, then
aggregate the per-window coefficients robustly (median-of-coefficients /
covariance weighting). It is bagging over the time axis: the hope is that
outliers fall in only some windows, so a robust aggregate rejects them and you
also get an empirical uncertainty band for free.

## Measured results

Exponential family, R^2 vs the clean signal as a fraction of points become gross
outliers (Exp 3):

| outlier % | stock EAC | **EAC-ensemble (#3)** | LSI | curve_fit |
|---|---|---|---|---|
| 0 | 1.000 | 0.995 | 1.000 | 1.000 |
| 5 | 0.861 | **0.960** | 0.949 | 0.949 |
| 10 | 0.622 | **0.792** | 0.935 | 0.936 |
| 15 | 0.551 | **-1.479** | 0.919 | 0.913 |
| 20 | -0.200 | 0.015 | 0.892 | 0.893 |

The soft-L1 robust-loss variant tracked stock EAC (little benefit).

## Reading the numbers

- At **<=10% outliers** the ensemble genuinely lifts EAC (0.792 vs 0.622) -- the
  bagging works while most windows are clean.
- At **15%** it *collapses* to R^2 = -1.48 -- worse than doing nothing.
- **LSI alone holds R^2 ~= 0.92 at 20%** -- better than the ensemble at every
  contaminated row, with none of the extra machinery.

## Why it is a specialist, not a general default (the mechanism)

The ensemble's robustness is governed by a **breakdown point**, and the geometry
of overlapping windows pushes that breakdown point *lower*, not higher:

1. **Median robustness needs a clean majority.** Median-of-coefficients tolerates
   corruption only while **<50% of windows** are corrupted. But because windows
   *overlap*, a single gross outlier contaminates *every* window that covers it --
   so a 15% point-contamination rate corrupts far more than 15% of windows. The
   effective breakdown point is well below 50%, which is exactly why the cliff
   appears between 10% and 15%.

2. **Per-window fits are higher-variance.** Each sub-window has fewer samples, so
   each coefficient estimate is noisier. The aggregate trades the bias-resistance
   of the median against a much larger per-member variance -- and when a few members
   blow up (a window dominated by outliers can produce a wild coefficient), the
   median can sit *between* good and catastrophic members and still be bad.

3. **There is a better tool already inside dtfit.** LSI's pre-processing --
   Savitzky-Golay smoothing followed by the integral projection -- rejects gross
   outliers *before* fitting and averages noise *during* fitting. That is a more
   principled robustness route than ensembling after the fact: it attacks the
   outliers at the data level, where they do the damage, rather than trying to
   out-vote them at the parameter level. The numbers bear this out -- LSI is the
   robustness standout, not the ensemble.

The general lesson: **ensembling adds robustness only when members fail
*independently*.** Overlapping windows make members fail *together* (shared
outliers), which is the worst case for any voting/median scheme.

## When it would help

- **Disjoint** (non-overlapping) windows with **sparse, isolated** outliers, where
  a corrupted point lands in one window only -- then independence holds and the
  median is genuinely robust.
- As an **uncertainty estimator** (the spread of member coefficients) rather than a
  robustness device -- that secondary benefit survives even where the robustness
  doesn't.

## Related

- The reliable robustness route is LSI's built-in prefilter -- see the
  [noise report](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/03_noise_robustness/03_noise_robustness.ipynb) "Reading it".
