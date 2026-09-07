# dtfit vs SciPy

> Also published in the dtfit docs site as its `comparison.md`. This is the wiki mirror.


**Short version.** For a one-shot batch curve fit with a reasonable initial
guess, `scipy.optimize.curve_fit` is the right tool: it is faster and more
general than dtfit, and just as accurate. dtfit earns its place on the problems
`curve_fit` is *not* built for -- **streaming / recursive** estimation,
**out-of-memory** and **many-channel** fits, **embedded / real-time** use, and
**robustness without hand-tuning**. Treat dtfit as complementary to SciPy (and
[lmfit](#a-note-on-lmfit)), not a replacement.

The numbers below are produced by
[`gen_comparison.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/docs/gen_comparison.py) (run it yourself with
the repo venv). Both fitters get the **same** data-driven initial guess (the
model's own `_seed_arrays`), so the comparison isolates the *method*, not the
seeding. Scenarios are drawn from dtfit's own accuracy corpus
(`tests/accuracy`). Timings are wall-clock medians on the development machine
(Windows, CPython 3.14, pure-Python kernels) and are **indicative** -- treat the
*ratios* as the signal, not the absolute milliseconds.

## Head-to-head: one-shot batch fits

`param err` is the maximum relative parameter-recovery error against ground
truth; `R^2` is against the clean signal. Lower `param err` and higher `R^2` are
better.

| scenario | noise | dtfit param err | dtfit R^2 | dtfit ms | scipy param err | scipy R^2 | scipy ms |
|---|---|---|---|---|---|---|---|
| exponential | 0.00 | 1.4e-16 | 1.00000 | 3.0 | 0.0e+00 | 1.00000 | 0.7 |
| exponential | 0.05 | 4.4e-03 | 0.99999 | 3.2 | 4.4e-03 | 0.99999 | 0.7 |
| logistic (sigmoid) | 0.00 | 5.6e-13 | 1.00000 | 7.3 | 1.8e-16 | 1.00000 | 2.0 |
| logistic (sigmoid) | 0.05 | 1.9e-02 | 0.99996 | 7.4 | 9.0e-03 | 0.99999 | 2.0 |
| damped oscillation | 0.00 | 2.6e-07 | 1.00000 | 3.0 | 1.3e-11 | 1.00000 | 1.9 |
| damped oscillation | 0.05 | 5.0e-03 | 0.99999 | 3.2 | 5.0e-03 | 0.99999 | 2.1 |

**Reading this honestly:**

- **Accuracy is essentially a tie.** On clean data both recover the parameters to
  machine / solver precision. Under 5% noise dtfit and `curve_fit` land within the
  same order of magnitude on every scenario; `curve_fit` is marginally better on
  the logistic (0.9% vs 1.9% worst-parameter error), dtfit matches it on the
  exponential and the damped oscillation.
- **`curve_fit` is faster** -- roughly 2-4x here -- because it runs a lean
  Levenberg-Marquardt loop, while dtfit builds an orthogonal-basis spectral (LSI)
  or area (EAC) system first. For a single well-posed batch fit, that overhead
  buys you nothing.

So on their shared home turf, **use `curve_fit`.** The rest of this page is where
that changes.

## Where dtfit wins: things `curve_fit` cannot do

### Out-of-core, one pass

`curve_fit` needs every sample resident in memory and runs its NLLS over the full
array. dtfit's `ImageStream` accumulator folds each chunk into a fixed-size image
(an additive basis projection), so it fits a dataset far larger than memory in a
single streaming pass -- and the same accumulators `merge()` across distributed
workers holding contiguous chunks of a uniform grid (or `grid="explicit"` for
other sample sets).

- **n = 2,000,000 points**, streamed in 20 chunks of 100,000 (peak working set
  ~1.6 MB of sample buffers).
- Recovered `a = 1.5000` (truth 1.5), `b = 0.8000` (truth 0.8) in **~60 ms**, one
  pass.
- `curve_fit` would need all 2,000,000 points resident (~32 MB for x + y) and a
  full NLLS over them; the map-reduce accumulator never holds more than one chunk.

### Online drift tracking

When the parameters *change over time*, a single batch fit can only report one
compromised value. dtfit's streaming filters (`EACFilter` / `LSIFilter`) update
recursively with `partial_fit`, tracking the drift sample-by-sample at O(1) cost
per step.

- True amplitude drifts **1.0 -> 3.0** over 600 samples of `A(t)*sin(t)`.
- A single `scipy.curve_fit` gives one value `A = 1.99` (the average) --
  RMSE **0.52** against the drifting truth.
- The streaming `EACFilter` tracks it online, RMSE **0.11** (final estimate
  `A = 2.93`, truth 3.00).
- `curve_fit` has no online / `partial_fit` mode: tracking drift means re-fitting
  the whole growing window every step.

### Robustness without tuning

- **Self-seeding models.** `models.logistic().fit(x, y)` or `auto_estimate`
  derive `p0`/`bounds` from the data, so you are not hand-feeding a starting guess
  (the thing `curve_fit` most often fails without).
- **Integral criteria denoise by construction.** EAC matches *areas*; integration
  is a low-pass operator, so it degrades gracefully as noise rises instead of
  chasing a high-order polynomial.
- **Outliers.** `ensemble_fit` (overlapping-window median) and the robust image
  (`fit_eac(..., robust=True)`) reject contamination with no scale to tune.

### Embedded / real-time

The streaming filters have a fixed per-step cost and a small state, and the
numeric core has optional compiled C kernels with a pure-NumPy fallback -- so the
same estimator runs on a workstation and on a microcontroller-class target. A
batch NLLS is not a natural fit for a per-sample real-time loop.

## When to reach for which

| Situation | Best tool |
|---|---|
| One-shot batch fit, good initial guess | **`scipy.optimize.curve_fit`** |
| Batch fit needing bounds, CIs, model comparison, GUI-friendly params | **lmfit** (or dtfit's `Model` / `suggest_models`) |
| Live / recursive tracking, drifting parameters | **dtfit** streaming filters |
| Data too big for memory, or distributed | **dtfit** `ImageStream` |
| Embedded / real-time, fixed per-step budget | **dtfit** streaming filters |
| Noisy / outlier-prone, no guess to give | **dtfit** self-seeding models + robust EAC / ensemble |

## A note on lmfit

[lmfit](https://lmfit.github.io/lmfit-py/) sits between `curve_fit` and dtfit's
model layer: it wraps SciPy's optimizers with named `Parameters`, bounds,
constraints, uncertainties and model composition -- a superb batch curve-fitting
ergonomics upgrade. It is still fundamentally a **batch** tool built on the same
NLLS core, so it shares `curve_fit`'s ceiling on streaming, out-of-core and
embedded work. dtfit does not depend on lmfit and does not try to replace it; if
you want the nicest *batch* experience, lmfit is an excellent choice, and dtfit
complements it exactly where batch NLLS runs out.
