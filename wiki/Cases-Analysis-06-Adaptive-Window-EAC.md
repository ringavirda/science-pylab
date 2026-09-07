# #6 -- Adaptive / multi-resolution EAC windows

> **Status (2026-09):** retired. Window placement by curvature left with
> the image core: `fit_eac` places `n_windows` equal windows and the
> estimate is the projection on them. The sections below describe the
> adaptation as it was run.

**Verdict.** Promoted at the time for localized transients that equal-span
windows smear; retired because the placement moved with the noise and gave
no statistical benefit over equal windows at the same count.

Source: [Noise & robustness (3)](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/03_noise_robustness/03_noise_robustness.ipynb) (transient class).

## What it is

Stock EAC splits the domain into **equal-area / equal-span** windows. This
adaptation places windows **adaptively** -- a dyadic pyramid or
curvature-weighted placement -- so resolution concentrates where the signal
actually changes, instead of being spread uniformly.

```
fit_eac(t, y, "K*(1-exp(-a*x))", "x", window_mode="curvature", p0=[1.0, 1.0])
```

## Measured result

Saturating transient `y = K.(1 - e^{-a x})`, truth `K=2.0`, `a=3.0` (Exp 3): the
adaptive fit recovers both parameters within tolerance (`K` to <0.2, `a` to <0.5)
where the fast initial rise dominates the information but occupies a small slice
of the time axis.

## Why it works (the mechanism)

EAC infers parameters from **window areas**. The information about a transient's
rate is concentrated in the **high-curvature region** -- the fast initial bend of
`1 - e^{-a x}`. Two consequences:

1. **Equal-span windows under-sample the informative region.** A transient that
   rises in the first 10% of the axis and then flattens puts almost all of its
   parameter information in that first 10%. Equal-area windows allocate most of
   their windows to the flat tail, where the area barely depends on `a` -- so the
   rate parameter is weakly constrained and noise-sensitive.

2. **Curvature-weighted placement matches windows to information density.**
   Placing more, narrower windows where `|y''|` is large means the area
   measurements that *do* depend strongly on the rate are resolved finely, while
   the uninformative flat region is covered cheaply. This is the EAC analogue of
   adaptive quadrature / mesh refinement: spend resolution where the integrand
   varies.

In effect it aligns the **measurement grid with the Fisher information** of the
problem -- the windows that most constrain the parameters get the most resolution.

## Outcome

The placement never settled on a stable per-signal layout -- it moved with
the noise realization from run to run.

## When it would help / when it wouldn't

- **Helps:** signals with **localized features on a slow background** -- transients,
  edges, bursts, anything where information density is spatially uneven.
- **Won't help:** signals whose information is **uniformly distributed** (a pure
  sinusoid, a steady exponential) -- there the equal-span grid is already optimal,
  and adaptive placement just adds machinery for no gain.

## Related

- Same experiment, different lesson: the [ensemble (#3)](Cases-Analysis-03-Overlapping-Ensemble)
  manipulates *which windows* are aggregated for robustness; this manipulates
  *where windows are placed* for resolution. Both are window-geometry ideas; this
  one targets information density, that one targets outlier independence.
