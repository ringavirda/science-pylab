# dtfit documentation

`dtfit` fits models that are **nonlinear in their parameters** -- exponentials,
sinusoids, logistic curves, saturating and peak shapes -- to noisy data, and
recovers the *physical parameters* (a growth rate, a frequency, an asymptote),
not just an opaque curve. It does this through **differential (non-Taylor)
transformations**: instead of comparing the model to the data sample-by-sample,
it compares *integral fingerprints* of the two, which is what makes it robust to
noise.

This folder is the documentation. Pick the door that matches what you need:

| If you want to... | Read |
|---|---|
| **Understand the ideas** from scratch, no heavy math assumed | [guides/](Guides) -- plain-language explanations of every method, with the proofs built up gently |
| **See the full map** -- every method, version, variant and adaptation | [guides/lineage-and-variants.md](Guides-Lineage-and-Variants) -- the complete atlas of where each approach came from and how it relates |
| **Look up a function or class** -- signatures, arguments, return types | [api/](API) -- complete reference for the public `dtfit` API |
| **See the rigorous math** -- the formal derivations and proofs | [methods/](Methods) -- the mathematical reference, one file per method |
| **Learn by running code** -- copy-paste examples | [examples/](Examples) -- quickstart -> methods -> models -> sklearn -> streaming -> scaling -> diagnostics |
| **Understand the research** -- the experimental adaptations and how they were validated | [experimental/](Experimental) -- the `dtfit-experimental` package, the experiment suite, and every baseline it is compared against |

## The shortest possible introduction

There are **three core fitting methods**, plus a streaming one. They are all the
same idea (match integral fingerprints) applied differently:

- **LSI** -- the accurate, general-purpose batch fitter. *Start here.*
- **EAC** -- the fast, most noise-robust batch fitter, best for few-parameter
  transient/saturating shapes.
- **DSB** -- a symbolic *reference* method, used to derive and check the others;
  not for production.
- **ImageFilter** -- the streaming version: feed one sample at a time, track
  parameters that change over time, and detect when the system changes regime.
  `LSIFilter` and `EACFilter` fix its basis to Legendre and block.

On top of those sit convenience layers: a [scikit-learn estimator](API-Estimator)
(`NonlineRegressor`), a [model catalog](API-Models) so you pick a *shape*
instead of writing a formula, [one-call "just fit it" entry points](API-Auto)
(`auto_estimate`, `auto_forecast`), and [scaling backends](API-Scaling) for
big or multi-channel data.

For genuinely **random** data (economic / financial series) there is a dedicated
[stochastic-series solution](API-Stochastic): it fits the deterministic
*functionals* of the process (its autocovariance, spectrum, trend/cycle) to
characterize, forecast, and even generate it (`fit_stochastic`, `Stochastic`), with
a streaming twin (`StochasticFilter`) that tracks the structure live.

New here? Open [guides/README.md](Guides).
