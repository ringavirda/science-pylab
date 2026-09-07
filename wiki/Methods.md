# dtfit methods -- mathematical reference

This folder is the **rigorous mathematical reference** for everything `dtfit`
ships: each method's grounding in **differential (non-Taylor) transformations**,
the full algorithm, the numerical optimizations and guards in the implementation,
and where it is best applied. It is the deepest of the three doc layers -- for
plain-language intuition see [../guides/](Guides), and for exact call signatures
see [../api/](API).

Every batch method is backed by a figure and a comparison table generated from
real code on model *and* real data -- see
[reproducing the figures](#reproducing-the-figures-and-tables).

## The method catalog

`dtfit` is organized as **one principle** (match a signal's image, the
discrete differential transform of its samples) realized across **four
execution tiers** -- symbolic reference, batch, streaming, and
batch-at-scale -- plus a **high-level composition** layer that routes to the
right variant automatically.

| tier | method | doc | runtime path | role |
|------|--------|-----|--------------|------|
| **reference** | **DSB** -- Differential Spectra Balance | [dsb.md](Methods-DSB) | symbolic (offline) | analytical ground truth / derivation |
| **core** | **Image** -- the discrete differential transform | [image.md](Methods-Image) | numeric (offline) | the statistic every batch method fits on |
| **batch** | **LSI** -- Least-Squares Integral | [lsi.md](Methods-LSI) | numeric (offline) | accurate batch fit in the Legendre image, model selection, oscillatory recipe |
| **batch** | **EAC** -- Equal-Areas Criterion | [eac.md](Methods-EAC) | numeric (offline) | fast batch fit in the block image, the robust image |
| **batch** | **Ensemble** -- overlapping-window aggregation | [ensemble.md](Methods-Ensemble) | numeric (offline) | **outlier-robust** bagging over EAC/LSI window fits (median + spread) |
| **streaming** | **EACFilter** -- recursive EAC | [equal_areas_filter.md](Methods-Equal-Areas-Filter) | numeric (online) | real-time tracking via an **area** measurement + drift detection |
| **streaming** | **LSIFilter** -- recursive LSI | [legendre_filter.md](Methods-Legendre-Filter) | numeric (online) | real-time tracking via a **spectrum** measurement (oscillatory plants) |
| **scale** | **ImageStream** -- streams, blocks, channels | [scaling.md](Methods-Scaling) | numeric (offline/online) | one-pass / block / many-channel image map-reduce |
| **compose** | **auto_estimate / auto_forecast** | [auto.md](Methods-Auto) | numeric (offline) | shape-routed estimation and structured forecasting |
| **stochastic** | **Stochastic series** -- fit the functionals of a *random* process | [stochastic.md](Methods-Stochastic) | numeric (offline + online) | characterize / forecast / generate / track random (economic, financial) data |

The production methods (LSI, EAC, the filters, ImageStream) are **numeric
successors** to the symbolic originals (DSBI -> LSI, DSBE -> EAC). DSB is kept as
the analytical reference. The split follows the dissertation's hard requirement
that the **runtime path carry no unbounded symbolic solve** -- SymPy is allowed
only once, offline, at derivation/initialization time. (For the full lineage --
which method came from which, and how the variants differ -- see
[../guides/lineage-and-variants.md](Guides-Lineage-and-Variants).)

---

## The common foundation: differential transformations

All methods operate on the **differential spectrum** of a signal rather than on
its samples directly. For an analytic function $x(t)$ the (scaled) differential
transform about $t_0=0$ is the sequence of *discretes*

$$
X(k) \;=\; \frac{H^{k}}{k!}\,\left.\frac{d^{k}x}{dt^{k}}\right|_{t=0},
\qquad k = 0,1,2,\dots
$$

where $H$ is a scale constant (in this library, the observation interval
$H = t_{\max}-t_{\min}$). The inverse transform reconstructs the function:

$$
x(t) \;=\; \sum_{k=0}^{\infty} X(k)\,\Big(\tfrac{t}{H}\Big)^{k}.
$$

So $X(k)$ is the $k$-th Maclaurin coefficient of $x$, rescaled by $H^k$. The
transform is a **linear bijection** between an analytic function and its spectrum
on the radius of convergence: two analytic functions are equal **iff** their
spectra agree discrete-by-discrete. This is the identity every method below leans
on -- matching spectra (DSB, LSI), matching integrals of spectra (EAC, the
filters), or accumulating spectra additively (ImageStream) all recover the
function and hence its parameters.

The numeric methods compute this on the finite sample, not the infinite
Taylor series above: the **image** ([Methods-Image](Methods-Image)) is the
discrete differential transform of a signal in a basis at a finite order, a
fixed-size statistic built directly from `(x, y)` that every batch method --
LSI, EAC, the filters, ImageStream -- fits on.

### The non-Taylor base (and why no table is needed)

The classical motivation is that elementary functions have **closed-form
discretes** -- you never truncate an infinite Taylor series for them:

| term | discrete $X(k)$ |
|------|------------------|
| constant $c$ | $c$ at $k=0$, else $0$ |
| $t^{\,n}$ | $H^{k}$ at $k=n$, else $0$ |
| $e^{w t}$ | $(wH)^{k}/k!$ |
| $\sin(w t)$ | $\dfrac{(wH)^{k}}{k!}\sin\!\frac{\pi k}{2}$ |
| $\cos(w t)$ | $\dfrac{(wH)^{k}}{k!}\cos\!\frac{\pi k}{2}$ |

A transcendental model such as $a\,e^{bt}$ or $a\,\arctan(wt)$ is represented by
the *exact* discrete of its basis function, so the parameters $a,b,w$ appear
analytically in the spectrum and can be solved for or fitted.

**Implementation note.** `dtfit` does not maintain this table in code. In a
*spectra balance* every equation matches model and data discretes at the same
order $k$, so the $H^{k}$ factor cancels on both sides (see [DSB](Methods-DSB)) and the
balance reduces to matching plain Maclaurin coefficients $g^{(k)}(0)/k!$. Those
are produced for any expression by generic SymPy differentiation
([`methods/_common.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/methods/_common.py)), which
both reproduces the table above and extends the scheme to *any* differentiable
model (rational, logarithmic, mixed) without a per-function rule. The numeric
methods go further and replace the monomial spectrum with a better-conditioned
**orthogonal-polynomial** spectrum ([LSI](Methods-LSI)).

---

## How the tiers relate

```
                 differential spectrum  X(k) = (H^k/k!) x^(k)(0)
                                  |
   +--------------+---------------+----------------+------------------+
   |              |               |                |                  |
 exact balance  weighted-L2    integral / area   additive over       compose /
 F(k;theta)=Z(k)    of spectra     matching          domain & channels   route
   |              |               |                |                  |
  DSB            LSI             EAC              ImageStream          auto_estimate
 (symbolic     (Legendre image      (block image           (streams, blocks,      auto_forecast
  solve)        residual)        residual)         channels, map-reduce)  (shape routing)
                  |               |
          run recursively, one sample at a time
                  |               |
              LSIFilter        EACFilter
           (spectrum meas.)  (area meas.)
```

- **DSB** sets the empirical spectrum (from a polynomial pre-fit) equal to the
  symbolic model spectrum, discrete by discrete, and *solves* the algebraic
  system -- exact when well-posed, but symbolic and noise-sensitive.
- **LSI** relaxes the exact balance to a **weighted integral least-squares**
  discrepancy of the two spectra on an orthogonal basis -- numeric, noise-tolerant,
  accurate; with an oscillatory recipe.
- **EAC** matches **integrals (areas)** of model and data over windows rather than
  spectra -- integration smooths noise, so it is the most robust; the robust
  image is its outlier defense.
- **EACFilter / LSIFilter** run EAC / LSI **recursively**, one sample at a time,
  through a shared `ImageFilter`, with drift detection and a calibrated
  `result()`; summing several filters' `nis_` pools their innovations into
  one fused fault test (`FilterBank` and `FusedChiSquareDetector` are the
  experimental multi-stream harness, not part of this table).
- **ImageStream** exploits that the image is **additive** (so a stream reduces
  chunk-by-chunk, and blocks assemble through the basis transfer) and
  **linear across channels** (so many channels share one Gram) -- exact
  batch fitting at scale.
- **auto_estimate / auto_forecast** route a signal to the variant its shape calls
  for, composing only the validated levers.

---

## Relation to classical (Western) methods

The differential-transformation framing is native to the Pukhov school and largely
invisible in the Western literature, so each batch method is restated here in the
vocabulary a Western reviewer knows. The two production fitters are the two faces of
one **weighted-residual (Galerkin) identification**:

| dtfit method | classical identity | classical relatives (same goal, different route) |
|---|---|---|
| **EAC** (area matching) | Galerkin with piecewise-constant **Haar** test functions; the overdetermined four-windows-per-parameter form is an **over-identified method of moments (GMM)** | Prony / Matrix-Pencil / **ESPRIT** (algebraic pole recovery), variable projection |
| **LSI** (orthogonal-basis spectral match) | a **spectral ($p$-version) Galerkin** projection / weighted integral least-squares in a Legendre basis | classical **method of moments** (the monomial form is LSI's ill-conditioned ancestor), variable projection, Prony/ESPRIT for cycles |

So EAC and LSI are the **$h$-version (local Haar)** and **$p$-version (global
spectral)** of the same projection, both derivable from classical moment-matching
([Pearson](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)),
[Hansen's GMM](https://en.wikipedia.org/wiki/Generalized_method_of_moments)). The
per-method pages develop this ([EAC](Methods-EAC#relation-to-classical-western-methods),
[LSI](Methods-LSI#relation-to-classical-western-methods)), and the experimental
suite runs all of these classical methods as **runnable baselines** -- see
[the baselines page](Experimental-Baselines).

---

## Reproducing the figures and tables

All figures in `figures/` and every comparison table in these docs are produced by
one script against real downloaded data plus clearly labelled model (synthetic)
data. The script lives in the separate `dtfit-experimental` package (install with
`pip install -e "packages/dtfit-experimental[bench]"`):

```bash
python -m dtfit_experimental.experiments.download_data   # fetch COVID-19 + USD/UAH (once)
python -m dtfit_experimental.experiments.benchmark        # write figures/*.png and print tables
```

Real datasets and the dissertation-domain rationale are documented in
[the experiments README](Experiments);
the full validation suites (per-adaptation `cases/` and per-domain `domains/`) and
the established **baselines** each method is measured against are documented in
[../experimental/](Experimental). Baselines include SciPy `curve_fit`
(Levenberg-Marquardt NLS, the gold standard), robust NLLS, `numpy.polyfit`, a
Gaussian process, ARIMA/ETS/Theta, MLP/LSTM nets, an EKF, RLS and a
constant-acceleration Kalman filter -- plus the **Western parameter-estimation
lineage** (Prony, Matrix-Pencil / ESPRIT, variable projection, method of moments)
and a **classical-estimator twin of the stochastic model** (ADF / OLS /
periodogram / DFA / GARCH-QMLE), so the differential-transformation route is
compared against the textbook toolkit it is meant to improve on.
