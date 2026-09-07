# Lineage, versions & variants -- the complete atlas

This page is the **single map of everything `dtfit` contains**: where each method
came from, how it was improved (its *versions*), every variant and tuning
*approach*, and every *adaptation* -- promoted or still experimental. If
[methods-explained.md](Guides-Methods-Explained) tells you *how each method works*,
this page tells you *how they all relate and how they evolved*.

It is written to be read top-to-bottom, but you can also use it as a reference:

- [1. The family tree](#tree) -- symbolic originals -> numeric successors -> streaming
- [2. Why the methods were rewritten](#why) -- the one hard design rule
- [3. The version history of each method](#versions) -- original vs. current
- [4. The variant catalog](#variants) -- every approach and knob, in one table
- [5. The adaptations](#adaptations) -- #1-#6 and the rest, promoted and experimental
- [6. The whole library at a glance](#glance)

---

<a name="tree"></a>
## 1. The family tree

Everything descends from one principle (see [the guide](Guides)): *a function
is identified by its differential spectrum (its integral "fingerprint"), so you
recover parameters by matching the model's fingerprint to the data's.* That
principle was first realized as a set of **symbolic** methods, which were then
rewritten as **numeric** methods for production, and finally extended to a
**streaming** tier.

```
                     Differential-transformation principle
                     "match the fingerprint to recover parameters"
                                      |
        +-----------------------------+------------------------------+
        |                                                             |
   SYMBOLIC ERA (analytical, exact, fragile)                  (kept only as reference)
        |                                                             |
   +----+-----+---------------+                                       |
  DSB        DSBI            DSBE  <--- the three original              |
 (balance)  (integral)     (equal areas)   "differential spectra        |
        |        |               |          balance" formulations       |
        |        |               |                                      |
        |   rewritten        rewritten                                  |
        |   numerically      numerically                               DSB
        v        v               v                              (the only symbolic
   (reference)  LSI             EAC                              method still shipped,
                |               |                                as a derivation tool)
                |               |
        NUMERIC ERA (production: noise-tolerant, bounded latency)
                |               |
                +-------+-------+
                        |  run recursively, one sample at a time
                        v
        STREAMING TIER (online, real-time, drift-aware)
                   +----+----+
               LSIFilter   EACFilter
```

- **DSB / DSBI / DSBE** are the original *symbolic* methods (the "differential
  spectra balance" family): DSB balances spectra and solves exactly, DSBI is the
  integral-least-squares variant, DSBE the equal-areas variant.
- **LSI** is the numeric successor of **DSBI**; **EAC** is the numeric successor
  of **DSBE**. These are what you use in production.
- **DSB** survives as the lone symbolic method, kept as the analytical *reference*
  the numeric methods are checked against (not for fitting noisy data).
- **LSIFilter / EACFilter** are the streaming twins of LSI / EAC -- the same
  matching run recursively per sample.
- **Stochastic** (`fit_stochastic`, `StochasticModel`, `StochasticFilter`) is a
  layer built *on top of* LSI/EAC, not a new matcher. A random series has no
  deterministic `f`, but its **functionals** (autocorrelation, spectrum,
  aggregated variance, trend+cycle) do -- and they have the exp/power-law/cosine
  shapes the numeric methods fit. So the stochastic tier fits those functionals
  with the same LSI/EAC machinery and reads the process parameters out. See
  [methods-explained.md#stochastic](Guides-Methods-Explained#stochastic).

---

<a name="why"></a>
## 2. Why the methods were rewritten -- the one hard rule

The dissertation imposes a single hard requirement on anything that runs at
**deploy time**: **no unbounded symbolic solve on the runtime path.** A symbolic
solver (`sympy.nonlinsolve`) has input-dependent, unbounded latency -- fine for an
offline derivation, unacceptable in a control loop or a streaming pipeline.

That rule is the reason the whole numeric era exists. SymPy is allowed exactly
**once, offline**, at derivation/initialization time (to differentiate the model
and compile a fast callable); after that, every fit is pure NumPy/SciPy with
bounded cost. So:

- the symbolic **DSB** stays a reference/derivation tool;
- the numeric **LSI / EAC** do the actual batch fitting;
- the **filters** do the real-time work, compiling once in `__init__` and then
  running fixed-cost updates.

A second motivation was **noise**. The exact symbolic balance matches the
high-order coefficients of a polynomial pre-fit -- exactly the coefficients noise
corrupts most. Relaxing "exact match" to "best least-squares match," and working
directly on the raw `(x, y)`, is what made the numeric methods noise-tolerant.

---

<a name="versions"></a>
## 3. The version history of each method

Each numeric method went through real revisions. Knowing the *old* version
explains why the *current* defaults are what they are.

### DSB -- from a hand-coded table to "any model"

| | original DSB | current DSB |
|---|---|---|
| **Model spectrum** | a hand-written table of closed-form "discretes" for `exp`, `sin`, `cos`, monomials -- plus a symbolic "reflection" pass | generic SymPy differentiation: the Maclaurin coefficients of *any* differentiable expression |
| **Models supported** | only the handful with table entries | anything SymPy can differentiate (rational, log, mixed) |
| **The `H^k` factor** | tracked explicitly | shown to **cancel** on both sides of the balance, so it's dropped -- the balance is just matching plain polynomial coefficients |

The key realization -- that the scale factor cancels in a *balance* -- is what let
DSB drop its per-function table and become general. Details:
[../methods/dsb.md](Methods-DSB).

### LSI -- from an ill-conditioned Hilbert matrix to a Gram-whitened image

| | original LSI (~= DSBI) | current LSI |
|---|---|---|
| **Basis** | plain monomials $1, t, t^2, \dots$ | **Legendre** polynomials, evaluated once into the data's own **image** |
| **The match matrix** | a (weighted) **Hilbert matrix** -- condition number explodes (~$10^7$ by order 5) | the image's own Gram matrix $G$ on the sample grid, whitened directly -- well-conditioned by the Legendre basis, no orthogonality assumed |
| **Empirical spectrum** | raw `numpy.polyfit` (ill-conditioned Vandermonde) | the data's image: weighted projections $S$ read straight off the samples |
| **Model spectrum** | Taylor-truncated | the model's own projection $S_f(\theta)$ on the same grid, exact in the basis's span |
| **High-order control** | a hand-tuned exponential weight $e^{-\alpha i}$ fighting the ill-conditioning | Gram-whitened projected least squares; no exponential weight needed |

In short, the original LSI spent effort *fighting* a bad basis; the current LSI
*changes the basis* so the problem is well-posed to begin with, and reads the
whole match off one fixed-size image. Details: [../methods/lsi.md](Methods-LSI).

### EAC -- from exactly-determined to overdetermined

| | original EAC (~= DSBE) | current EAC |
|---|---|---|
| **Windows** | exactly $m$ (one per parameter) | $4m$ by default (**overdetermined**) |
| **Noise** | no redundancy -> throws away the averaging that integration buys | extra equations average out per-window noise |
| **Uncertainty** | none (square system) | a **covariance** from the leftover residuals |
| **Statistic** | -- | the block **image**: window sums $S$ and their Gram $G$, exact in the block basis's span |
| **Best for** | -- | transients, saturating shapes, few params |

Details: [../methods/eac.md](Methods-EAC).

---

<a name="variants"></a>
## 4. The variant catalog -- every approach in one table

A "variant" here means a distinct *approach* you can call or switch on. This is
the complete list across the stable API.

| approach | call / switch | what it is |
|---|---|---|
| **LSI (default)** | `fit_lsi(...)` | accurate batch fit on the Legendre image; `k_star=None`/`"auto"` (an omitted `order`) takes `order_for`'s default |
| **LSI, global search** | `fit_lsi(..., bounds=...)` | trust-region local solve, with a differential-evolution stage when it's poor -- escapes bad local minima |
| **LSI oscillatory recipe** | `fit_lsi(..., freq_param="w")` or `oscillatory=True` | order raised to resolve a cycle, frequency seeded from the FFT -- recovers sinusoids to <1% |
| **EAC (default)** | `fit_eac(...)` | overdetermined block image, most robust/fastest |
| **Robust image** | `robust=True` on `fit`, `fit_lsi`, `fit_eac` | Huber-reweights the image before any model is fit -- self-scaling, no scale to tune |
| **EAC, bounded** | `fit_eac(..., bounds=...)` | constrained trust-region fit |
| **Missing data** | `fit_lsi/fit_eac(..., nan_policy="omit")` | drop NaNs instead of raising |
| **Ensemble** | `ensemble_fit(...)` | overlapping-window median + spread -- outlier-robust, no scale to tune |
| **DSB** | `fit_dsb(...)` | symbolic exact balance (reference only) |
| **EACFilter** | `EACFilter(...)` | streaming EAC (area measurement) |
| **LSIFilter** | `LSIFilter(...)` | streaming LSI (spectrum measurement) -- for oscillatory plants |
| **Gap coasting** | `filter.coast(...)`, `coast_cov(...)` | dead-reckon a streaming fit through measurement dropouts (uncertainty grows with the gap) |
| **Filter bank** | `FilterBank.from_model(...)` | many streams in lockstep |
| **Fused detector** | `bank.fused_detector(...)` | pool stream innovations for a shared-fault test |
| **Stochastic fit** | `fit_stochastic(...)` -> `StochasticModel` | detect the regime of a *random* series (long-memory / mean-reversion / GARCH / cycle), forecast it, and `.simulate` it -- by fitting its functionals |
| **Stochastic streaming** | `StochasticFilter(...)` | per-sample regime tracking + change detection |
| **Stochastic estimators** | `hurst_aggvar`, `hurst_spectral`, `ar1_reversion`, `garch_persistence`, `cycle_period`, `ar_order`, `fit_ar`, `fractional_difference` | read one process parameter from a functional (all in `dtfit.stochastic`) |
| **Model wrapper** | `Stochastic().fit(series)` | the catalog `.fit()` convention over `fit_stochastic` |
| **sklearn estimator** | `NonlineRegressor(..., method=)` | LSI/EAC/DSB behind `fit`/`predict`/`score` |
| **Auto-estimate** | `auto_estimate(...)` | routes to the variant matching the signal's shape |
| **Auto-forecast** | `auto_forecast(...)` | structured fit-then-extrapolate, with guards |
| **Model catalog** | `models.<family>()` | self-seeding named families; `+` to compose |
| **Model inference** | `suggest_models(x, y)` | fit a shortlist, rank by AIC |

Every entry is documented with full signatures under [../api/](API).

---

<a name="adaptations"></a>
## 5. The adaptations -- #1 through #6, and the rest

Beyond the core methods, the research program produced a numbered series of
**structural adaptations** -- new *ways to compose* the fingerprint machinery.
Each was prototyped in `dtfit-experimental`, validated across the experiment
suite, and either **promoted** into stable `dtfit` (physically moved there) or
kept experimental until it proves itself. Here is the complete list with status.

### Promoted into stable `dtfit`

| # | adaptation | now in `dtfit` as | what it is & how it works |
|---|---|---|---|
| **#1** | one-pass / distributed map-reduce | `ImageStream` accumulator | the image is **additive over the domain** (a sum of per-chunk projections), so a dataset too big for memory is reduced chunk-by-chunk in one pass, and distributed workers' partial images `merge()` exactly on contiguous chunks of a uniform grid (or `grid="explicit"` for other sample sets); the estimators of the original study live in `dtfit_experimental.scale`. -> [../api/scaling.md](API-Scaling) |
| **--** | GEMM-batched projection | `ImageStream(channels=B)` | the image is **linear across channels**, so `B` channels' projections are one matrix multiply over one shared Gram, on CPU/GPU by swapping only the backend; the estimators of the original study live in `dtfit_experimental.scale`. -> [../api/scaling.md](API-Scaling) |
| **#6** | curvature-adaptive windows | retired | EAC places its windows uniformly, $4m$ by default. |
| **--** | LSI oscillatory recipe | `fit_lsi(oscillatory=..., freq_param=...)`, `fft_frequency_seed` | high order + FFT-seeded frequency, so a cycle isn't erased. -> [../api/fitting.md#fit_lsi](API-Fitting#fit_lsi) |
| **--** | fused multi-axis detection | `FusedChiSquareDetector` | pool a filter bank's per-stream innovations into one `chi2(K)` statistic to catch a fault too weak in any single stream. -> [../api/streaming.md#fused](API-Streaming#fused) |
| **#3** | overlapping-window ensemble | `ensemble_fit`, `EnsembleResult` | fit on many overlapping sub-windows and take the **median** of the per-window estimates -- bagging over time; rejects outlier windows and yields a spread. Outlier-robust with no scale to tune. -> [../methods/ensemble.md](Methods-Ensemble) |

### Still experimental (in `dtfit-experimental`)

| # | adaptation | function | what it is & how it works |
|---|---|---|---|
| **#2** | pluggable basis LSI | `fit_lsi_basis` | keep LSI's criterion but choose the basis -- **Fourier** for periodic signals (a wiggle is 2-3 harmonics, not many polynomial orders), **Laguerre** for decays, Chebyshev/Legendre otherwise |
| **#4** | joint shared-parameter fit | `fit_joint` | stack several channels' area equations into one system with **shared** parameters (a common frequency/rate) plus per-channel **private** ones -- more equations per shared unknown |
| **#5** | stage-wise residual boosting | `boosted_fit` | fit stage 1 (e.g. an LSI trend), subtract it, fit stage 2 (e.g. an EAC cycle) on the residual; the sum is more expressive than either alone (the fingerprint is linear, so component fits add up) |
| **--** | inverse-covariance fusion primitive | `InformationFilter` | information-form (inverse-covariance) recursive linear estimator: additive updates, exact/associative `fuse()` for sensor fusion and streaming map-reduce. Shares **no code** with the nonlinear EAC/LSI filters (they run the covariance form directly); exercised by no domain study, so it has **not cleared the >=2-domain promotion gate** and was moved out of stable `dtfit` |

Full signatures and usage for the experimental four:
[../experimental/adaptations-api.md](Experimental-Adaptations-API). The
conceptual write-up and the math each rests on:
[../experimental/README.md](Experimental).

### How promotion works

An adaptation graduates only after the **domain validation suite** shows it helps
across a *range* of applications, not one cherry-picked case. On promotion it is
**moved** into `dtfit` and imported from there (no re-export shim -- the dependency
points one way only). The validation methodology and the baselines each
adaptation is measured against are in
[../experimental/baselines.md](Experimental-Baselines).

---

<a name="glance"></a>
## 6. The whole library at a glance

```
dtfit (stable, public)
+-- batch fitting          fit_lsi . fit_eac . fit_dsb . ensemble_fit
|   support                find_degree . fft_frequency_seed
+-- result type            FittingResult
+-- sklearn estimator      NonlineRegressor
+-- one-call entry points  auto_estimate . auto_forecast
+-- model framework        models.<family> . Model . suggest_models
+-- streaming / online     EACFilter . LSIFilter . FilterBank . FusedChiSquareDetector
|                          (coast/coast_cov dead-reckon through gaps)
+-- stochastic (random)    fit_stochastic . StochasticModel . StochasticFilter . Stochastic
|                          estimators: hurst_aggvar/spectral . ar1_reversion . garch_persistence
|                          cycle_period . decompose_trend_cycle . ar_order . fit_ar . fractional_difference
+-- scaling backends       fit_many . ImageStream
|                          (accumulator . block streams . channels; merge/checkpoint/resume)
+-- diagnostics            fit_report . residual_diagnostics . FitDisplay . ResidualsDisplay

dtfit-experimental (separate; promotes into dtfit when validated)
+-- adaptations in trial   fit_lsi_basis(#2) . fit_joint(#4) . boosted_fit(#5)
+-- fusion primitive       InformationFilter   (info-form; not yet through the gate)
+-- backend helpers        available_backends . resolve_backend . Backend
+-- experiments            cases/ (each lever) . domains/ (vs the real toolkit)
```

Where to go next:

- **What each one does, plainly** -> [methods-explained.md](Guides-Methods-Explained)
- **Which to pick** -> [choosing-a-method.md](Guides-Choosing-a-Method)
- **Exact signatures** -> [../api/](API) (stable) and
  [../experimental/adaptations-api.md](Experimental-Adaptations-API) (experimental)
- **The proofs** -> [../methods/](Methods)
- **How it was validated** -> [../experimental/](Experimental)
