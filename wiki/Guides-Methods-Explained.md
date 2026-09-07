# The methods explained

Each method below follows the same four-part shape:

1. **The intuition** -- what it's doing, in plain words.
2. **How it works** -- the actual steps.
3. **Why it's correct** -- the proof, built up gently.
4. **Knobs & adaptations** -- what you can tune and the variants that exist.

They all rest on the one idea from [the guide](Guides): *match the integral
fingerprint (differential spectrum) of the model to that of the data.* If you
haven't read that page, read its sections 2-3 first. For the fully formal
treatment of any method, follow the link to [../methods/](Methods).

Quick map:

- [DSB -- the exact, symbolic reference](#dsb)
- [The image -- what every method fits](#image)
- [LSI -- the accurate batch fitter](#lsi)
- [EAC -- the robust, fast batch fitter](#eac)
- [The streaming filters -- real-time tracking](#streaming)
- [Stochastic -- fitting the functionals of a random series](#stochastic)

---

<a name="dsb"></a>
## DSB -- Differential Spectra Balance (the reference)

Full math: [../methods/dsb.md](Methods-DSB).

### The intuition

DSB is the purest expression of the core idea. It says: *make the model's
fingerprint exactly equal the data's fingerprint, number for number, and solve
for the parameters.* No approximation, no least-squares slack -- an exact balance.

Think of it as a set of scales. On the left pan you put the model's fingerprint
numbers (which contain the unknown parameters as algebra); on the right pan the
data's fingerprint numbers. You balance the pans one level of detail at a time.
Each balanced pan is one equation; with as many equations as unknown parameters,
you solve the system exactly.

### How it works

1. **Summarize the data with a polynomial.** A polynomial (a sum of powers of
   $t$) can be fit to data by *linear* least squares -- easy and stable. Its
   coefficients *are* the data's fingerprint numbers, read off directly.
2. **Write the model's fingerprint** by differentiating the model expression
   symbolically (with SymPy). The unknown parameters appear inside these
   expressions.
3. **Balance** -- set model fingerprint = data fingerprint at each level of
   detail, producing a small system of equations.
4. **Solve symbolically** for the parameters; if there are more equations than
   unknowns, refine the solution numerically.

### Why it's correct

This rests on the faithful-ID property from the guide: two smooth functions are
equal **if and only if** their fingerprints agree at every level. So if the data
truly came from $f(t;\theta^*)$ for some true parameters $\theta^*$, and your
polynomial captured enough levels of the fingerprint, then forcing the model
fingerprint to equal the data fingerprint forces the model to equal the true
function -- which pins down $\theta = \theta^*$. On clean, ideal data this is an
**exact identification**, not an approximation. That is precisely why DSB is the
*reference*: the numeric methods are judged against the answer DSB would give on
perfect data.

A neat simplification the implementation exploits: in the balance, a scaling
factor that appears on *both* pans (the $H^k$ in the formal definition) cancels,
so DSB just matches plain polynomial coefficients -- and that works for **any**
model you can differentiate, not just a hand-coded list of exp/sin/cos.

### Knobs & limits

- `rank` -- how many fingerprint levels (equations) to balance.
- **Why it's not for production:** the symbolic solve has unpredictable runtime,
  and it leans on the polynomial's *high-order* coefficients, which are exactly
  the ones noise corrupts most. On noisy data DSB becomes an unreliable curve fit.
  Use it to derive and validate; use LSI/EAC to actually fit.

---

<a name="image"></a>
## The image -- what every method fits

Full math: [../methods/image.md](Methods-Image).

### The intuition

Every method below needs a way to turn raw samples into a fingerprint before
it can compare data to model. In the code, that fingerprint has a name: the
**image**. An image is a fixed-size summary of the data in a chosen basis
(Legendre polynomials for LSI, block indicators for EAC), built once from the
samples, that the model is matched against instead of the raw points.

### How it works

Evaluate the basis at the sample positions and form two objects: the
projections $S$ (how much of each basis function the data contains) and the
Gram matrix $G$ (how the basis functions correlate with each other on this
exact grid of samples). Together with the sample count and the raw sum of
squares, $S$ and $G$ are the sufficient statistic of the data's regression
onto the basis -- everything a fit needs, and nothing more. The model gets the
same treatment on the same grid, with the same weights: its own projection
$S_f(\theta)$, function of the unknown parameters. Fitting is then reduced to
closing the gap between $S$ and $S_f(\theta)$.

### Why it's correct

$S$ and $G$ are sums over samples, so they are **additive**: pool two sample
sets and their images add, term by term, whatever the sample order or overlap.
Comparing the model's projection to the data's is exactly comparing the two
curves' fingerprints in the least-squares sense over the basis's span -- exact
when the model lies in that span, and costing no further pass over the data
once the image is built: many candidate models can be matched against the same
image.

LSI is this image in the Legendre basis; EAC is this image in the block basis.
Everything from here on is which basis, and which knob.

---

<a name="lsi"></a>
## LSI -- Least-Squares Integral (the accurate default)

Full math: [../methods/lsi.md](Methods-LSI).

### The intuition

LSI keeps DSB's idea but drops the demand for an *exact* balance. Real data is
noisy, so an exact match is both impossible and undesirable (you'd be matching
the noise). Instead LSI asks for the **best possible** match: make the model's
fingerprint as close as possible to the data's, in the least-squares sense. This
one change turns a brittle symbolic solve into a robust numerical fit.

It also fixes a subtle numerical trap. If you build the fingerprint from plain
powers ($1, t, t^2, t^3, \dots$), the high powers look almost identical to each
other over an interval, which makes the math *ill-conditioned* -- tiny data
changes cause huge parameter swings. LSI swaps the plain powers for **Legendre
polynomials**, a set of "spread-out," mutually-independent shapes that don't step
on each other. With them the problem becomes perfectly stable.

(If "Legendre polynomials" means nothing to you: think of them as a better set of
measuring sticks. Plain powers are like rulers that are almost the same length,
so you can't tell them apart; Legendre polynomials are rulers of clearly
different, independent lengths, so each measures something distinct.)

### How it works

1. **Build the image.** Project the data onto the Legendre polynomials up to
   order `k_star` (default from `order_for`): the weighted projections $S$ and
   the Gram matrix $G$ of the basis on the data's own sample grid.
2. **Project the model** onto the same basis at the same order, $S_f(\theta)$
   -- the same projection, taken on the model instead of the data.
3. **Whiten and match:** with $G = L L^T$, minimize the whitened gap
   $\lVert L^{-1}(S - S_f(\theta)) \rVert^2$.
4. **Solve:** Levenberg-Marquardt when the fit is unbounded; a bounded
   trust-region solve with `bounds`, with a differential-evolution search
   behind it when that local solve comes out poor.

### Why it's correct

Start from the most natural goal: make the model's projection onto the basis
match the data's, in the metric the Gram matrix defines,

$$ J(\theta) = (S - S_f(\theta))^T G^{-1} (S - S_f(\theta)) . $$

Because $G$ is exactly the correlation of the basis functions on this sample
grid -- not an idealized orthogonal inner product -- this identity holds
whatever the grid: uniform, clustered or random; the Legendre basis is not
required to be orthogonal for it to work, only well-conditioned. Minimizing
$J$ is minimizing the actual weighted residual sum of squares of any function
in the basis's span. That is the whole justification -- and it's why LSI is
both faithful to the data (it minimizes real reconstruction error) and
numerically stable (Legendre polynomials keep $G$ well-conditioned, unlike the
notorious *Hilbert matrix* of plain powers). The full derivation is in
[../methods/lsi.md](Methods-LSI).

### Knobs & adaptations

- `k_star` / `order` -- the Legendre order (spectral resolution). `k_star=None`
  or `"auto"`, and an omitted `order`, both take the default from `order_for`.
- `order_for(model, params, domain)` -- the smallest order at which every
  parameter sensitivity is represented to within 2% relative error; raise
  `order` when `fit`'s coverage warning fires.
- **The oscillatory recipe** (`oscillatory=True` / `freq_param=`): a low order
  *erases* a cycle, so for sinusoids LSI raises the order to resolve it
  (`osc_order`) and seeds the frequency from the data's FFT peak
  (`fft_frequency_seed`). See [fft_frequency_seed](API-Fitting#fft_frequency_seed).
- `bounds` -- per-parameter ranges switch the solve to a bounded trust
  region and put a differential-evolution search behind a poor local solve,
  which is what lets LSI fit stubborn exponential/transcendental models
  without a good starting guess.
- `robust=True` -- the robust image: Huber-reweights the basis regression
  before the model is ever fit, so a handful of outliers cannot pull the
  image off the clean signal.

---

<a name="eac"></a>
## EAC -- Equal-Areas Criterion (the robust, fast one)

Full math: [../methods/eac.md](Methods-EAC).

### The intuition

EAC matches the **simplest** fingerprint of all: the **area under the curve**.

Split the data into a handful of consecutive windows. For each window, measure
the area under the data, and the area under the model. Tune the parameters until
every window's model-area equals its data-area. That's it -- "equal areas."

Why does so crude a quantity work? Because area is an *integral*, and (from the
guide) integrals average out noise. EAC never differentiates the data and never
builds a wobbly high-order polynomial; it only ever sums the data up. That makes
it the **most noise-robust** of the batch methods and, because each window is one
cheap equation, the **fastest**.

### How it works

1. **Pick windows.** Split the domain into `n_windows` equal windows (default
   $4$ per parameter, so the system has redundancy to average over).
2. **Build the image.** Each window's indicator is one basis function;
   projecting the data onto them gives the window sums $S$ (the data's area in
   each window) and their Gram matrix $G$ (which windows overlap -- for equal,
   disjoint windows, none, so $G$ is diagonal).
3. **Model areas.** Project the model onto the same block basis, giving its own
   window sums $S_f(\theta)$.
4. **Solve** by minimizing the whitened gap between $S$ and $S_f(\theta)$.

### Why it's correct

Each window sum is itself a summary of the fingerprint -- integrating the curve
over a window is a particular weighted combination of its fingerprint numbers.
So matching sums over $M$ well-placed windows is matching $M$ independent
summaries of the fingerprint, and once you've matched as many independent
summaries as the model has free parameters, the model is pinned down (this is
a *weak-form*, or Galerkin, identification -- the formal statement is in
[../methods/eac.md](Methods-EAC)).

The robustness has a one-line proof: zero-mean noise integrates toward zero,
$\int_W \varepsilon(t)\,dt \to 0$ as the window grows. The data enters EAC *only*
through these integrals, so the noise is mostly gone once the image is built,
before the fit even starts.

**Why overdetermine it?** Using more windows than parameters ($4m$ by default)
means the random per-window integration errors partly cancel across windows,
lowering the variance of the estimate -- and it lets EAC report a parameter
**covariance** (uncertainty) from the leftover residuals.

### Knobs & adaptations

- `n_windows` -- number of block windows (default $4m$). More windows localize
  information; too many makes each window tiny and noisy.
- `robust=True` -- the robust image: Huber-reweights the block regression
  before the model is fit, the outlier defence for a single fit.
- `bounds` -- constrained fits (switches to a trust-region solver).
- **Overlapping-window ensemble** (`ensemble_fit`): for a *densely*
  contaminated record, fit many overlapping sub-windows and take the
  **median** of the per-window estimates -- whole corrupted windows are
  outvoted, with no scale to tune, and the inter-window spread is a free
  uncertainty band. On clean data prefer a single fit.
  -> [../methods/ensemble.md](Methods-Ensemble)

---

<a name="streaming"></a>
## The streaming filters -- real-time tracking

Full math: [../methods/legendre-filter.md](Methods-Legendre-Filter).

### The intuition

The batch methods above look at the *whole* dataset at once. But sometimes data
arrives one sample at a time (a sensor, a live feed), the parameters **drift**
over time, and you need an answer *now*, at fixed cost per sample. `ImageFilter`
does this by treating each sliding window of the stream as its own **image** --
the same projection a batch fit builds -- and updating the parameter estimate
one sample at a time instead of re-fitting the window from scratch.

- You hold a current estimate of the parameters, plus a gain state that tracks
  how much new information each update carries.
- Each new sample produces a **surprise**, the window image's whitened
  innovation: how far the window's projection was from what the current
  parameters predict.
- You nudge the parameters to reduce the surprise -- nudging the uncertain
  directions more and the confident ones less.

The "surprise" is **not** a single-point error (which would be noisy). It's the
mismatch between the data's and the model's projection over the whole sliding
window -- an area sum (`EACFilter`, the block basis) or a Legendre spectrum
(`LSIFilter`) -- so the streaming filter inherits EAC/LSI's
integrate-don't-differentiate robustness.

### How it works (per sample)

1. Add the new sample to a sliding window; drop the oldest.
2. Image the window in the chosen basis for the data and for the model ->
   the whitened **innovation** (the surprise) and its sensitivity to each
   parameter.
3. Apply the correction in information form: move the parameters by gain x
   surprise, and update the gain state. All pure NumPy -- the symbolic work
   was done once at construction, so each update has **bounded cost** and is
   real-time safe.

### Detecting regime changes (drift)

A single smoothly-updated estimate cannot represent a **sudden structural break**
(a currency un-pegging, a plant fault). So the filter watches the stream of
surprises for two patterns, through the same detector `ImageStream` uses on its
blocks:

- a single **big** surprise (a sudden jump) -- caught by a chi-squared test (NIS);
- a **sustained** lean in one direction (slow drift) -- caught by a two-sided
  CUSUM accumulator.

When either fires, the filter **re-arms** (resets or inflates its gain state) so
it re-adapts to the new regime instead of stubbornly averaging across the break.

### Knobs & adaptations

- **Presets** -- rather than set the knobs below by hand, start from a curated
  classmethod: `ImageFilter.tracking(...)` favors fast re-adaptation (the
  default), and `.robust(...)` favors stability under outliers/dropouts. The
  individual knobs below still override anything a preset sets.
- `window_size` -- the window cap: smoothing vs responsiveness. `adaptive_window`
  (on by default) sizes it from the data instead of holding it fixed.
- `q_diag` -- how fast you allow each parameter to drift.
- `noise_var` -- how much you trust each measurement; left alone it is
  estimated online from the window residual.
- `cusum_k` / `cusum_h` -- drift-detector sensitivity vs false-alarm rate.
- **`LSIFilter` vs `EACFilter`:** `ImageFilter` with `basis="legendre"` or
  `basis="block"` fixed. Use `LSIFilter` (spectrum measurement) for
  **oscillatory** plants, where the shape and frequency come straight out
  of the spectrum; use the cheaper `EACFilter` for monotone/saturating
  signals, or for the smallest embedded footprint.
- `result()` -- the current window as a batch fit, with a calibrated
  covariance; `P` itself is a gain state, not a confidence measure.
- **Pooling several streams** -- summing several filters' `nis_` is a fused
  fault test with more degrees of freedom and power than any one filter's
  innovation alone.
  -> [api/streaming.md#several-streams](API-Streaming#several-streams)
- **Coasting through gaps** (`filter.coast(x, order=)`, `coast_cov`) -- when
  measurements drop out, roll the current parameter model forward
  (dead-reckoning) and grow the uncertainty band with the length of the gap,
  instead of freezing the last estimate or letting a raw extrapolation diverge.

---

<a name="stochastic"></a>
## Stochastic -- fitting the functionals of a random series

Full math: [../methods/stochastic.md](Methods-Stochastic).

### The intuition

Everything above fits a *deterministic* law `y = f(t; theta)`. But some series --
asset returns, interest rates, river levels -- are genuinely **random**: there is
no smooth `f` to match, and asking a batch method for one is a category error (it
just fits the noise). The insight is that a random process still has
**deterministic functionals** -- its autocorrelation, its spectrum, its
aggregated variance, its trend-plus-cycle -- and *those* have exactly the shapes
dtfit is built for: an autocorrelation that decays like `exp(-k/tau)`, a
low-frequency spectrum that follows a power law, a damped cosine. So you fit the
**functional** with LSI/EAC and read the process's parameters out of its shape.

### How it works

1. Compute a deterministic functional of the series (the sample autocorrelation,
   or the aggregated-variance curve across block sizes, or the low-frequency
   spectrum).
2. Fit that functional with `fit_lsi` / `fit_eac` -- the same machinery as any
   curve, because its shape is a decaying exponential / power law / damped cosine.
3. Read the stochastic parameter off the fitted shape: the AR(1) mean-reversion
   `phi` from the ACF decay, the long-memory **Hurst** exponent from the
   aggregated-variance (or spectral) slope, the **GARCH** volatility persistence
   from the ACF of `|returns|`, a stochastic cycle's period from a damped-cosine
   ACF fit.

### The merged solution

`fit_stochastic(y)` composes these routes behind significance gates into one
`StochasticModel`: a unit-root test decides stationary vs. random-walk, then the
trend / seasonal / long-memory / mean-reversion / volatility stages each fire only
when their signal clears a gate -- so a plain random walk is **not** handed a
spurious trend or cycle. The model **identifies the regime**, **forecasts** by
picking the best regime-appropriate model on a rolling backtest (`.forecast`), and
can **generate** fresh realizations of the same process (`.simulate`, with
Gaussian or fat-tailed Student-t innovations). `StochasticFilter` is the
per-sample streaming twin (online regime tracking + change detection). The
individual estimators -- `hurst_aggvar`, `hurst_spectral`, `ar1_reversion`,
`garch_persistence`, `cycle_period`, `decompose_trend_cycle`, plus `ar_order` /
`fit_ar` (finite-order AR, so an AR(2)/AR(3) isn't mistaken for long memory) and
`fractional_difference` -- are all public in `dtfit.stochastic`.

### Why it's correct

The functionals are population quantities with known parametric forms: an AR(1)'s
theoretical ACF *is* `phi^k`; an ARFIMA's spectrum *is* a power law near zero
frequency. Matching the sample functional to that form is an ordinary curve fit,
and the integral methods' noise-averaging is exactly what a jittery empirical ACF
needs. The honest limit: dtfit recovers the **regime and its parameters**, not a
path -- you cannot out-forecast a martingale, and on clean data a dedicated
likelihood estimator (GARCH-QMLE) can be a touch sharper on the raw parameter. Its
edge is blind **regime identification** and a single, coherent API from estimator
to forecast to generator.

### Knobs

- `period=` / `max_harmonics=` -- seasonal control for the trend+cycle route.
- `forecaster=` -- force a specific forecaster, or `"auto"` to let the backtest pick.
- `dist="t"`, `df=` -- fat-tailed innovations for `simulate` / forecast intervals.

---

## Side-by-side summary

| | DSB | LSI | EAC | EACFilter / LSIFilter |
|---|---|---|---|---|
| **Matches** | exact fingerprint | image (least-squares, Legendre) | image (block, window sums) | area / spectrum, one sample at a time |
| **Mode** | symbolic, offline | batch, offline | batch, offline | streaming, online |
| **Best at** | derivation / reference | accurate general fitting | robust, fast, few-parameter | real-time tracking + drift detection |
| **Noise** | fragile | tolerant | most robust | robust (integral measurement) |
| **Use for production?** | no | yes (default) | yes | yes (real-time) |

Where to go next:

- **Which one for my data?** -> [choosing-a-method.md](Guides-Choosing-a-Method)
- **Exact signatures and arguments** -> [../api/](API)
- **The full proofs** -> [../methods/](Methods)
- **The experimental adaptations and how they were validated** ->
  [../experimental/](Experimental)
