# The dtfit guide -- the big idea, in plain language

This guide explains **what dtfit does, how it works, and why** -- starting from
zero. You do **not** need to know advanced math or algorithms to follow it. Where
a formula appears, it is introduced in words first, and you can skip the symbols
and still understand the point. When you want the fully rigorous derivations,
each section links to the matching file in [../methods/](Methods).

- **This page** -- the one core idea behind everything, with pictures in words.
- [methods-explained.md](Guides-Methods-Explained) -- each fitting method (LSI, EAC,
  DSB, the streaming filters): intuition -> how it works -> the proof -> the tuning
  knobs and adaptations.
- [lineage-and-variants.md](Guides-Lineage-and-Variants) -- the **complete atlas**:
  where each method came from, how it was improved (its *versions*), and every
  variant and adaptation (#1-#6 and the rest) in one map.
- [choosing-a-method.md](Guides-Choosing-a-Method) -- a decision guide: which method,
  which model, which knob, for your data.

---

## 1. The problem dtfit solves

Suppose you measured something over time and it looks like a curve -- bacteria
growing, a currency falling, a sensor settling, a chemical reaction saturating.
You believe it follows a known *shape* with a few unknown numbers in it. For
example "exponential growth," written

$$ y = a \cdot e^{b t}, $$

where $a$ and $b$ are the two numbers you want to find. $b$ is the growth rate;
$a$ is the starting level. These numbers are the **parameters**, and they
usually *mean* something physical, which is why recovering them matters -- a
black-box curve that merely "looks right" doesn't tell you the growth rate.

The catch is twofold:

1. The relationship is **nonlinear in the parameters.** Doubling $b$ does not
   double $y$; it reshapes the whole curve. You cannot solve for $a, b$ with
   ordinary (linear) least squares the way you can fit a straight line.
2. Your data is **noisy.** Every measurement is the true value plus some random
   jitter, and some points may be wild **outliers**.

The standard tool for this is *nonlinear least squares* (e.g. SciPy's
`curve_fit`): guess the parameters, see how far the curve is from each data
point, nudge the parameters to reduce the total squared distance, repeat. It
works well, but it compares the model to the data **point by point**, so noise
in individual points pushes it around, and it needs a decent starting guess or
it gets lost.

`dtfit` takes a different route that is gentler on noise. That route is the one
idea this whole library is built on.

---

## 2. The one core idea: compare *fingerprints*, not points

Here is the analogy. Imagine two songs and you want to know if they are the same
tune. You could line them up and compare them note-by-note at every instant --
but if one recording is hissy, you'll see differences everywhere even when the
tune is identical. A better way: compute each song's **frequency fingerprint**
(which pitches, how loud) and compare *those*. The fingerprint averages over the
whole song, so the hiss mostly cancels out, and two recordings of the same tune
have matching fingerprints even when the raw waveforms look noisy.

`dtfit` does exactly this for curves. Every smooth function has a **fingerprint**
-- a list of numbers that captures its shape. dtfit computes the fingerprint of
your **data** and the fingerprint of your **model** (which depends on the unknown
parameters), and then tunes the parameters until the two fingerprints match.

The mathematical name for this fingerprint is the **differential spectrum** (also
called the *differential transform* or *non-Taylor transform*). It is the central
object in the author's dissertation, and the source of the package name:
**d**ifferential-**t**ransform **fit**. In the code it has a shorter name, the
**image**: `Original(x, y).image(basis, order)` builds it once from the samples.
See [methods/image.md](Methods-Image).

### What the fingerprint actually is

For a smooth function, the fingerprint is the sequence of numbers

$$ X(0),\ X(1),\ X(2),\ \dots $$

where each $X(k)$ describes how the curve behaves at the $k$-th level of detail:
$X(0)$ is its starting value, $X(1)$ its initial slope, $X(2)$ its curvature, and
so on. (Formally, $X(k)$ is the $k$-th coefficient of the function's Taylor/
Maclaurin series, scaled by a constant -- the full definition is in
[methods/README.md](Methods). You do not need it to read on.)

Two facts make this fingerprint the perfect thing to match:

- **It is a faithful ID.** Two smooth functions are identical *if and only if*
  their fingerprints agree number-for-number. So if you make the model's
  fingerprint equal the data's fingerprint, you have made the model equal the
  data -- and therefore found the right parameters. (This "if and only if" is the
  property every method leans on.)
- **The parameters appear in it cleanly.** For the standard building-block
  functions -- powers, exponentials, sines, cosines -- the fingerprint has a known
  closed form, so the unknown parameters $a, b, w, \dots$ show up as plain
  algebra inside the fingerprint, ready to be solved for or fitted.

That second fact is the "**non-Taylor**" part. Classically you would worry that a
function like $e^{bt}$ has an *infinite* Taylor series you'd have to truncate. The
differential-transform scheme sidesteps that: the elementary functions have exact,
closed-form fingerprints, so nothing is truncated. (In the current code, dtfit
goes one step further and computes the fingerprint of *any* expression you write
by symbolic differentiation, so you are not limited to a hand-coded table -- see
[methods/dsb.md](Methods-DSB).)

---

## 3. Why this is robust to noise

The magic is **integration**. To get the data's fingerprint, dtfit *integrates*
the data (adds it up over intervals) rather than *differentiating* it (looking at
point-to-point differences).

This matters enormously:

- **Differentiating amplifies noise.** Subtracting two nearby noisy points and
  dividing by a tiny gap blows the noise up. Methods that need derivatives of raw
  data are fragile.
- **Integrating averages noise away.** Random jitter is equally likely to be up
  or down, so when you sum many samples the ups and downs cancel and the noise
  shrinks toward zero. The signal's shape survives; the noise washes out.

So by phrasing the fit as "**match integral fingerprints**," dtfit gets noise
robustness essentially for free. This single choice -- integrate, don't
differentiate -- is why the methods degrade gracefully as noise rises, and it is
the thread connecting all of them.

---

## 4. The family of methods -- same idea, four flavors

All the methods match fingerprints; they differ in *which* fingerprint and *how
strictly* they match it. From strictest/most-fragile to most-relaxed/robust:

```
            the differential spectrum (fingerprint) of a signal
                                  |
   +------------------+----------------------+---------------------+
   |                  |                      |                     |
 DSB                 LSI                    EAC                EACFilter / LSIFilter
 match the           match the              match integrated     match fingerprints
 fingerprint         fingerprints in a      AREAS over           one sample at a
 EXACTLY             least-squares sense    windows              time (streaming)
 (symbolic solve)    (accurate batch)       (robust batch)       (real-time tracking)
```

- **DSB -- Differential Spectra Balance.** Demands the two fingerprints be
  *exactly* equal and solves the resulting equations symbolically. Beautiful and
  exact on clean, ideal data -- but brittle once there is noise. Kept as the
  **reference method** used to derive and check the others, not for production.
  -> [methods-explained.md#dsb](Guides-Methods-Explained#dsb)

- **LSI -- Least-Squares Integral.** Relaxes "exactly equal" to "as close as
  possible in a least-squares sense." This is the **accurate, general-purpose
  batch fitter** -- your default. It also uses a smarter fingerprint basis
  (Legendre polynomials) that is numerically stable.
  -> [methods-explained.md#lsi](Guides-Methods-Explained#lsi)

- **EAC -- Equal-Areas Criterion.** Matches the simplest possible integral
  fingerprint: the **area** under the curve over a handful of windows. Because it
  only ever integrates the data (never differentiates, never builds a high-order
  polynomial), it is the **most noise-robust** and the **fastest** batch method.
  Best for few-parameter transient and saturating shapes.
  -> [methods-explained.md#eac](Guides-Methods-Explained#eac)

- **EACFilter / LSIFilter -- the streaming versions.** Run EAC's (or LSI's)
  matching **recursively**, updating the estimate with each new sample at fixed
  cost, like a Kalman filter. They **track parameters that change over time** and
  **detect regime changes** (a sudden break in the data). This is the real-time
  path for control loops and live data streams.
  -> [methods-explained.md#streaming](Guides-Methods-Explained#streaming)

---

## 5. What sits on top (so you rarely touch the raw math)

You can call the methods directly, but dtfit also gives you friendlier layers:

- **A model catalog** -- instead of writing `"L/(1 + exp(-k*(x - x0)))"` you ask
  for `models.logistic()`, and it even reads good starting guesses off your data.
  `suggest_models(x, y)` will try a shortlist and rank them for you.
  -> [api/models.md](API-Models)
- **One-call entry points** -- `fit(model, data, basis="auto")` fits the
  candidate bases and keeps the one with the lowest residual, so you never
  pick the basis yourself; `auto_forecast(x, y, horizon)` does a structured
  fit-then-extrapolate forecast with safety guards.
  -> [api/auto.md](API-Auto)
- **A scikit-learn estimator** -- `NonlineRegressor` plugs into `Pipeline`,
  `GridSearchCV`, `cross_val_score`. -> [api/estimator.md](API-Estimator)
- **Scaling backends** -- fit thousands of channels in one matrix multiply, or
  stream a dataset too big for memory in one pass.
  -> [api/scaling.md](API-Scaling)
- **Diagnostics** -- `fit_report` gives R^2/RMSE/AIC/BIC and residual tests so you
  can tell whether the fit is trustworthy. -> [api/diagnostics.md](API-Diagnostics)

---

## 6. The honest limits

Good documentation says where a method *doesn't* win. dtfit's are:

- **It needs a modest dynamic range.** The fingerprint is taken around a point,
  so a signal that spans many orders of magnitude should be normalized first
  (the docs show how). This is a real constraint, not a bug.
- **On a near-random-walk series** (like daily exchange rates), no parametric
  model beats "tomorrow = today" one step ahead -- and dtfit doesn't pretend to.
  Its streaming value there is *tracking and regime detection*, not beating a
  random walk.
- **It matches, rather than beats, a well-initialized `curve_fit`** on clean,
  simple shapes. Its edge shows up under noise, on outliers, in streaming, and at
  scale -- where it is more robust, faster, or cheaper, while still returning
  interpretable parameters.

These trade-offs, with real numbers, are documented per method in
[../methods/](Methods) and validated across application domains in
[../experimental/](Experimental).

Next: [methods-explained.md](Guides-Methods-Explained) walks through each method in
the same intuition-first style, including the proofs.
