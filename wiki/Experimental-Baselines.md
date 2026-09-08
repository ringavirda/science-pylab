# The baselines -- what they are and why they were chosen

To claim a method is good you must compare it against what people **actually use**
today. The experiment suite therefore measures `dtfit` against a broad set of
*established* baselines -- the methods a working practitioner in each domain would
reach for. This page explains each one in plain terms: **what it does**, **how it
works**, and **why it earns a place in the comparison**.

Source: [`experiments/common/baselines.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/common/baselines.py).
Deep-learning and statsmodels backends are imported lazily and skipped (not
faked) when not installed, so the suite still runs on a core install.

A guiding principle runs through the selection: each baseline is the **fair,
same-job counterpart** to a dtfit method. A black-box learner that fits a curve
but recovers *no physical parameters* is the right foil for showing dtfit
recovers parameters; a robust NLLS is the right foil for dtfit's outlier
robustness; an EKF is the right foil for the streaming filters; and so on.

---

## 1. Classical curve fitting (the parameter-estimation foils)

### SciPy `curve_fit` -- Levenberg-Marquardt nonlinear least squares
**What:** the standard tool for fitting a known nonlinear model. **How:** guesses
parameters, measures the point-by-point squared error, and iteratively nudges the
parameters downhill (Levenberg-Marquardt / trust-region). **Why chosen:** it is
the **gold standard** for parameter recovery -- the number every other estimator
is judged against. dtfit's goal is to *match* it on clean data and *beat* it under
noise/outliers/streaming while being more robust. It is the most important
baseline in the suite.

### Robust NLLS (`least_squares` with a robust loss)
**What:** `curve_fit`'s outlier-resistant sibling. **How:** the same nonlinear
least squares, but large residuals are **down-weighted** by a robust loss
(soft-L1 / Huber / Cauchy) so a few wild points don't dominate. **Why chosen:** it
is the *established* way to fit a known model when outliers are present -- the
direct, fair comparison for EAC's robust-loss and the ensemble adaptation.

### `numpy.polyfit` -- polynomial least squares
**What:** fit a plain polynomial. **How:** linear least squares on powers of `x`.
**Why chosen:** the simplest *linear-in-parameters surrogate*. It can trace a
curve but returns **opaque polynomial coefficients**, not the physical parameters
-- so it illustrates exactly what dtfit provides that a generic curve fit doesn't.

### Gaussian-process regression (`gp_curve`)
**What:** a flexible nonparametric Bayesian smoother. **How:** models the curve as
a draw from a distribution over smooth functions and conditions on the data,
giving a fitted curve *and* uncertainty. **Why chosen:** the standard
**nonparametric** smoother -- fits any smooth shape beautifully but recovers **no
physical parameters**. It is the nonparametric counterpart to dtfit's *structured*
fit: a fair way to ask "is structure worth it?"

### MLP curve fit (`mlp_curve`)
**What:** a small neural network used as a black-box `f(x)>y`. **How:** a
multilayer perceptron trained to map input to output. **Why chosen:** the
black-box-learner foil -- high flexibility, zero interpretability -- to contrast
with dtfit recovering the actual model parameters.

---

## 1b. The Western parameter-estimation lineage (the signal-processing / system-ID foils)

dtfit's EAC and LSI descend from the Pukhov differential-transformation school,
which is largely invisible in the Western literature. A reviewer at a Western
signal-processing or system-identification venue will instead reach for the
*Western* methods that solve the **same** problem -- recovering nonlinear
parameters (rates, frequencies, amplitudes) from noisy data. These baselines are
those methods, so the comparison is in the reviewer's own vocabulary. Each takes
a different route to the same goal, which is exactly what makes the head-to-head
informative. Source: the same `baselines.py`; all are pure NumPy/SciPy.

### Prony's method (`prony_fit`)
**What:** the 1795 original of the whole exponential-fitting lineage -- fit a sum
of exponentials. **How:** an exponential sum obeys a **linear recurrence**, so its
discrete poles are the roots of a characteristic polynomial recovered by two
linear solves (the recurrence coefficients, then the amplitudes). Non-iterative
and exact without noise. **Why chosen:** it is the **algebraic** counterpart to
dtfit's integral route -- same parameters, opposite mechanism (rooting a
difference equation vs matching integral moments). Including it shows the
contrast honestly: classical Prony is **noise-sensitive** (it trails on the
exponential and breaks down on a noisy sinusoid in the suite), which is precisely
why its subspace successors exist.

### Matrix Pencil / ESPRIT (`matrix_pencil_fit`)
**What:** the modern, SVD-robust successor of Prony (Hua & Sarkar 1990), in the
same family as **ESPRIT** and **MUSIC** -- the methods a signal-processing
reviewer actually names. **How:** build a Hankel matrix from the samples, denoise
it by truncating its SVD to the signal subspace, and read the poles off a
one-row **shift on that subspace** (eigenvalues), rather than rooting a noisy
polynomial. **Why chosen:** it is the established high-accuracy answer to "how
does this compare to ESPRIT for exponential/sinusoid fitting?" In the suite it
ties dtfit and gold NLLS on rate/frequency recovery -- the fair, strong
competitor on the families it is designed for (sums of damped sinusoids).

### Variable projection -- VarPro (`varpro_fit`)
**What:** Golub & Pereyra's 1973 method for **separable** nonlinear least squares
(the standard tool in modern exponential / Fourier fitting software). **How:**
many models are linear in some parameters (amplitudes) and nonlinear in the rest
(rates / frequencies); VarPro eliminates the linear ones in closed form and
optimises only over the smaller, better-conditioned nonlinear set. **Why chosen:**
it is the structural cousin of LSI (also linear in an amplitude once projected
onto its basis) and the method that converges fastest on multi-exponential /
harmonic models -- the fair "best-in-class separable NLLS" foil.

### Method of moments / GMM (`moment_match_fit`, wired as `est_moment`)
**What:** the classical statistical estimator (Pearson 1894; Hansen's GMM 1982),
here in its deterministic integral-moment form. **How:** match the model's first
`m` **integral moments** to the data's. **Why chosen:** this is the *same*
integrate-don't-sample idea as EAC/LSI but with **monomial** test functions,
which form an ill-conditioned Hilbert-like system. It is therefore the **direct
ancestor of LSI before its Legendre reconditioning** -- the honest "what does
switching to an orthogonal basis buy?" foil. It is wired into the
parameter-estimation recovery table (`est_moment`) and, as expected, holds up on
low-parameter families and trails as the parameter count (and the monomial
ill-conditioning) grows.

> **Positioning, in one line.** The EAC equal-areas criterion is a Galerkin
> weighted-residual identification with piecewise-constant (Haar) test functions;
> LSI is a weighted integral least-squares in an orthogonal Legendre basis. Both
> descend from the Pukhov differential-transformation school and can be derived
> independently from classical moment-matching principles -- so the methods above
> are not just competitors but *relatives*, which is why they make the fairest
> baselines.

---

## 2. Forecasting baselines (the time-series toolkit)

These train on a series and predict `horizon` steps ahead. The set spans the
*entire* spread a forecasting practitioner uses, from trivial benchmarks to
competition winners and deep nets -- so a win is meaningful.

### Naive random walk (`random_walk_forecast`)
**What:** "tomorrow = today." **How:** persist the last observed value. **Why
chosen:** the famously **hard-to-beat benchmark**, especially for financial
series. The suite reports honestly that on near-random-walk data *nothing*
(including dtfit) beats it one step out -- using it keeps the claims grounded.

### Seasonal-naive (`seasonal_naive_forecast`)
**What:** "this season repeats." **How:** copy the values from one period ago.
**Why chosen:** the standard **seasonal** benchmark -- the minimum bar for any
method claiming to handle seasonality.

### Drift method (`drift_forecast`)
**What:** random walk **with** a trend. **How:** extrapolate the average per-step
change (Hyndman's drift method). **Why chosen:** the standard trend benchmark; a
fair, trivial competitor for dtfit's linear/structured trend forecasts.

### Polynomial extrapolation (`poly_extrap_forecast`)
**What:** fit a global polynomial and continue it. **How:** `polyfit` then evaluate
past the end. **Why chosen:** the **surrogate-fit** baseline -- extrapolates by raw
curvature with no parametric structure, so it shows the value of dtfit's
structured extrapolation (and the danger of unstructured extrapolation).

### Holt-Winters / ETS (`ets_forecast`)
**What:** exponential smoothing with level + optional trend + season. **How:**
recursively updates a smoothed level/trend/seasonal state (statsmodels
`ExponentialSmoothing`). **Why chosen:** the **workhorse** classical forecaster --
ubiquitous in practice, the realistic bar to clear.

### Theta method (`theta_forecast`)
**What:** a robust decomposition forecaster. **How:** combines a long-run trend
line with a short-run smoothed component (statsmodels `ThetaModel`). **Why
chosen:** the **M3-competition winner** -- a famously strong, simple method, so
beating it is a strong result.

### (S)ARIMA (`arima_forecast` / `sarima_forecast`)
**What:** the classic statistical time-series model. **How:** models the series as
auto-regressive + moving-average terms on differenced (and optionally seasonal)
data. **Why chosen:** the **standard statistical** forecaster taught and deployed
everywhere; the canonical comparison for any new forecaster.

### MLP and LSTM forecasters (`mlp_forecast`, `torch_mlp_forecast`, `lstm_forecast`)
**What:** neural-network sequence forecasters (a feed-forward net on a lookback
window, and a recurrent LSTM). **How:** trained on sliding windows to predict the
next value, then rolled forward recursively. **Why chosen:** the **modern
machine-learning** foils. The LSTM in particular represents deep sequence models.
(Cutting-edge deep forecasters -- DLinear/TimesNet/Time-LLM -- are **not**
re-implemented; instead the suite compares against their *published* benchmark
numbers in `experiments/06_benchmark_ltsf`, which is the fair way to cite them.)

---

## 3. Online / streaming baselines (the real-time foils)

These ingest one sample at a time -- the fair competitors for `EACFilter` /
`LSIFilter`, which also run online.

### Extended Kalman Filter for parameters (`EKFParam`)
**What:** the textbook method for **online nonlinear parameter estimation**.
**How:** treats the parameters as a slowly-drifting state, takes each sample's
*pointwise* value as the measurement, and linearizes the model about the current
estimate via its parameter-Jacobian (compiled once with SymPy). **Why chosen:**
the **same-job** baseline for the dtfit filters -- both track a known model's
parameters online; the *only* difference is the measurement (a pointwise value
here vs an integrated **area/spectrum** for dtfit). That makes it the cleanest
possible test of dtfit's "integrate, don't sample" thesis in the streaming
setting.

### Recursive Least Squares (`RLSPredictor`)
**What:** the classic online adaptive filter. **How:** tracks the coefficients of
a linear auto-regressive predictor online with a forgetting factor so it adapts to
drift. **Why chosen:** the standard online **system-identification / adaptive-
filtering** algorithm -- but a *black box* that yields one-step predictions and **no
physical parameters**. The streaming counterpart to the MLP foil.

### Constant-acceleration Kalman filter (`KalmanCA`)
**What:** the trajectory-tracking gold standard. **How:** a per-axis Kalman filter
with a position/velocity/acceleration state and position measurements. **Why
chosen:** the established method for **tracking and short-horizon trajectory
prediction** -- the right baseline for the GPS-trajectory and embedded-control
domains. It is wired to expose the *same* self-calibrating re-arming hook
(`inflate`) and innovation statistics as the dtfit filters, so the maneuver-
detection comparison is driven by identical machinery -- a fair fight.

### Incremental MLP (`mlp_forecast(..., incremental=True)`)
**What:** a neural net trained by `partial_fit` over mini-batches. **How:** the
sklearn MLP updated incrementally rather than in one batch. **Why chosen:** the
**streaming-friendly neural** baseline used by the big-data domain -- the online
black-box learner to contrast with dtfit's online structured filter.

---

## 4. The classical stochastic model (the same-model, classical-estimator foil)

`dtfit.stochastic.fit_stochastic` characterizes a random series by fitting the
*deterministic functionals* of the process (its ACF, spectrum, aggregated
variance, trend/cycle) with dtfit's own integral fitters. To test whether that
route actually **improves on the textbook toolkit**, the suite ships a *classical
twin* -- `fit_classical_stochastic`
([`classical_stochastic.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/common/classical_stochastic.py))
-- that runs the **same gated routing and the same regime-appropriate forecasters**
but swaps every estimator for its established classical counterpart:

| quantity | dtfit route | classical route |
|---|---|---|
| unit root | vendored ADF | `statsmodels` ADF (`adfuller`) |
| trend slope | LSI line fit | OLS line |
| cycle period | damped-cosine ACF fit (LSI) | FFT periodogram peak |
| AR(1) phi | LSI exp fit to the ACF | OLS regression `x_t ~ x_{t-1}` |
| Hurst H | LSI power-law spectral fit | detrended fluctuation analysis |
| vol persistence | LSI exp fit to \|resid\| ACF | GARCH(1,1) Gaussian QMLE |
| vol-cluster gate | ACF significance | `statsmodels` ARCH-LM test |

Because the routing and gates are identical, the comparison isolates **the
estimator**, not the pipeline. `fit_classical_stochastic` returns a
`ClassicalStochasticModel` with the same surface (`regime`, `components`, the
`has_*` flags, the parameters, `forecast`) as the dtfit model, so the two drop
into one harness. GARCH is fit by a compact in-house Gaussian QMLE (the
conditional-variance recursion run as a one-pole IIR filter), so no `arch`
package is needed.

**Why chosen / what it shows.** This closes the two previously-unbaselined
stochastic experiments (E4 volatility persistence, E6 trend+cycle decomposition)
and adds two head-to-heads: a **regime-identification** contest (E7) and a
**held-out forecast** contest (E8b, `exp_model_comparison`). The honest finding it
produces: on clean synthetic processes the mature classical estimators (GARCH
QMLE, OLS, periodogram) are *competitive or slightly better* on raw per-parameter
accuracy and short-horizon forecast, while **dtfit's clear advantage is unified
regime identification** -- it routes long-memory and white-noise series correctly
where DFA-based detection and ARCH gates misfire (E7: dtfit ~96% vs classical
~86%). That is the defensible "what dtfit buys here" claim, reported with its
limits rather than overstated.

**On GARCH specifically (a deliberate design choice, not a gap to patch).** QMLE
recovers the persistence more accurately than dtfit's ACF-decay fit (~0.3% vs
~0.9%), and we keep it that way *on purpose*: dtfit's `garch_persistence` is an
optimizer-free differential-transformation functional fit, whereas QMLE is a
model-specific likelihood maximization -- not a DT method. The efficiency gap is
**intrinsic**: QMLE uses the full conditional likelihood, which carries strictly
more information than any single functional. We verified there is no DT-native way
to close it -- applying dtfit's own EAC area-matching directly to the GARCH
variance recursion (so it uses the GARCH structure exactly as QMLE does) is
*worse* than the ACF fit, not better. So dtfit stays a fast, no-per-model-optimizer
functional estimator that is competitive on persistence, and the efficient QMLE
remains the (correctly attributed) classical baseline rather than being absorbed
into the library.

---

## Why this particular set

Three deliberate choices shape the list:

1. **Same-job pairing.** Every dtfit method has a matched, established competitor
   doing the *same* task with the *same* information: gold-standard NLLS for batch
   recovery, robust NLLS for outliers, the Western signal-parameter lineage
   (Prony / Matrix Pencil / ESPRIT / VarPro / method of moments) for the *same*
   nonlinear-parameter recovery by different routes, GP/MLP for "structure vs
   black box," EKF/Kalman/RLS for streaming. This isolates *what dtfit changes*
   (the integral measurement and the fingerprint formulation) rather than
   confounding it with a different problem setup.
2. **Real runnable code, honestly skipped.** Baselines are actually executed on the
   same data, not quoted from memory -- except modern deep LTSF models, which are
   compared via their *published* numbers (re-implementing them faithfully would be
   its own research project, so citing their benchmark is the fair move).
3. **Including the ones dtfit can't beat.** The random walk on FX, the Lorentzian
   for NLLS -- these stay in precisely so the comparison is honest. A method that
   only ever shows its wins isn't validated; it's advertised.

See the per-domain reports (`domains/*/report.md`) for the full tables, and
[README.md](Experimental) for the suite overview.
