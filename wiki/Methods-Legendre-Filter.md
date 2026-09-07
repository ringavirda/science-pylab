# ImageFilter -- recursive estimation on the window image

> Numeric **online** method. Source:
> [`streaming/filter.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/src/dtfit/streaming/filter.py).
> Invoke via `ImageFilter(model, var, p0=, window_size=, order=, basis=,
> ...)` then `flt.partial_fit(t, y)` per sample; `flt.predict(x)`,
> `flt.params_`, `flt.result()`.

ImageFilter tracks the parameters of `f(t; theta)` online by treating each
sliding window of the stream as [an image](Methods-Image) `(S_w, G_w)` in a
chosen basis -- the same statistic a batch fit builds -- and updating it
one sample at a time in information form. `LSIFilter` and `EACFilter` fix
its basis to Legendre and block, the streaming twins of
[LSI](Methods-LSI) and [EAC](Methods-EAC).

## Mathematical grounding

The state is the parameter vector `theta`, modelled as a random walk
`theta_t = theta_{t-1} + w_t`, `w_t ~ N(0, Q)`. Over the current window the
measurement is the window image `S_w = Phi_w^T y`, `G_w = Phi_w^T Phi_w`,
`Phi_w` the basis evaluated at the window's sample positions mapped to
`[-1, 1]` -- cached against the window length while the normalized
positions repeat, rebuilt in `O(W K)` when the length changes; `S_w` and
the innovation are recomputed every step, since the window's positions
shift by one sample every update. With
`S_f(theta) = Phi_w^T f(t_w; theta)` the model's projection on the same
window and `G_w = L_w L_w^T`, the **whitened innovation** and Jacobian are

$$
e = L_w^{-1}\big(S_w - S_f(\theta)\big),
\qquad
H = L_w^{-1}\, \Phi_w^{\top}\, \frac{\partial f}{\partial \theta} .
$$

The measurement noise is `s2 I` with `s2` an EWMA of the window's model
residual variance (or a fixed `noise_var`). The update is the
**information form** of a Kalman correction, which needs no matrix
inverse of the measurement covariance:

$$
A = \frac{H^{\top} H}{s^2}, \qquad
b = \frac{H^{\top} e}{s^2}, \qquad
P_{\text{post}} = \big(P^{-1} + A\big)^{-1}, \qquad
\theta \leftarrow \theta + P_{\text{post}}\, b ,
$$

followed by `P <- P_post + Q`, symmetrized. `P` is a gain state, not a
calibrated covariance -- the calibrated read-out is [`result()`](#result).
A step that raises the window's whitened misfit `e^T e` is halved, up to
eight times, before it is skipped; a nonlinear model can otherwise
overshoot on the linearized step.

`nis_`, the normalized innovation squared reported after each update, is
`e^T e / s2` corrected for the information gained in this step
(`b^T P_post b`), chi-square with `n_coef` degrees of freedom under the
model.

## Drift detection

Every `W` samples once the window is full -- or the shorter, current
adaptive window length while it is still growing, so the detector
tests sooner and more often before the window reaches its cap -- the
whitened innovation is rotated by a Householder reflection so its first
component is the window-mean channel (the direction that carries a level
shift in any basis), and tested by a
[`DriftDetector`](API-Streaming#driftdetector) shared with
`ImageStream(detect=...)`: a jump test on the innovation energy against
an exponentially weighted baseline, and a two-sided CUSUM on the first
component. On a detection, `drift_reset="inflate"` (the default)
multiplies `P` by `drift_inflation` and keeps the window; `"full"` resets
`P` to its initial value and clears the window. Either way the adaptive
window collapses to `min_window` and the detector's own baselines
restart.

## The adaptive window

`adaptive_window=True` (the default) sizes the window from the data: it
grows from `min_window` by one sample per update while the model still
fits, and shrinks by one while an EWMA of the window residual's lag-1
autocorrelation stays above 0.35 -- a curve the model is lagging or
missing the dynamics of. A static model grows to `window_size`; a
manoeuvring signal settles where the model still tracks it. `False` keeps
the window fixed at `window_size`.

## Robustness

`robust=True` winsorizes each sample's residual to the current model at
`huber_c` MAD sigmas around the window's median residual before it enters
the image: a single spike cannot carry into `S_w`, while a sustained
shift still passes through unclipped and reaches the drift test.

## The two bases

`basis="legendre"` (`LSIFilter`) resolves an oscillatory plant's shape
and frequency: the window's Legendre spectrum carries the amplitude,
frequency and phase directly. `basis="block"` (`EACFilter`) is the
cheapest statistic -- window sums, a diagonal `G_w` -- and the one an
embedded target runs; see [the embedded tool](Domain-Embedded-Control).
Both are the same recursion on the same image type, differing only in
`Phi_w`.

<a name="result"></a>
## `result()` -- the calibrated read-out

`P` tracks information, not uncertainty: a run of easy samples shrinks it
regardless of how well the parameters actually match the plant.
`result()` instead runs a batch fit ([`fit`](Methods-Image#the-projected-estimator))
on the current window, in the filter's own basis and order, started from
the current estimate -- the window's parameters, covariance, standard
errors and prediction band in the same type a batch fit returns, imaged
robustly when the filter is robust. Measured in tracking: the
covariance's nominal interval covers the actual filter error 70 to 96
percent of the time, and is conservative (wider than needed) whenever
the process noise `q_diag` is small relative to the true drift rate.

## Coasting

`coast(x, order=1|2)` and `coast_cov` dead-reckon past the window from
its last ingested sample by a Taylor expansion of the current model,
anchored at that sample; an external-regressor model splits into its
extrapolable and nuisance parts.

## Algorithm (per `partial_fit`)

1. Ingest `(t, y[, regressors])`; a non-finite sample is skipped with a
   warning, leaving every state untouched. Feed the attached `stream`, if
   any, before the window is mutated.
2. Evict past the window cap (the current adaptive length, else `W`);
   return early below `min_window`.
3. Build `Phi_w`, its Cholesky factor and the mean-channel rotation
   (cached against the window length while the normalized positions
   repeat); winsorize the residual first when `robust`.
4. Compute the whitened innovation `e` and Jacobian `H`; reject a
   non-finite pair, keeping the last good estimate.
5. On a full-window stride, run the drift test; a detection re-arms and
   returns.
6. Otherwise take the damped information-form step, update `s2`, `p` and
   `P`, and adjust the adaptive window length from the residual
   autocorrelation.

## Where it is best applied

**Use `ImageFilter` for:** real-time tracking of a stream where the
parameters drift, at bounded per-sample cost, with regime-change
detection. Pick `basis="legendre"` (`LSIFilter`) for oscillatory or
sustained-cycle plants; `basis="block"` (`EACFilter`) for monotone or
saturating ones, or where the per-sample cost must be the smallest
possible. `result()` is the calibrated uncertainty, not `P` directly.
For a static batch fit use [LSI](Methods-LSI) or [EAC](Methods-EAC); for
several streams pooled into one fault test, see
[several streams](API-Streaming#several-streams).
