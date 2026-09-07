# API: streaming / online estimators

`ImageFilter` ingests one sample at a time and tracks parameters online at
bounded per-update cost, for control loops and live streams. The model and
its Jacobian are compiled once at construction; every update is pure NumPy.
Concept: [../guides/methods-explained.md#streaming](Guides-Methods-Explained#streaming);
math: [../methods/legendre-filter.md](Methods-Legendre-Filter).

- [`ImageFilter`](#imagefilter) -- recursive estimation on the window image
- [`LSIFilter`, `EACFilter`](#aliases) -- the Legendre and block basis aliases
- [`DriftDetector`](#driftdetector) -- the shared change-detection logic
- [Several streams](#several-streams) -- pooling `nis_` across filters

---

<a name="imagefilter"></a>
## `ImageFilter`

```python
ImageFilter(model, var, *, basis="legendre", order=None, window_size=50,
            min_window=None, adaptive_window=True, q_diag=None,
            noise_var=None, p0=None, regressors=None, param_names=None,
            robust=False, huber_c=3.0, drift_reset="inflate",
            drift_inflation=100.0, alpha=0.001, cusum_k=0.5, cusum_h=5.0,
            stream=None)
```

The measurement is the window image `(S_w, G_w)`, recomputed per sample on
the window's own domain: `S_w = Phi_w^T y`, `G_w = Phi_w^T Phi_w` with
`Phi_w` the basis at the window's sample positions. The innovation
`S_w - Phi_w^T f(t; p)` and the Jacobian `Phi_w^T df/dp` are whitened by the
Cholesky factor of `G_w`, so the measurement noise is `s2 I` with `s2` an
EWMA of the window residual variance, and the gain state `(p, P)` is
updated in information form with random-walk process noise `Q`. A step
that raises the window's whitened misfit is halved, up to eight times,
before it is skipped. `P` is a gain state, not a calibrated covariance --
that is [`result()`](#result).

**Presets** -- most callers should start from a preset classmethod and
only pass overrides:

- `ImageFilter.tracking(model, var, **overrides)` -- `adaptive_window=True`
  (the default); `overrides` win.
- `ImageFilter.robust(model, var, **overrides)` -- `robust=True` and
  `drift_reset="inflate"`; `overrides` win.

**Constructor arguments**

| name | default | meaning |
|---|---|---|
| `model`, `var` | -- | a SymPy-expression string, a `sympy.Expr` or a callable `f(t, *params)`; a symbolic model may reference `regressors`. A callable has no time derivatives, so `coast` / `coast_cov` raise for it and it accepts no regressors. `var` is a label only for a callable |
| `basis` | `"legendre"` | `"legendre"`, `"block"`, or a `Basis` instance (which fixes `order`) |
| `order` | `None` | the Legendre order, or the window count of the block basis, default 5; the image needs at least as many coefficients as the model has parameters. A `Basis` instance carries its own order |
| `window_size` | `50` | the window cap `W` (samples), at least the image's coefficient count plus one |
| `min_window` | `None` | samples from which the filter measures; defaults to `order + 2` (Legendre) or `2 * order` (block, two samples per window), clamped to `[that floor, window_size]` |
| `adaptive_window` | `True` | grow the window from `min_window` by one sample per update while the model fits it, shrink by one while the residual is autocorrelated (EWMA lag-1 autocorrelation above 0.35), collapse to `min_window` on a drift; `False` keeps the window at `window_size` |
| `q_diag` | `None` | process-noise variances per parameter, default 0.01 each |
| `noise_var` | `None` | the measurement variance `s2`; `None` estimates it as an EWMA of the window's model residual variance |
| `p0` | `None` | initial parameters in canonical order (sorted names for a symbolic model, signature order for a callable), default ones |
| `regressors` | `None` | names of external-regressor symbols in a symbolic model; every `partial_fit` and `predict` then takes their values |
| `param_names` | `None` | a callable's parameter names when its signature cannot be introspected |
| `robust` | `False` | winsorize each sample's residual to the current model at `huber_c` MAD sigmas around the window's median residual before imaging |
| `huber_c` | `3.0` | the winsorization threshold in robust sigmas |
| `drift_reset` | `"inflate"` | `"inflate"` multiplies `P` by `drift_inflation` and keeps the window; `"full"` resets `P` and clears the window |
| `drift_inflation` | `100.0` | the factor of `"inflate"` and of `inflate()` |
| `alpha` | `0.001` | significance of the detector's jump test; `1e-15` disables it |
| `cusum_k` | `0.5` | CUSUM slack in standard deviations; `inf` disables the CUSUM |
| `cusum_h` | `5.0` | CUSUM decision threshold |
| `stream` | `None` | an `ImageStream` that receives every ingested `(t, y)` as well, for a running whole-stream image |

**Methods**

- `partial_fit(t_new, y_new, regressors=None) -> self` -- ingest one sample
  and update in place; `update` is an alias. A non-finite sample is skipped
  with a `RuntimeWarning`. A sample is ingested but not measured when the
  window is not yet full enough, a block window holds an empty bin, the
  step never lowers the window's whitened misfit, or the innovation or the
  Jacobian comes out entirely non-finite.
- `result(**kwargs) -> FittingResult` -- see [below](#result).
- `predict(x, regressors=None) -> ndarray` -- the model at the current
  estimate on `x`.
- `predict_cov(x, regressors=None) -> ndarray` -- the output variance `P`
  implies at `x` (the delta method); the calibrated covariance is
  `result()`.
- `coast(x, *, order=1, regressors=None) -> ndarray` -- dead-reckon past the
  window from its last sample; `order=1` is constant-velocity, `order=2`
  adds constant acceleration. Reduces to `predict` at and before the
  anchor. Raises for a callable model, or for a regressor model without
  `regressors`.
- `coast_cov(x, *, order=1) -> ndarray` -- the variance of `coast(x)` from
  `P`, growing with the gap length. Raises for a callable model or a
  regressor model.
- `inflate(factor=None) -> None` -- multiply `P` by `factor` (default
  `drift_inflation`); the hook for an external change detector.

**Attributes**

- `p`, `P`, `Q` -- the parameter estimate, the gain state and the process
  noise.
- `params_` -- `{name: value}` of the current estimate.
- `innovation_` -- the last whitened innovation (NaN before the first
  measurement).
- `nis_` -- the last normalized innovation squared, chi-square with
  `n_coef` degrees of freedom under the model; NaN before the first
  measurement. See [several streams](#several-streams).
- `last_residual_` -- the one-step residual `y - f(t; p)` at the newest
  sample before the update; NaN before the first measurement.
- `detector` -- the `DriftDetector` (see [below](#driftdetector)). It sees
  the innovation rotated so its first component is the window-mean
  innovation, the direction channel for every basis.
- `n_drifts_`, `drift_flag_`, `last_drift_direction_` -- detections so
  far, whether the last update detected one, and its direction (`1` up,
  `-1` down, `0` before any).

```python
from dtfit import ImageFilter

flt = ImageFilter.tracking("a*exp(b*t)", "t", window_size=40)
t = np.linspace(0, 8, 200)
b_true = np.where(t < 4, 0.30, 0.90)
y = np.exp(b_true * t) + rng.normal(0, 0.05, t.size)
for ti, yi in zip(t, y):
    flt.partial_fit(ti, yi)
print(flt.params_, flt.n_drifts_)
```

<a name="result"></a>
### `result()`

```python
result(**kwargs) -> FittingResult
```

A batch fit of the model on the current window, with the filter's basis
and order, started from the current estimate: the window's parameters,
covariance, standard errors and prediction band in the same
[`FittingResult`](API-Types) a batch fit returns. The window is imaged
robustly when the filter is robust. A regressor model is fitted as a
callable closed over the window's regressor columns, interpolated
linearly onto whatever grid a diagnostic evaluates and exact on the
window grid the fit runs on. `kwargs` go to [`fit`](API-Fitting#fit)
(`bounds`, `solver_options`). Raises `ValueError` for fewer than
`min_window` samples ingested.

Measured in tracking: the covariance covers the filter error at 0.7 to
0.96, conservative with small process noise.

```python
flt = ImageFilter("a*exp(b*t)", "t", p0=[1.0, 0.1], window_size=40)
for ti, yi in zip(t, y):
    flt.partial_fit(ti, yi)
res = flt.result()
print(res.stderr())
```

---

<a name="aliases"></a>
## `LSIFilter`, `EACFilter`

```python
LSIFilter(model, var, **kwargs) -> None
EACFilter(model, var, **kwargs) -> None
```

`ImageFilter` with `basis` fixed to `"legendre"` and `"block"`
respectively -- the streaming twins of `fit_lsi` and `fit_eac`. Each takes
every `ImageFilter` keyword except `basis`, which raises `TypeError`.
`LSIFilter`'s window image resolves the shape and frequency of an
oscillatory plant; `EACFilter`'s block image is the cheaper statistic,
the one an embedded target runs.

```python
from dtfit import LSIFilter

t = np.linspace(0, 20, 400)
y = 2.0 * np.sin(1.3 * t) + rng.normal(0, 0.05, t.size)
flt = LSIFilter.tracking("A*sin(w*x)", "x", p0=[1.0, 1.0])
for ti, yi in zip(t, y):
    flt.partial_fit(ti, yi)
print(flt.params_)
```

---

<a name="driftdetector"></a>
## `DriftDetector`

```python
DriftDetector(dim, *, alpha=0.001, cusum_k=0.5, cusum_h=5.0, ewma=0.15,
              warmup=20)
```

Two tests on successive whitened innovations `e` of dimension `dim`: a
jump test on the innovation energy `e @ e` against an exponentially
weighted baseline, and a two-sided CUSUM on the first component
standardized by its own baseline. Shared by `ImageFilter` and
`ImageStream(detect=...)`.

| name | default | meaning |
|---|---|---|
| `dim` | -- | innovation length, at least 1 |
| `alpha` | `0.001` | nominal significance of the jump test before a 1.6 safety factor; the energy ratio threshold is `1.6 * chi2.ppf(1 - alpha, dim) / dim` |
| `cusum_k` | `0.5` | CUSUM slack in standard deviations, non-negative |
| `cusum_h` | `5.0` | CUSUM decision threshold in accumulated standard deviations, positive |
| `ewma` | `0.15` | forgetting weight of the baselines in `(0, 1]` |
| `warmup` | `20` | innovations that only build the baselines after construction and after every detection; the baselines are bias-corrected exponential averages |

- `update(innovation) -> bool` -- test one whitened innovation (any shape
  flattening to length `dim`); returns `True` on a detection, after which
  the detector has reset itself. Raises `ValueError` for the wrong length
  or a non-finite entry.
- `state() -> dict`, `restore(state)` -- the complete detector state as
  plain Python numbers, and loading it back into a detector built with the
  same parameters.

Attributes: `n_tests_` (innovations seen since the last reset), `n_drifts_`
(detections since construction), `flag_` (whether the last `update`
detected a change), `last_direction_` (`1` up, `-1` down, `0` before any),
`threshold` (the energy ratio threshold).

```python
from dtfit.streaming import DriftDetector

det = DriftDetector(2, warmup=5)
for _ in range(5):
    det.update(rng.normal(0, 1, 2))
print(det.update(rng.normal(0, 1, 2)), det.n_tests_)
```

---

<a name="several-streams"></a>
## Several streams

`nis_` is chi-square under the model, so several filters' `nis_` sum to a
chi-square with the summed degrees of freedom, giving the pooled test
more degrees of freedom and power than any one filter's innovation
alone.

```python
from scipy.stats import chi2

K = 3
flts = [LSIFilter("A*sin(1.2*t + p)", "t", p0=[1.0, 0.0], window_size=40,
                  order=4, adaptive_window=False, alpha=1e-15,
                  cusum_k=float("inf"))
        for p in (0.7 * k for k in range(K))]
dof = sum(f.basis.n_coef for f in flts)
threshold = chi2.ppf(1 - 1e-4, dof)
amp = np.where(t < 10, 1.0, 0.5)
for i in range(t.size):
    for k, f in enumerate(flts):
        f.partial_fit(t[i], amp[i] * np.sin(1.2 * t[i] + 0.7 * k))
    pooled = sum(f.nis_ for f in flts)
    if np.isfinite(pooled) and pooled > threshold:
        break
print(i, pooled > threshold)
```

A bank of filters driven in lockstep and a fused detector built on this
reduction live in `dtfit_experimental.streaming` (`FilterBank`,
`FusedChiSquareDetector`) for the experiment harnesses; see
[the experimental adaptations API](Experimental-Adaptations-API).
