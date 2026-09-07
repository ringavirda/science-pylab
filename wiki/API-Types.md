# API: `FittingResult`

## What it is

`FittingResult` is the object every batch fitter hands back. It is deliberately
**self-describing**: instead of returning a bare array of numbers, it bundles the
fitted **coefficients** together with everything needed to *interpret* them -- the
parameter **names**, the model **expression** and **variable**, an estimate of the
parameters' **uncertainty** (a covariance matrix), and a ready-to-call **model
function**. That bundle is enough to do four things without any extra context:

1. **Read the parameters by name** -- `res.params` -> `{'a': 3.0, 'w': 1.5}`.
2. **Predict** -- `res.predict(x)`, optionally with an error band.
3. **Quantify uncertainty** -- standard errors and confidence intervals.
4. **Serialize & redeploy** -- round-trip to a plain dict (JSON) and back.

The design goal is that a fit you computed today can be saved, shipped, reloaded
elsewhere, and still know what it represents -- no separate metadata to carry
around.

```python
FittingResult(coeffs, cov=None, expr=None, var=None, names=(), model=None,
              *, label=None, error=None, converged=None, message=None,
              x_range=None, param_model=None,
              n_obs=None, rss=None, tss=None, nfev=None, cost=None)
```

You rarely construct one yourself; the fitters return it. `FittingResult` is
picklable, so it survives a process-pool round trip. Batch and single fits
return this **same** type -- including [`fit_many`](API-Scaling#fit_many).

## Attributes

| attribute | type | meaning |
|---|---|---|
| `coeffs` | ndarray | fitted coefficients, ordered by parameter name for a symbolic model, by **signature order** for a callable-fit result |
| `cov` | ndarray \| None | parameter covariance (`nxn`) when the method produced one from an overdetermined system; else `None`. Diagonal square-roots are the standard errors |
| `expr` | str \| None | the model expression (enables serialization & error bands). **`None` for a callable-fit result** -- `predict`/`model` still work via `param_model`, but `to_dict` raises |
| `var` | str \| None | the main variable name (a label only for a callable-fit result) |
| `names` | tuple[str] | parameter names aligned with `coeffs` |
| `model` | callable | the fitted model `f(x) > y` (lambdified lazily from `expr`+`coeffs`, or -- for a callable-only result -- from `param_model` with the coefficients frozen) |
| `param_model` | callable \| None | a numeric, parameters-explicit evaluator `f(x, coeffs) -> y` set instead of `expr` for a **callable-fit** result; `predict(return_std=True)` finite-differences it for a band, and `model` falls back to it. `None` when the model is symbolic |
| `n_obs` | int \| None | number of observations in the fit; enables `aic`/`bic`. `None` when not recorded |
| `rss` | float \| None | residual sum of squares; enables `rsquared`/`aic`/`bic`. `None` when not recorded |
| `tss` | float \| None | total sum of squares of the data; enables `rsquared`. `None` when not recorded |
| `nfev` | int \| None | number of model evaluations the optimizer used. `None` when the method does not report it |
| `cost` | float \| None | final optimizer cost (typically `0.5 * rss` on the least-squares objective). `None` when not recorded |
| `converged` | bool \| None | whether the underlying optimizer reported convergence; `None` when the method does not report it. **A successful call with `converged is False` is the silent-failure case to watch** -- a result came back but the solver did not settle |
| `message` | str \| None | the optimizer's termination message, when available |
| `x_range` | tuple[float, float] \| None | `(min, max)` of the training `x`, recorded so [`predict`](#predictx-return_stdfalse-warn_extrapolationfalse) can warn on extrapolation; `None` when unknown |
| `label` | Any | optional tag carried through batch/parallel fits (a channel name, grid cell, ...); `None` for a plain single fit |
| `error` | str \| None | set to a message instead of coefficients when a fit failed inside a batch ([`fit_many`](API-Scaling#fit_many)), so one bad problem does not abort the batch; `None` for a success |
| `rss_source` | str \| None | where `rss` was computed: `"samples"` when the fit ran on an `Original`, `"image"` when it ran on an `Image`; `None` when not recorded |
| `image_order` | int \| None | the order of the image the fit ran on; `None` when not recorded |
| `basis_name` | str \| None | the basis of the image the fit ran on (`"legendre"` or `"block"`); `None` when not recorded |

## Methods & properties

### `params -> dict[str, float]`
Fitted parameters as a `{name: value}` mapping (the most common accessor).

```python
res.params          # {'a': 3.001, 'w': 1.498}
```

### `predict(x, *, return_std=False, warn_extrapolation=False)`
Evaluate the fitted model at `x`. With `return_std=True` it also returns a 1-sigma
prediction band.

**How the band is computed.** The uncertainty in the *parameters* (the covariance
`cov`) propagates into uncertainty in the *prediction* via the **delta method**:
the model is locally linearized at each `x` (numerically, by perturbing each
parameter), and the parameter covariance is pushed through that linearization to
give a per-point variance. So the band is wide where the model is sensitive to the
uncertain parameters and narrow where it isn't. This needs `cov` and either `expr`
or, for a **callable-fit** result (`expr is None`), the `param_model` evaluator --
so a callable-only fit still produces a band.

**Extrapolation guard.** With `warn_extrapolation=True` a `UserWarning` is issued
when any `x` falls outside the fitted training range (`x_range`) -- predicting past
the data is the most common curve-fitting mistake, and a nonlinear model can
extrapolate to nonsense. It is opt-in (default off) and a no-op when `x_range` is
unknown.

```python
y_hat = res.predict(x)
y_hat, sigma = res.predict(x, return_std=True)   # 1sigma band from parameter covariance
res.predict(x + 10, warn_extrapolation=True)     # warns since x + 10 leaves the fitted range
```

### `stderr() -> dict[str, float]`
Per-parameter standard errors (sqrt of the covariance diagonal). Raises if `cov` is
`None`.

### `confidence_intervals(level=0.95) -> dict[str, (lo, hi)]`
Per-parameter confidence intervals (normal approximation) at the given level.

<a name="fit-quality-diagnostics-v03"></a>
### Fit-quality diagnostics (`rsquared`, `aic`, `bic`, `residuals`)
The iterative fitters ([`fit_lsi`](API-Fitting#fit_lsi), [`fit_eac`](API-Fitting#fit_eac))
record the raw fit statistics (`n_obs`, `rss`, `tss`, plus the optimizer's `nfev`
and `cost`) on the result, and three read-only properties derive the usual
model-comparison numbers from them:

- **`rsquared -> float | None`** -- coefficient of determination `1 - rss/tss`.
  `None` when the fit did not record both `rss` and `tss`, or the data is constant
  (`tss == 0`).
- **`aic -> float | None`** / **`bic -> float | None`** -- Akaike / Bayesian
  information criteria, computed from `rss` and `n_obs`. `None` when either is
  unrecorded.
- **`residuals(x, y) -> ndarray`** -- the fit residuals `y - model(x)` at the given
  samples.

```python
r = fit_lsi(x, y, "a*exp(b*x)", "x")
r.rsquared, r.aic, r.bic          # fit-quality numbers straight off the result
resid = r.residuals(x, y)         # y - r.predict(x)
```

These are also reachable from the sklearn route via
[`NonlineRegressor.result_`](API-Estimator) and are round-tripped by
`to_dict`/`from_dict`. `summary()` appends `R^2` when it is available.

### `summary() -> str`
A short human-readable text summary -- parameters `+/-` standard errors when a
covariance is available. If the optimizer did not converge (`converged is False`)
it appends a warning line with the `message`.

```python
print(res.summary())
# FittingResult: a*atan(w*x)
#   a = 3.00123 +/- 0.0142
#   w = 1.49831 +/- 0.0119
```

### Checking convergence
The iterative fitters ([`fit_lsi`](API-Fitting#fit_lsi),
[`fit_eac`](API-Fitting#fit_eac)) record whether the solver settled. A fit can
return *successfully* yet not have converged -- on a misspecified model, a bad
seed, or degenerate data -- so check the flag before trusting the parameters:

```python
res = fit_lsi(x, y, "a*exp(b*x)", "x")
if res.converged is False:        # note: `is False`, not falsy (None = not reported)
    print("optimizer did not converge:", res.message)
```

It is also surfaced in [`fit_report`](API-Diagnostics) (a `"converged"` key when
the fit reports it).

### `to_dict() -> dict` / `from_dict(d) -> FittingResult`
JSON-friendly round-trip for storage/deployment. `to_dict()` captures everything
needed to rebuild the model -- `expr`, `var`, `names`, `coeffs`, `cov`, and
`x_range`, plus any recorded fit-quality diagnostics (`n_obs`/`rss`/`tss`/`nfev`/`cost`,
so `rsquared`/`aic`/`bic` survive) -- as plain Python types, so it serializes
cleanly to JSON; `from_dict()` reconstructs a fully functional `FittingResult` (the
callable model is re-lambdified lazily on first use), with the training range
preserved so a redeployed fit still warns on extrapolation. This is the recommended
way to **persist a fit** or **ship it to another process/service**. Requires
`expr`/`var` -- a **callable-fit** result (`expr is None`, holding only a
precomputed callable / `param_model`) cannot be serialized, since there'd be
nothing to rebuild from.

```python
import json
blob = json.dumps(res.to_dict())
res2 = FittingResult.from_dict(json.loads(blob))   # rebuilt, ready to predict
```

## Notes

- The `model` callable broadcasts scalars: a constant model still returns an array
  matching `x`.
- `cov` is `None` for exactly- or under-determined fits (e.g. EAC with
  `n_windows == n_params`); uncertainty methods then raise with a clear message.
- Parameter **order is by sorted name** for a symbolic model, consistently across
  the whole library -- so `coeffs`, `names`, and `params` always agree. A
  **callable-fit** result instead follows the callable's **signature order**;
  `coeffs`/`names`/`params` still agree, just in that layout.
