# API: `NonlineRegressor`

A scikit-learn-compatible estimator over [`fit`](API-Fitting#fit). It exposes
the standard `fit` / `predict` / `score` API (plus `get_params` /
`set_params` from `BaseEstimator`), so it composes with
`sklearn.pipeline.Pipeline`, `GridSearchCV`, and `cross_val_score`. The basis
and the order of the image are estimator parameters, so a grid search tunes
them like any other hyperparameter.

It lives in `dtfit.sklearn` and is the only part of dtfit that imports
scikit-learn: `import dtfit` does not.

```python
from dtfit.sklearn import NonlineRegressor

NonlineRegressor(expr="a0 + a1*x", var="x", param_names=None, basis="auto",
                 order=None, bounds=None, p0=None, random_state=0,
                 robust=False, nan_policy="raise")
```

## Constructor arguments

| name | default | meaning |
|---|---|---|
| `expr` | `"a0 + a1*x"` | model, e.g. `"a0 + a1*x + a2*exp(a3*x)"` -- a SymPy-expression string, a `sympy.Expr`, or a plain Python **callable** `f(x, *params)` (resolved via [`resolve_model`](API-Models#resolve_model)). Effectively required -- the affine default exists only so `NonlineRegressor()` is constructible with no arguments (the sklearn `clone` / meta-estimator contract) |
| `var` | `"x"` | main variable name (the single input feature); a label only for a callable model |
| `param_names` | `None` | parameter names for a **callable** model, in signature order (the parameters after the leading `x`); introspected from the callable's signature when omitted, and only required when that signature cannot be introspected (a `*args` model). For a symbolic model, optional and validated against the parsed names |
| `basis` | `"auto"` | the basis to image the samples in: `"auto"` fits the candidates and keeps the lowest-residual one, `"legendre"` and `"block"` fix it. See [`fit`](API-Fitting#fit) |
| `order` | `None` | the basis order (polynomial degree for `"legendre"`, window count for `"block"`); `None` takes `fit`'s order rule for `"legendre"`/`"block"`, and is forwarded verbatim to every candidate for `basis="auto"` |
| `bounds` | `None` | per-parameter bounds; same forms as the fitters (pair list, partial `{name: (lo, hi)}` dict, scipy tuple). Fully finite bounds enable a global search |
| `p0` | `None` | initial guess: positional or `{name: value}` dict, passed through to `fit` |
| `random_state` | `0` | seed for the deterministic global / differential-evolution search used when `bounds` are given, so a bounded fit is reproducible under `GridSearchCV` / `clone`; `None` uses the global RNG |
| `robust` | `False` | Huber-reweight the image built from the samples |
| `nan_policy` | `"raise"` | `"omit"` drops non-finite `(x, y)` pairs before fitting (accepted in `X` / `y` then) |

Tuning the basis and the order is the point of exposing them:

```python
from sklearn.model_selection import GridSearchCV
from dtfit.sklearn import NonlineRegressor

search = GridSearchCV(
    NonlineRegressor("a*exp(b*x)", "x"),
    {"basis": ["legendre", "block"], "order": [4, 6, 12]},
    cv=5,
)
search.fit(x.reshape(-1, 1), y)   # a 2-D column of the single feature
```

DSB is not one of the routes: it is the exact-balance reference method and
lives in [`dtfit.reference`](Methods-DSB), where the polynomial pre-fit is two
explicit lines.

## Fitted attributes (after `fit`)

| attribute | meaning |
|---|---|
| `coef_` | fitted coefficients (ordered by sorted parameter name) |
| `model_` | callable model at the fitted coefficients |
| `result_` | the full [`FittingResult`](API-Types) -- `cov`, `stderr()`, `confidence_intervals()`, `converged`, and the fit-quality stats [`rsquared` / `aic` / `bic`](API-Types#fit-quality-diagnostics-v03) (from the LSI / EAC routes) are all reachable from the sklearn route |
| `n_features_in_` | number of input features (always 1) |

Samples are sorted by `x` before dispatch (the integral fitters need a monotone
axis), so fit results are sample-order invariant. Fitted estimators pickle
cleanly (the lambdified `model_` is rebuilt from `result_` on unpickle), so
`joblib`-parallel cross-validation works.

## Methods

### `fit(X, y, sample_weight=None) -> self`
Fit the model. `X` is a 1-D feature vector or a single-column 2-D array (a bare
1-D vector is promoted to one column; DataFrames keep their column name).
**Exactly one feature** is supported -- multi-feature `X` raises `ValueError`.

`sample_weight` (the scikit-learn convention) is optional per-sample weights,
translated to the image's per-sample `sigma = 1 / sqrt(sample_weight)` and
forwarded with `absolute_sigma=False` (relative weights), so a down-weighted
sample pulls the fit less without being dropped. A weight of `0` effectively
ignores its sample (a huge `sigma`); negative or all-zero weights raise. It
applies in every basis, as does a **callable** model.

### `predict(X) -> ndarray`
Evaluate the fitted model. Broadcasts a constant model to `X`'s shape.

### `score(X, y) -> float`
Inherited from `RegressorMixin` -- the R^2 of the prediction.

## Examples

Plain use:

```python
import numpy as np
from dtfit.sklearn import NonlineRegressor

x = np.linspace(0, 4, 200)
y = 1.4 * np.exp(0.8 * x) + np.random.default_rng(0).normal(0, 0.15, x.size)

reg = NonlineRegressor("a*exp(b*x)", "x").fit(x, y)
print(reg.coef_)          # [a, b]
print(reg.score(x, y))    # R^2
```

Tuning the basis and the order in a grid search:

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

cross_val_score(NonlineRegressor("a*exp(b*x)", "x"), x, y, cv=4)

GridSearchCV(
    NonlineRegressor("a*exp(b*x)", "x"),
    {"basis": ["legendre", "block"], "order": [4, 6, 12]},
).fit(x.reshape(-1, 1), y)
```

In a pipeline (e.g. with a scaler):

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
make_pipeline(StandardScaler(), NonlineRegressor("a*exp(b*x)", "x")).fit(x.reshape(-1, 1), y)
```

## Notes

- `get_params` / `set_params` expose every constructor argument, so all of them
  are accepted by `GridSearchCV`; `basis` and `order` are the two worth tuning.
- For uncertainty (`stderr`, confidence intervals, prediction bands), use
  `reg.result_` -- the full [`FittingResult`](API-Types) is stored on fit.
