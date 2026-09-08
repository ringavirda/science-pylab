# API: diagnostics

Tools **specific to evaluating a fitted dtfit model** and report goodness of fit,
model-selection criteria, and residual structure. [`fit_report`](#fit_report) and
[`residual_diagnostics`](#residual_diagnostics) want a
[`FittingResult`](API-Types) specifically -- they read both `.predict(x)` **and**
`.coeffs` (for the parameter count), so a bare [`NonlineRegressor`](API-Estimator)
does not qualify: it predicts, but exposes its coefficients as `coef_` (the sklearn
convention), not `coeffs`. The [`*Display`](#displays) helpers are the estimator-
friendly path -- their `from_estimator` only needs `.predict`, so they accept a
`NonlineRegressor`. For plain scalar metrics on `(y_true, y_pred)` arrays, use
`sklearn.metrics` / `scipy.stats` directly; dtfit doesn't reship those.

```python
from dtfit.diagnostics import (fit_report, residual_diagnostics,
                               FitDisplay, ResidualsDisplay)
```

(Imported from the `dtfit.diagnostics` submodule, not the top level -- the
sklearn convention. The `*Display` classes need matplotlib:
`pip install 'dtfit[viz]'`.)

- [`fit_report`](#fit_report) -- goodness-of-fit + parsimony (R^2/RMSE/AIC/BIC)
- [`residual_diagnostics`](#residual_diagnostics) -- is there structure left in the residuals?
- [`residual_stats`](#residual_stats) -- the same residual tests from a plain residual array
- [`FitDisplay`](#displays) / [`ResidualsDisplay`](#displays) -- plots

---

<a name="fit_report"></a>
## `fit_report`

```python
fit_report(result, x, y) -> dict
```

Goodness-of-fit + parsimony report for a fitted model on `(x, y)`.

**Returns a dict with:**

| key | meaning |
|---|---|
| `n`, `n_params` | sample and parameter counts |
| `rss`, `rmse`, `r2` | residual sum of squares, root-mean-square error, R^2 |
| `aic`, `bic` | Gaussian-likelihood information criteria -- for **comparing candidate models on the same data** (lower is better) |
| `durbin_watson` | residual-autocorrelation statistic (~= 2 = none) |
| `params`, `stderr` | parameter values and standard errors -- **only when the fit carries a covariance** |

AIC/BIC make this the building block for model selection: fit several candidates,
keep the lowest IC (this is exactly what [`suggest_models`](API-Models#suggest_models)
does internally).

```python
from dtfit.diagnostics import fit_report
rep = fit_report(res, x, y)
print(f"r2={rep['r2']:.4f} rmse={rep['rmse']:.4f} aic={rep['aic']:.1f}")
```

---

<a name="residual_diagnostics"></a>
## `residual_diagnostics`

```python
residual_diagnostics(result, x, y) -> dict
```

Tests for **structure the model left behind**. A correct structured fit leaves
white-noise residuals; leftover autocorrelation means the model *class* is wrong
(e.g. a trend fit to a seasonal series).

**Returns:** `residuals` (the array), `durbin_watson`, `lag1_autocorr`,
`normality_p` (Shapiro-Wilk p-value, for `3 <= n <= 5000`), `mean`, `std`.

```python
d = residual_diagnostics(res, x, y)
if abs(d["lag1_autocorr"]) > 0.3:
    print("residuals autocorrelated - model class likely wrong (add a cycle?)")
```

Interpretation guide: `durbin_watson` near 2 and `lag1_autocorr` near 0 mean the
residuals are unstructured (good). A high `normality_p` is consistent with
Gaussian residuals. See [../guides/choosing-a-method.md Sec.4](Guides-Choosing-a-Method).

---

<a name="residual_stats"></a>
## `residual_stats`

```python
residual_stats(resid) -> dict
```

The residual-structure statistics of
[`residual_diagnostics`](#residual_diagnostics) from the residuals alone,
without a `FittingResult`. `resid` is a 1-D float array; the returned dict
holds `residuals`, `durbin_watson`, `lag1_autocorr`, `normality_p`, `mean`
and `std`, with `durbin_watson` NaN for a zero residual, `lag1_autocorr`
NaN for a constant residual or fewer than three samples, and `normality_p`
NaN outside `3 <= n <= 5000`.

[`Original.diagnostics(model, params, var=None)`](API-Fitting) is the usual
way in: it evaluates the model on the Original's own samples and hands the
residuals here, so a model and its parameters can be judged without fitting
them first.

```python
from dtfit import Original

orig = Original(x, y)
print(orig.diagnostics("a0 + a1*exp(a2*x)", [0.5, 2.0, 0.5], "x")["durbin_watson"])
```

---

<a name="displays"></a>
## `FitDisplay` and `ResidualsDisplay`

scikit-learn-style `*Display` plot helpers (require matplotlib). Each has three
entry points:

- `Display(x, y_true, y_pred, ...)` then `.plot(ax=None, ...)` -- construct + render
- `Display.from_estimator(estimator, X, y, *, ax=None, **kwargs)` -- from a fitted
  estimator (e.g. a [`NonlineRegressor`](API-Estimator)) and data
- `Display.from_predictions(x, y_true, y_pred, *, ax=None, estimator_name=None, **kwargs)`
  -- from raw predictions

**`FitDisplay`** -- observed data and the fitted curve over the single input
feature. `plot(ax=None, *, data_kwargs=None, line_kwargs=None)` lets you style the
scatter and the line.

**`ResidualsDisplay`** -- residuals (`y_true - y_pred`) against the predicted
values, for spotting heteroscedasticity or leftover trend.

```python
from dtfit import NonlineRegressor
from dtfit.diagnostics import FitDisplay, ResidualsDisplay

reg = NonlineRegressor("a*exp(b*x)", "x").fit(x, y)
FitDisplay.from_estimator(reg, x, y)
ResidualsDisplay.from_estimator(reg, x, y)
```

Each `plot` returns the display object (with the matplotlib `ax_`/artists), so you
can compose them into subplots.
