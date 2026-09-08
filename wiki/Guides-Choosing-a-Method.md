# Choosing a method, a model, and the knobs

A practical decision guide. If you just want *something that works*, the very
short answer is: **call `fit(model, data, basis="auto")` or `Model.fit` (to
recover parameters) or `auto_forecast` (to forecast)** and let dtfit route
for you ([api/auto.md](API-Auto)). The rest of this page is for when you
want to choose deliberately.

---

## 0. A deterministic curve, or a random series?

The first split decides everything else: does your series follow a
**deterministic law** `y = f(t; theta)` (a decay, growth curve, oscillation,
saturating rise), or is it genuinely **random** -- asset returns, interest rates,
river levels -- with no smooth curve to fit?

- **Random series** -> use the **stochastic tier**. `fit_stochastic(y)` detects the
  regime (long-memory / mean-reversion / GARCH volatility / stochastic cycle /
  trend+cycle), forecasts it on a rolling backtest, and can `simulate` fresh paths;
  `StochasticFilter` is the per-sample streaming twin. Fitting a deterministic
  curve to a martingale is a category error -- you would just fit the noise. See
  [api/stochastic.md](API-Stochastic) and
  [methods-explained.md#stochastic](Guides-Methods-Explained#stochastic).
- **Deterministic curve** -> continue below.

---

## 1. Which fitting method?

```
Is the data arriving live / do the parameters change over time?
|
+- YES > use ImageFilter (a STREAMING filter)
|        +- signal oscillates (a cycle)         > LSIFilter   (Legendre spectrum)
|        +- signal is monotone / saturating      > EACFilter   (block sums, cheaper)
|
+- NO (you have the whole batch) > use a BATCH method
         +- you want one reliable default        > LSI            (fit_lsi)
         +- data is very noisy / few parameters
         |   / a transient or saturating shape    > EAC           (fit_eac)
         +- a sinusoid / clear cycle               > LSI oscillatory recipe
         |                                           (fit_lsi(..., freq_param="w"))
         +- outliers / glitches present            > robust image (robust=True on
                                                      fit_lsi / fit_eac), on whichever
                                                      basis the shape chose
```

**DSB** is not in this tree on purpose: it is a reference/derivation tool, not a
production fitter (see [methods-explained.md#dsb](Guides-Methods-Explained#dsb)).

### Rules of thumb

- **Start with LSI.** It's the accurate general default and handles most smooth,
  nonlinear-in-parameters models.
- **Switch to EAC for a jump or regime change, at very high order, or when
  you need speed** and the model has few (2-4) parameters. EAC is ~5x faster
  than LSI; it is not more noise-robust -- robustness comes from
  `robust=True`, not the basis.
- **Use the oscillatory recipe for anything with a cycle.** A plain fit
  erases cycles; you must pass `freq_param`/`oscillatory=True`.
- **Use the robust image when outliers/glitches contaminate a record.**
  `robust=True` on `fit_lsi` / `fit_eac` Huber-reweights the image before any
  model is fit -- no scale to tune. For a *densely* contaminated record use
  the robust image on the global (Legendre) basis: a contiguous burst fills
  whole block windows the per-window reweighting cannot isolate, so
  `fit_lsi(..., robust=True)` beats `fit_eac(..., robust=True)` there.
- **One image, several models.** Build the image once --
  `Original(x, y).image("legendre", order)` -- and `fit` each candidate on it;
  the fits are exact in the span and cost no further data pass.
- **Streaming with dropouts?** Use `filter.coast(...)` / `coast_cov` to
  dead-reckon through measurement gaps (the uncertainty band grows with the gap)
  instead of freezing or diverging. For several streams pooled into one fault
  test, sum their `nis_` (a `FilterBank` doing this over many streams lives in
  `dtfit-experimental` for the experiment harnesses; see
  [../experimental/adaptations-api.md](Experimental-Adaptations-API)).

---

## 2. Do I even need to write a model string?

No, if a catalog family fits -- use [`dtfit.models`](API-Models):

```python
from dtfit import models

fit = models.logistic().fit(x, y)          # self-seeds p0/bounds from the data
fit = (models.linear() + models.sine()).fit(x, y)   # compose trend + cycle
```

And if you don't know which family:

```python
from dtfit import suggest_models
for s in suggest_models(x, y)[:3]:
    print(s.name, round(s.r2, 4), round(s.aic, 1))   # ranked best-first by AIC
```

The catalog is grouped by shape -- trend, growth, decay, sigmoid, saturating,
peak, oscillatory -- so you pick *structure*, not a formula. Full list in
[api/models.md](API-Models).

---

## 3. Which scaling backend?

Only relevant for large or many-channel data ([api/scaling.md](API-Scaling)):

| Situation | Tool |
|---|---|
| Many **independent** fits (different series/models) | `fit_many` (process/thread fan-out) |
| A dataset **too big for memory**, one pass | `ImageStream` accumulator (folds each chunk into a fixed-size image) |
| **Distributed** workers, then combine | the same `ImageStream` accumulators, `.merge()`d (contiguous chunks on a uniform grid, or `grid="explicit"`) |
| Many **channels on a shared x-grid**, fit at once | `ImageStream(channels=B)` (one GEMM per chunk, over one shared Gram) |
| A **long-running stream**, blocks over time | `ImageStream(block=...)` (block images, `assemble` onto a coarse domain) |

---

## 4. Is my fit any good?

Always check ([api/diagnostics.md](API-Diagnostics)):

```python
from dtfit.diagnostics import fit_report, residual_diagnostics
rep = fit_report(fit, x, y)              # r2, rmse, aic, bic, durbin_watson
diag = residual_diagnostics(fit, x, y)   # leftover autocorrelation / normality
```

- **`r2` near 1, low `rmse`** -> the curve fits.
- **`durbin_watson` ~= 2** -> residuals are white noise (good). Far from 2 ->
  leftover structure, meaning the *model class is wrong* (e.g. you fit a trend to
  a seasonal series). Try adding a cycle (`+ models.sine()`) or `suggest_models`.
- **`fit.stderr()` / `fit.confidence_intervals()`** -> parameter uncertainty,
  when the method produced a covariance.

---

## 5. The pre-flight checklist

Before trusting any fit:

1. **Watch for the coverage warning.** `fit` warns when the image's order
   doesn't cover the model's parameter sensitivities to within 2% (`coverage`);
   raise `order` when it warns. `order_for` is the default, chosen for exactly
   this.
2. **Raise the order for an oscillatory model.** A low order erases a cycle;
   pass `oscillatory=True` (or `freq_param`, which implies it) to raise the
   order to `osc_order` when that resolves the cycle better than `order_for`.
3. **Seed oscillations.** Always pass `freq_param` for sinusoids.
4. **Give bounds for hard models.** Bounds switch LSI to a bounded solve and
   put a global search behind it when the local one comes out poor.
5. **Read `residual_diagnostics`.** If the residuals aren't white, the parameters
   you recovered are answering the wrong question.

For the why behind each of these, see [methods-explained.md](Guides-Methods-Explained)
and the per-method [../methods/](Methods) references.
