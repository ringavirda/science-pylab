# #5 — Stage-wise residual boosting

**Verdict: EXPERIMENTAL — a genuine win, but on one domain only.** The strongest
promotion candidate after #1: it *beat every baseline* on the one series whose
structure it matched. Needs a confirming second domain to clear the gate.

Source: [`../../../boosting.py`](../../../boosting.py).
Tested in: [Forecasting (4)](../04_realworld_forecasting/04_realworld_forecasting.ipynb).

## What it is

A gradient-boosting-style staging of the dt methods: fit a coarse model (e.g. an
LSI trend), then fit a second model to the **residual** (e.g. the seasonal cycle),
and sum them into an additive composite. Each stage sees what the previous stage
left behind.

```
boosted_fit(t, y, [
    dict(expr="a0 + a1*x + a2*x**2", var="x", method="lsi"),   # trend
    dict(expr="A*sin(w*x + p)",      var="x", method="lsi"),   # seasonal residual
])
```

## Measured results

Mauna Loa CO₂ (smooth rising trend + annual cycle), 20% holdout (Exp 4):

| method | RMSE | MAPE % | R² |
|---|---|---|---|
| **boosted LSI (#5)** | **3.772** | **0.84** | **0.402** |
| MLP | 4.747 | 1.06 | 0.053 |
| random walk | 8.641 | 2.01 | −2.136 |
| LSTM | 9.743 | 2.32 | −2.988 |
| ARIMA | 9.826 | 2.37 | −3.056 |

→ **Best method on CO₂**, and the only one with a positive R² — it beat the LSTM
and ARIMA decisively.

## Why it works (the mechanism)

This is the mirror image of why #2 (pluggable basis) disappointed: boosting won
because it matched the **generative structure**, not merely the basis.

1. **CO₂ is genuinely additive: `trend(t) + season(t) + noise`.** The composite
   model has exactly that shape, so each component has a clean target and there is
   no model mis-specification to fight. When the model family contains the truth,
   a structured extrapolator beats a general learner that has to *discover* the
   decomposition from limited data.

2. **Staging decouples a hard joint fit into two easy ones.** Fitting trend and
   seasonality *simultaneously* is a coupled, multi-modal optimisation (the trend
   can absorb part of the cycle and vice-versa). Fitting the trend first, then the
   cycle on the residual, makes each stage well-conditioned and near-convex —
   classic boosting logic. The trend stage removes the low-frequency energy so the
   residual is dominated by the seasonal term, which the second stage then pins
   cleanly.

3. **Extrapolation stays disciplined.** Both components are simple and bounded
   (quadratic trend + single sinusoid), so the forecast does not diverge the way a
   high-order single model or an under-determined learner can. The deep models
   here actually went *negative* R² — they overfit the short training span and
   extrapolated badly, exactly where a structured additive model is safe.

## Why it isn't promoted (yet)

The promotion gate is a **clear win on ≥2 distinct domains**, to guard against a
single fortunate fit. Boosting has one (CO₂). The LTSF benchmark was *not* a fair
second domain (those channels have no clean parametric trend+season — see
[02_pluggable_basis.md](02_pluggable_basis.md)). A fair second test would be
another genuinely additive trend+seasonal series (electricity-demand seasonality,
tidal + drift, a growth-plus-cycle economic series). **If it wins there, it should
be promoted** — it is the clearest near-miss in the suite.

## When to use

- Any signal that is genuinely a **sum of separable structures** (slow trend +
  cycle, baseline + transient, drift + oscillation), especially when you need to
  *extrapolate* rather than interpolate.
- **Not** for signals where the components interact non-additively, or where the
  residual after stage 1 has no clean structure (then stage 2 just fits noise).

## Related

- Contrast with [#2 pluggable basis](02_pluggable_basis.md): same Fourier
  vocabulary, but #5 won by matching additive structure while #2 lost trying to
  pin an unstable period.
