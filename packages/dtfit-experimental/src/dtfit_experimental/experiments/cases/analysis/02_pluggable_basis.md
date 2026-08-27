# #2 — Pluggable orthogonal basis (Fourier / Chebyshev / Laguerre)

**Verdict: EXPERIMENTAL — it is a *vocabulary*, not a source of predictive
power.** It lets dtfit *express* periodic/decay models cleanly, but on the tested
data it did not improve accuracy, and on the LTSF benchmark it was a net **loss**.

Source: [`../../../basis_lsi.py`](../../../basis_lsi.py),
[`_spectral.py`](../../../../../../dtfit/src/dtfit/_core/_spectral.py).
Tested in: [Forecasting (4)](../04_realworld_forecasting/04_realworld_forecasting.ipynb),
[LTSF (6)](../06_benchmark_ltsf/06_benchmark_ltsf.ipynb).

## What it is

Stock `fit_lsi` hard-codes the Legendre basis. This adaptation generalises the
spectral match to any orthogonal basis `∈ {legendre, chebyshev, fourier,
laguerre}`. The motivation is sound on paper:

- **Fourier** keeps the diagonal least-squares criterion on *periodic* domains —
  the natural basis for seasonal/oscillatory signals;
- **Laguerre** suits decay/transients on `[0, ∞)`;
- **Chebyshev** has near-minimax conditioning.

Each basis exposes the same interface (`_gemm_factors`, `model_spectrum`,
`integral_to_spectrum`, `sqrt_w`), so the solver is unchanged.

## Measured results

**Sunspots, ~11-year cycle (Exp 4)** — fit `c + A·sin(w·x + p)` on the Fourier
basis, forecast the 20% holdout:

| method | RMSE | R² |
|---|---|---|
| **Fourier-LSI (#2)** | **66.62** | **−0.593** |
| ARIMA | 46.91 | 0.210 |
| LSTM | 32.13 | 0.630 |
| random walk | 56.05 | −0.127 |

→ Fourier-LSI was beaten by ARIMA and LSTM, and even landed *below* the random
walk on R².

**LTSF (Exp 6)** — adding a data-driven, energy-gated Fourier seasonal term to the
trend helped **only the cleanest periodic series (electricity, +6%)** and **hurt
everywhere else** (ETTh1 short-horizon −26%). Scored **`ltsf: loss`** in the
matrix.

## Why it disappoints (the deeper reasoning)

The key distinction is between **expressiveness** and **identifiability /
predictive power**. The basis controls the *form* dtfit can write down; it does
not create information that isn't in the data window.

1. **A basis cannot pin a parameter the data doesn't constrain.** A Fourier basis
   makes "offset + one harmonic" expressible, but recovering the *right* frequency
   and phase still requires the data to constrain them. On sunspots the cycle is
   quasi-periodic with a wandering period; a single fixed harmonic fit on the
   training span locks in one period/phase that is already wrong by the holdout —
   so it extrapolates confidently in the wrong place (hence R² below random walk).

2. **Phase error compounds with horizon.** A period estimated from a finite window
   has uncertainty `~1/(window length)`. Continue that sinusoid forward and the
   phase error grows *linearly* with the number of extrapolated cycles. On the
   LTSF 96-point lookback, a harmonic continued over 96–720 steps drifts out of
   phase with the true future, so the "seasonal" term subtracts signal and adds
   error. It helped electricity only because that load series has a strong, clean
   daily cycle that *is* tightly pinned within 96 points.

3. **Orthogonality is on the criterion, not the data.** The basis's diagonal
   criterion is exact for the *continuous* inner product; on finite, noisy,
   non-uniformly-sampled real data the diagonality is approximate, eroding the
   theoretical advantage that motivated the swap.

In short: the basis is the right tool for **writing a model whose periodicity is
genuinely stable and observable**; it is the wrong tool for *discovering* a
period that the data only weakly determines — which is exactly the forecasting
regime where learned models (LSTM/ARIMA) that pool the whole history win.

## When it would actually pay off

- A signal with a **known, fixed** period (e.g. mains frequency, a calibrated
  rotation) — supply the period, fit amplitude/phase, and Fourier-LSI is both
  natural and well-conditioned.
- **Laguerre** for a transient whose decay is genuinely exponential on `[0,∞)` —
  not yet tested, a fair candidate for future evaluation.
- It is genuinely useful as the *representation* layer (the empirical Fourier
  spectrum), independent of whether the parametric extrapolation wins.

## Related

- The LTSF reframe that quantified the failure is in
  [the LTSF report](../06_benchmark_ltsf/06_benchmark_ltsf.ipynb); the gap to deep models there
  is global structure, not noise.
- Contrast with [#5 boosting](05_stagewise_boosting.md), which *did* win on a
  trend+season series — because it matched the **additive generative structure**,
  not just the basis.
