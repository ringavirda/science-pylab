# Accuracy validation corpus

The single source of truth for *what good looks like* across the whole model
catalogue. Every catalogued family is pinned to a known-ground-truth scenario and
exercised across a noise sweep, so a method can no longer look good merely by
being tested on a hand-picked, method-favourable dataset.

## Layout

| file | role |
|---|---|
| `scenarios.py` | one ground-truth `Scenario` per catalogue family (true params, domain, metric) |
| `harness.py` | shared scoring + fitting (`metrics_for`, `param_err`, `r2`, `curve_fit_baseline`) |
| `make_golden.py` | regenerates `golden_baseline.json` |
| `golden_baseline.json` | checked-in accuracy snapshot guarded against regression |

## What consumes it

- **`tests/validation/test_accuracy_matrix.py`** (Phase 1) — recovery on the
  realistic self-seeded path + an "every method on every model returns a finite
  fit" guard.
- **`tests/validation/test_default_robustness.py`** (Phase 2) — bare-default
  entry points (`Model.fit`, `NonlineRegressor`, `suggest_models`) work or fail
  loudly.
- **`tests/validation/test_seed_robustness.py`** (Phase 3) — stability across
  noise realizations and a perturbed initial guess (no lucky-seed results).
- **`tests/validation/test_accuracy_regression.py`** (Phase 4) — no scenario
  drifts worse than the golden snapshot.
- **`dtfit_experimental.experiments.accuracy_explore`** — a runnable report of
  dtfit-vs-`curve_fit` recovery across the corpus (no thresholds; for inspection).

## Metric per scenario

- `metric="params"` — the parameters are identifiable; score *recovery* of the
  ground truth (relative error).
- `metric="r2"` — the family is only weakly identifiable (sums of exponentials,
  overlapping peaks, harmonic phases); score *curve quality* (R²). See the
  "Accuracy and known limits" section of the
  [API: Models](https://github.com/ringavirda/science-nonline/wiki/API-Models) wiki page.

## Regenerating the golden baseline

Every entry is the median over the five noise draws of `harness.SEEDS`, which
takes 12.2 s. Run **only** after an *intended* accuracy change, then review the
JSON diff:

```bash
cd packages/dtfit/tests
python -m accuracy.make_golden
```
