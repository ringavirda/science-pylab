# dtfit examples

Self-contained, runnable scripts that walk the public API top to bottom. Each one
is plain ASCII, prints its results (no plots required), and runs headless:

```
pip install -e packages/dtfit          # or: pip install dtfit
python packages/dtfit/examples/01_quickstart.py
```

| Script | Covers |
|--------|--------|
| `01_quickstart.py` | first fit with `fit_lsi`, `FittingResult` (params / stderr / CI / prediction band), `auto_estimate` |
| `02_fitting_methods.py` | the three batch fitters: `fit_lsi` (`k_star`, oscillatory recipe with `freq_param`), `fit_eac` (`robust=True`, the block preset on a transient), `fit_dsb` (+ `find_degree`) |
| `03_models_and_auto.py` | the model catalog, self-seeding `Model.fit`, composition with `+`, `suggest_models`, `auto_estimate` / `auto_forecast` |
| `04_sklearn_estimator.py` | `NonlineRegressor` with `fit` / `predict` / `score`, `GridSearchCV`, `cross_val_score` |
| `05_streaming.py` | `EACFilter` (+ `.tracking()` / `.robust()` presets), `LSIFilter` (streaming amplitude + frequency), `result()`, pooled `nis_` fused detection |
| `06_scaling.py` | `fit_many` (+ `FittingProblem`) fanned across CPU cores, `ImageStream` one-pass accumulation and map-reduce `merge()` |
| `07_diagnostics.py` | `fit_report`, `residual_diagnostics`, `to_dict` / `from_dict`, opt-in logging, and (with `--plot`) the `*Display` plots |
| `08_stochastic.py` | `fit_stochastic` / `StochasticModel` (regime, forecast bands, `simulate` round-trip), `StochasticFilter` online tracking, the `Stochastic` model wrapper |

The only example with an optional dependency is `07_diagnostics.py --plot`, which
needs the `viz` extra (matplotlib): `pip install "dtfit[viz]"`.

These scripts are executed in CI (`tests/test_examples.py`), so they stay in step
with the API. The wiki renders them as the "Examples" guide pages.
